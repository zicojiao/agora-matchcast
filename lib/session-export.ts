import type {
  TranscriptMessage,
  TranscriptSegment,
} from '@/lib/captions';
import type { TranscriberId } from '@/lib/transcribers';

export const SESSION_EXPORT_SCHEMA_VERSION = 1;

export type SessionTimingKey =
  | 'tokenReadyMs'
  | 'rtcJoinedMs'
  | 'transcriberStartedMs'
  | 'mediaSignalMs'
  | 'firstCaptionMs'
  | 'firstFinalMs';

export type SessionTimings = Partial<Record<SessionTimingKey, number>>;

type CaptionEvent = {
  sequence: number;
  receivedAt: string;
  elapsedMs: number;
  messageType: TranscriptMessage['object'];
  revision: number | null;
  replaceFrom: number | null;
  dropFromStart: number | null;
  total: number | null;
  segmentIndex: number | null;
  segmentId: string;
  turnId: number | null;
  state: TranscriptSegment['state'] | '';
  text: string;
  segmentCreatedAt: number | null;
};

export type SessionCapture = {
  schemaVersion: typeof SESSION_EXPORT_SCHEMA_VERSION;
  clientSessionId: string;
  sessionId: string;
  transcriberId: TranscriberId;
  transcriptionMode: string;
  model: string;
  watchStartedAt: string;
  endedAt: string;
  endedElapsedMs: number | null;
  timings: SessionTimings;
  captionEvents: CaptionEvent[];
  nextSequence: number;
};

export function createSessionCapture(
  transcriberId: TranscriberId,
  watchStartedAt: string,
  clientSessionId: string,
): SessionCapture {
  return {
    schemaVersion: SESSION_EXPORT_SCHEMA_VERSION,
    clientSessionId,
    sessionId: '',
    transcriberId,
    transcriptionMode: '',
    model: '',
    watchStartedAt,
    endedAt: '',
    endedElapsedMs: null,
    timings: {},
    captionEvents: [],
    nextSequence: 1,
  };
}

export function markSessionTiming(
  capture: SessionCapture,
  key: SessionTimingKey,
  elapsedMs: number,
) {
  if (capture.timings[key] === undefined) {
    capture.timings[key] = Math.max(0, Math.round(elapsedMs));
  }
}

export function identifySession(
  capture: SessionCapture,
  session: {
    session_id: string;
    transcription_mode: string;
    model: string;
  },
) {
  capture.sessionId = session.session_id;
  capture.transcriptionMode = session.transcription_mode;
  capture.model = session.model;
}

export function completeSessionCapture(
  capture: SessionCapture,
  endedAt: string,
  elapsedMs: number,
) {
  if (capture.endedAt) return;
  capture.endedAt = endedAt;
  capture.endedElapsedMs = Math.max(0, Math.round(elapsedMs));
}

function noteFirstCaptionTimings(
  capture: SessionCapture,
  segments: Array<TranscriptSegment | null>,
  elapsedMs: number,
) {
  const visible = segments.filter(
    (segment): segment is TranscriptSegment => Boolean(segment?.text.trim()),
  );
  if (visible.length > 0) {
    markSessionTiming(capture, 'firstCaptionMs', elapsedMs);
  }
  if (visible.some((segment) => segment.state === 'caption')) {
    markSessionTiming(capture, 'firstFinalMs', elapsedMs);
  }
}

export function recordCaptionMessage(
  capture: SessionCapture,
  message: TranscriptMessage,
  receivedAt: string,
  elapsedMs: number,
) {
  const roundedElapsedMs = Math.max(0, Math.round(elapsedMs));
  const sequence = capture.nextSequence;
  capture.nextSequence += 1;

  if (message.object === 'matchcast.caption') {
    const state = message.final ? 'caption' : 'listening';
    noteFirstCaptionTimings(
      capture,
      [{
        id: `legacy-${message.turn_id}`,
        text: message.text,
        state,
        created_at: message.created_at,
      }],
      roundedElapsedMs,
    );
    capture.captionEvents.push({
      sequence,
      receivedAt,
      elapsedMs: roundedElapsedMs,
      messageType: message.object,
      revision: null,
      replaceFrom: null,
      dropFromStart: null,
      total: null,
      segmentIndex: null,
      segmentId: `legacy-${message.turn_id}`,
      turnId: message.turn_id,
      state,
      text: message.text,
      segmentCreatedAt: message.created_at,
    });
    return;
  }

  if (message.object === 'matchcast.transcript.segment') {
    noteFirstCaptionTimings(capture, [message.segment], roundedElapsedMs);
    capture.captionEvents.push({
      sequence,
      receivedAt,
      elapsedMs: roundedElapsedMs,
      messageType: message.object,
      revision: message.revision,
      replaceFrom: message.replace_from,
      dropFromStart: null,
      total: message.total,
      segmentIndex: message.index,
      segmentId: message.segment?.id ?? '',
      turnId: null,
      state: message.segment?.state ?? '',
      text: message.segment?.text ?? '',
      segmentCreatedAt: message.segment?.created_at ?? null,
    });
    return;
  }

  noteFirstCaptionTimings(capture, message.segments, roundedElapsedMs);
  const segments = message.segments.length > 0
    ? message.segments
    : [null];
  segments.forEach((segment, offset) => {
    capture.captionEvents.push({
      sequence,
      receivedAt,
      elapsedMs: roundedElapsedMs,
      messageType: message.object,
      revision: message.revision,
      replaceFrom: message.replace_from,
      dropFromStart: message.drop_from_start,
      total: message.total,
      segmentIndex: segment ? message.replace_from + offset : null,
      segmentId: segment?.id ?? '',
      turnId: null,
      state: segment?.state ?? '',
      text: segment?.text ?? '',
      segmentCreatedAt: segment?.created_at ?? null,
    });
  });
}

const CSV_COLUMNS = [
  'schema_version',
  'record_type',
  'client_session_id',
  'session_id',
  'transcriber_id',
  'transcription_mode',
  'model',
  'watch_started_at',
  'session_ended_at',
  'exported_at',
  'duration_ms',
  'received_at',
  'elapsed_ms',
  'event_sequence',
  'message_type',
  'revision',
  'replace_from',
  'drop_from_start',
  'total',
  'segment_index',
  'segment_id',
  'turn_id',
  'state',
  'text',
  'segment_created_at',
  'token_ready_ms',
  'rtc_joined_ms',
  'transcriber_started_ms',
  'media_signal_ms',
  'first_caption_ms',
  'first_final_ms',
] as const;

type CsvColumn = (typeof CSV_COLUMNS)[number];
type CsvRow = Partial<Record<CsvColumn, string | number | null>>;

function csvCell(value: string | number | null | undefined) {
  let text = value == null ? '' : String(value);
  if (/^[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

function sharedRow(capture: SessionCapture, exportedAt: string): CsvRow {
  return {
    schema_version: capture.schemaVersion,
    client_session_id: capture.clientSessionId,
    session_id: capture.sessionId,
    transcriber_id: capture.transcriberId,
    transcription_mode: capture.transcriptionMode,
    model: capture.model,
    watch_started_at: capture.watchStartedAt,
    session_ended_at: capture.endedAt,
    exported_at: exportedAt,
  };
}

export function sessionCaptureToCsv(
  capture: SessionCapture,
  finalSegments: TranscriptSegment[],
  exportedAt: string,
  currentElapsedMs: number,
) {
  const common = sharedRow(capture, exportedAt);
  const durationMs = capture.endedElapsedMs
    ?? Math.max(0, Math.round(currentElapsedMs));
  const rows: CsvRow[] = [{
    ...common,
    record_type: 'session_summary',
    duration_ms: durationMs,
    token_ready_ms: capture.timings.tokenReadyMs,
    rtc_joined_ms: capture.timings.rtcJoinedMs,
    transcriber_started_ms: capture.timings.transcriberStartedMs,
    media_signal_ms: capture.timings.mediaSignalMs,
    first_caption_ms: capture.timings.firstCaptionMs,
    first_final_ms: capture.timings.firstFinalMs,
  }];

  for (const event of capture.captionEvents) {
    rows.push({
      ...common,
      record_type: 'caption_update',
      received_at: event.receivedAt,
      elapsed_ms: event.elapsedMs,
      event_sequence: event.sequence,
      message_type: event.messageType,
      revision: event.revision,
      replace_from: event.replaceFrom,
      drop_from_start: event.dropFromStart,
      total: event.total,
      segment_index: event.segmentIndex,
      segment_id: event.segmentId,
      turn_id: event.turnId,
      state: event.state,
      text: event.text,
      segment_created_at: event.segmentCreatedAt,
    });
  }

  finalSegments.forEach((segment, index) => {
    rows.push({
      ...common,
      record_type: 'final_snapshot',
      segment_index: index,
      segment_id: segment.id,
      state: segment.state,
      text: segment.text,
      segment_created_at: segment.created_at,
    });
  });

  const lines = [
    CSV_COLUMNS.map(csvCell).join(','),
    ...rows.map((row) => CSV_COLUMNS.map((column) => csvCell(row[column])).join(',')),
  ];
  return `\uFEFF${lines.join('\r\n')}\r\n`;
}

export function sessionCaptureFilename(capture: SessionCapture) {
  const started = capture.watchStartedAt
    .replaceAll(':', '-')
    .replace(/\.\d{3}Z$/, 'Z');
  const sessionId = capture.sessionId || capture.clientSessionId;
  const safeSessionId = sessionId.replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 24);
  return `matchcast-${capture.transcriberId}-${started}-${safeSessionId}.csv`;
}
