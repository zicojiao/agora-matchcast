export type Caption = {
  text: string;
  final: boolean;
  turn_id: number;
  created_at: number;
};

export type TranscriptSegment = {
  id: string;
  text: string;
  state: 'caption' | 'listening';
  created_at: number;
};

export type TranscriptRevisionMessage = {
  object: 'matchcast.transcript.segment';
  revision: number;
  replace_from: number;
  index: number;
  total: number;
  segment: TranscriptSegment | null;
};

export type TranscriptPatchMessage = {
  object: 'matchcast.transcript.patch';
  revision: number;
  drop_from_start: number;
  replace_from: number;
  total: number;
  segments: TranscriptSegment[];
};

type LegacyCaptionMessage = Caption & {
  object: 'matchcast.caption';
};

export type TranscriptMessage =
  | TranscriptPatchMessage
  | TranscriptRevisionMessage
  | LegacyCaptionMessage;

export type TranscriptState = {
  revision: number;
  segments: Array<TranscriptSegment | null>;
  legacy: Caption[];
};

export const MAX_TRANSCRIPT_ENTRIES = 100;

export function createTranscriptState(): TranscriptState {
  return {
    revision: 0,
    segments: [],
    legacy: [],
  };
}

function isTranscriptSegment(value: unknown): value is TranscriptSegment {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<TranscriptSegment>;
  return (
    typeof candidate.id === 'string'
    && typeof candidate.text === 'string'
    && (candidate.state === 'caption' || candidate.state === 'listening')
    && typeof candidate.created_at === 'number'
  );
}

export function decodeTranscriptMessage(
  data: Uint8Array,
): TranscriptMessage | null {
  try {
    const payload = JSON.parse(new TextDecoder().decode(data));
    if (
      payload?.object === 'matchcast.transcript.patch'
      && Number.isInteger(payload.revision)
      && Number.isInteger(payload.drop_from_start)
      && Number.isInteger(payload.replace_from)
      && Number.isInteger(payload.total)
      && Array.isArray(payload.segments)
      && payload.segments.every(isTranscriptSegment)
    ) {
      return payload as TranscriptPatchMessage;
    }
    if (
      payload?.object === 'matchcast.transcript.segment'
      && Number.isInteger(payload.revision)
      && Number.isInteger(payload.replace_from)
      && Number.isInteger(payload.index)
      && Number.isInteger(payload.total)
      && (payload.segment === null || isTranscriptSegment(payload.segment))
    ) {
      return payload as TranscriptRevisionMessage;
    }
    if (
      payload?.object === 'matchcast.caption'
      && typeof payload.text === 'string'
      && typeof payload.final === 'boolean'
      && Number.isInteger(payload.turn_id)
      && typeof payload.created_at === 'number'
    ) {
      return payload as LegacyCaptionMessage;
    }
    return null;
  } catch {
    return null;
  }
}

function normalized(text: string) {
  return text.trim().replace(/\s+/g, ' ').toLocaleLowerCase();
}

function compactEnglish(text: string) {
  return normalized(text).replace(/[^a-z0-9]+/g, '');
}

function isShortTailCorrection(previous: Caption, current: Caption) {
  const currentWords = normalized(current.text).split(' ');
  const currentCompact = compactEnglish(current.text);
  return (
    currentWords.length <= 3
    && currentCompact.length >= 5
    && compactEnglish(previous.text).endsWith(currentCompact)
  );
}

function collapseAdjacentFinalDuplicates(items: Caption[]) {
  const collapsed: Caption[] = [];
  for (const item of items) {
    const previous = collapsed.at(-1);
    if (
      item.final
      && previous?.final
      && normalized(previous.text) === normalized(item.text)
    ) {
      collapsed[collapsed.length - 1] = item;
    } else if (
      item.final
      && previous?.final
      && isShortTailCorrection(previous, item)
    ) {
      continue;
    } else {
      collapsed.push(item);
    }
  }
  return collapsed;
}

function reconcileLegacyCaption(items: Caption[], next: Caption) {
  const text = next.text.trim();
  if (!text) return items;

  const incoming = { ...next, text };
  const latest = items.at(-1);
  if (
    latest
    && normalized(latest.text) === normalized(incoming.text)
    && latest.final === incoming.final
  ) {
    return items;
  }

  const sameTurnInterimIndex = items.findLastIndex(
    (item) => item.turn_id === incoming.turn_id && !item.final,
  );
  const updated = [...items];
  if (sameTurnInterimIndex >= 0) {
    updated[sameTurnInterimIndex] = incoming;
  } else {
    updated.push(incoming);
  }
  return collapseAdjacentFinalDuplicates(updated).slice(
    -MAX_TRANSCRIPT_ENTRIES,
  );
}

export function applyTranscriptMessage(
  state: TranscriptState,
  message: TranscriptMessage,
): TranscriptState {
  if (message.object === 'matchcast.caption') {
    const legacy = reconcileLegacyCaption(state.legacy, message);
    return {
      revision: state.revision,
      legacy,
      segments: legacy.map((caption) => ({
        id: `legacy-${caption.turn_id}`,
        text: caption.text,
        state: caption.final ? 'caption' : 'listening',
        created_at: caption.created_at,
      })),
    };
  }

  if (message.object === 'matchcast.transcript.patch') {
    if (
      message.revision <= state.revision
      || message.drop_from_start < 0
      || message.drop_from_start > state.segments.length
      || message.replace_from < 0
      || message.total < 0
      || message.total > MAX_TRANSCRIPT_ENTRIES
      || message.segments.length > MAX_TRANSCRIPT_ENTRIES
      || message.replace_from + message.segments.length !== message.total
    ) {
      return state;
    }
    const afterDrop = state.segments.slice(message.drop_from_start);
    if (message.replace_from > afterDrop.length) {
      return state;
    }
    const segments = [
      ...afterDrop.slice(0, message.replace_from),
      ...message.segments,
    ];
    if (segments.length !== message.total) {
      return state;
    }
    return {
      revision: message.revision,
      segments,
      legacy: [],
    };
  }

  if (
    message.revision < state.revision
    || message.replace_from < 0
    || message.replace_from > message.total
    || message.index < message.replace_from
    || (message.segment !== null && message.index >= message.total)
    || (message.segment === null && message.index > message.total)
    || message.total < 0
    || message.total > MAX_TRANSCRIPT_ENTRIES
  ) {
    return state;
  }

  let segments: Array<TranscriptSegment | null>;
  if (message.revision > state.revision) {
    // A viewer can miss the baseline revision during startup or a transient
    // data-channel interruption. Preserve known prefix rows, leave unknown
    // indexes empty, and render the received suffix immediately instead of
    // rejecting every future revision forever.
    segments = state.segments.slice(
      0,
      Math.min(message.replace_from, state.segments.length),
    );
    segments.length = message.total;
    for (let index = 0; index < message.total; index += 1) {
      if (segments[index] === undefined) segments[index] = null;
    }
    segments.fill(null, message.replace_from);
  } else {
    segments = [...state.segments];
    segments.length = message.total;
  }

  if (message.segment !== null && message.index < message.total) {
    segments[message.index] = message.segment;
  }
  return {
    revision: message.revision,
    segments,
    legacy: [],
  };
}

export function visibleTranscriptSegments(state: TranscriptState) {
  return state.segments.filter(
    (segment): segment is TranscriptSegment => segment != null,
  );
}
