import assert from 'node:assert/strict';
import {
  completeSessionCapture,
  createSessionCapture,
  identifySession,
  markSessionTiming,
  recordCaptionMessage,
  sessionCaptureFilename,
  sessionCaptureToCsv,
} from '../lib/session-export';
import { scrollTranscriptToLatest } from '../lib/transcript-scroll';

const capture = createSessionCapture(
  'gemini-transcribe',
  '2026-08-10T03:44:00.000Z',
  'browser-session',
);
markSessionTiming(capture, 'tokenReadyMs', 41.7);
markSessionTiming(capture, 'rtcJoinedMs', 104.2);
markSessionTiming(capture, 'tokenReadyMs', 999);
identifySession(capture, {
  session_id: 'backend-session',
  transcription_mode: 'gemini-live',
  model: 'models/gemini-3.5-transcribe-live',
});

recordCaptionMessage(capture, {
  object: 'matchcast.transcript.patch',
  revision: 1,
  drop_from_start: 0,
  replace_from: 0,
  total: 2,
  segments: [
    {
      id: 'caption-1',
      text: 'Faker, "behind me"',
      state: 'caption',
      created_at: 1_754_795_441_000,
    },
    {
      id: 'caption-2',
      text: '=unsafe spreadsheet text',
      state: 'listening',
      created_at: 1_754_795_442_000,
    },
  ],
}, '2026-08-10T03:44:02.000Z', 2_000.4);

recordCaptionMessage(capture, {
  object: 'matchcast.transcript.segment',
  revision: 2,
  replace_from: 1,
  index: 1,
  total: 2,
  segment: {
    id: 'caption-2',
    text: 'What on earth is going on?',
    state: 'caption',
    created_at: 1_754_795_443_000,
  },
}, '2026-08-10T03:44:03.000Z', 3_000);

completeSessionCapture(capture, '2026-08-10T03:44:04.000Z', 4_000.2);
completeSessionCapture(capture, '2026-08-10T03:45:00.000Z', 60_000);

assert.equal(capture.timings.tokenReadyMs, 42);
assert.equal(capture.timings.firstCaptionMs, 2_000);
assert.equal(capture.timings.firstFinalMs, 2_000);
assert.equal(capture.captionEvents.length, 3);
assert.equal(capture.endedElapsedMs, 4_000);

const csv = sessionCaptureToCsv(
  capture,
  [
    {
      id: 'caption-1',
      text: 'Faker, "behind me"',
      state: 'caption',
      created_at: 1_754_795_441_000,
    },
    {
      id: 'caption-2',
      text: 'What on earth is going on?',
      state: 'caption',
      created_at: 1_754_795_443_000,
    },
  ],
  '2026-08-10T03:45:00.000Z',
  60_000,
);

assert.ok(csv.startsWith('\uFEFF'));
assert.match(csv, /"session_summary"/);
assert.equal((csv.match(/"caption_update"/g) ?? []).length, 3);
assert.equal((csv.match(/"final_snapshot"/g) ?? []).length, 2);
assert.match(csv, /"Faker, ""behind me"""/);
assert.match(csv, /"'=unsafe spreadsheet text"/);
assert.match(csv, /"gemini-transcribe"/);
assert.match(csv, /"models\/gemini-3.5-transcribe-live"/);
assert.match(csv, /"4000"/);
assert.equal(
  sessionCaptureFilename(capture),
  'matchcast-gemini-transcribe-2026-08-10T03-44-00Z-backend-session.csv',
);

const rail = { scrollHeight: 720, scrollTop: 115 };
scrollTranscriptToLatest(rail);
assert.equal(rail.scrollTop, 720);

console.log('Session CSV export and transcript auto-scroll contracts passed.');
