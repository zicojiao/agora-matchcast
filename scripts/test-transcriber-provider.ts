import assert from 'node:assert/strict';
import {
  TRANSCRIBERS,
  availableTranscribers,
  isAllowedTranscriberId,
  isPublicDemoMode,
  isTranscriberId,
} from '../lib/transcribers';

assert.deepEqual(
  TRANSCRIBERS.map((transcriber) => transcriber.value),
  ['gemini-transcribe'],
);
assert.equal(
  TRANSCRIBERS.find(
    (transcriber) => transcriber.value === 'gemini-transcribe',
  )?.label,
  'gemini-3.5-transcribe-live',
);
assert.deepEqual(
  availableTranscribers(true).map((transcriber) => transcriber.value),
  ['gemini-transcribe'],
);
assert.equal(isPublicDemoMode('TRUE'), true);
assert.equal(isPublicDemoMode('false'), false);
assert.equal(isTranscriberId('gemini-transcribe'), true);
assert.equal(isTranscriberId('unknown'), false);
assert.equal(isTranscriberId(undefined), false);
assert.equal(isAllowedTranscriberId('gemini-transcribe', true), true);
assert.equal(isAllowedTranscriberId('unknown', false), false);

console.log('Transcriber selection contract passed.');
