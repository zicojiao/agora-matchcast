import {
  applyTranscriptMessage,
  createTranscriptState,
  decodeTranscriptMessage,
  visibleTranscriptSegments,
  type TranscriptPatchMessage,
  type TranscriptRevisionMessage,
} from '../lib/captions';

function patch(
  revisionNumber: number,
  replaceFrom: number,
  total: number,
  segments: TranscriptPatchMessage['segments'],
  dropFromStart = 0,
): TranscriptPatchMessage {
  return {
    object: 'matchcast.transcript.patch',
    revision: revisionNumber,
    drop_from_start: dropFromStart,
    replace_from: replaceFrom,
    total,
    segments,
  };
}

function revision(
  revisionNumber: number,
  replaceFrom: number,
  index: number,
  total: number,
  text: string | null,
  state: 'caption' | 'listening' = 'listening',
): TranscriptRevisionMessage {
  return {
    object: 'matchcast.transcript.segment',
    revision: revisionNumber,
    replace_from: replaceFrom,
    index,
    total,
    segment: text === null
      ? null
      : {
          id: `r${revisionNumber}-s${index}`,
          text,
          state,
          created_at: revisionNumber,
        },
  };
}

const encoded = new TextEncoder().encode(JSON.stringify(
  revision(1, 0, 0, 1, 'Bang is hiding'),
));
const decoded = decodeTranscriptMessage(encoded);
if (!decoded || decoded.object !== 'matchcast.transcript.segment') {
  throw new Error('Revision payload decoding failed.');
}

const atomicEncoded = new TextEncoder().encode(JSON.stringify(
  patch(1, 0, 1, [{
    id: 'draft-1',
    text: 'Reckles goes down',
    state: 'listening',
    created_at: 1,
  }]),
));
const atomicDecoded = decodeTranscriptMessage(atomicEncoded);
if (!atomicDecoded || atomicDecoded.object !== 'matchcast.transcript.patch') {
  throw new Error('Atomic patch payload decoding failed.');
}
let atomicStore = applyTranscriptMessage(
  createTranscriptState(),
  atomicDecoded,
);
atomicStore = applyTranscriptMessage(
  atomicStore,
  patch(2, 0, 1, [{
    id: 'draft-2',
    text: 'Rekkles goes down to Tai.',
    state: 'listening',
    created_at: 2,
  }]),
);
atomicStore = applyTranscriptMessage(
  atomicStore,
  patch(3, 0, 1, [{
    id: 'caption-3',
    text: 'Rekkles goes down to Tai.',
    state: 'caption',
    created_at: 3,
  }]),
);
let atomicVisible = visibleTranscriptSegments(atomicStore);
if (
  atomicVisible.length !== 1
  || atomicVisible[0]?.text !== 'Rekkles goes down to Tai.'
  || atomicVisible[0]?.state !== 'caption'
) {
  throw new Error('Atomic draft commit did not replace in one transition.');
}
const stablePrefix = atomicVisible[0];
atomicStore = applyTranscriptMessage(
  atomicStore,
  patch(4, 1, 2, [{
    id: 'draft-4',
    text: 'Gets a Baron steal',
    state: 'listening',
    created_at: 4,
  }]),
);
atomicVisible = visibleTranscriptSegments(atomicStore);
if (
  atomicVisible[0] !== stablePrefix
  || atomicVisible[1]?.text !== 'Gets a Baron steal'
) {
  throw new Error('Atomic patch rewrote the committed prefix.');
}
const beforeMalformed = atomicStore;
atomicStore = applyTranscriptMessage(
  atomicStore,
  patch(5, 1, 3, [], 0),
);
if (atomicStore !== beforeMalformed) {
  throw new Error('Malformed atomic patch changed transcript state.');
}
atomicStore = applyTranscriptMessage(
  atomicStore,
  patch(6, 0, 1, [{
    id: 'caption-6',
    text: 'Gets a Baron steal!',
    state: 'caption',
    created_at: 6,
  }], 1),
);
atomicVisible = visibleTranscriptSegments(atomicStore);
if (
  atomicVisible.length !== 1
  || atomicVisible[0]?.text !== 'Gets a Baron steal!'
) {
  throw new Error('Atomic rolling-window drop failed.');
}

let store = createTranscriptState();
store = applyTranscriptMessage(store, decoded);
store = applyTranscriptMessage(
  store,
  revision(2, 0, 0, 1, "Bang is hiding. He's coming"),
);
store = applyTranscriptMessage(
  store,
  revision(
    3,
    0,
    0,
    1,
    "Bang is hiding. He's coming. Bang looking to come in",
  ),
);

let visible = visibleTranscriptSegments(store);
if (
  visible.length !== 1
  || visible[0]?.text
    !== "Bang is hiding. He's coming. Bang looking to come in"
) {
  throw new Error('Growing interim was appended instead of replaced.');
}

store = applyTranscriptMessage(
  store,
  revision(4, 0, 0, 3, 'TP!', 'caption'),
);
store = applyTranscriptMessage(
  store,
  revision(4, 0, 1, 3, 'Oh, Shockwave!', 'caption'),
);
store = applyTranscriptMessage(
  store,
  revision(
    4,
    0,
    2,
    3,
    "It lands, and that's going to force Zilean into an early Zhonya's.",
    'caption',
  ),
);
visible = visibleTranscriptSegments(store);
if (
  visible.length !== 3
  || visible[2]?.text
    !== "It lands, and that's going to force Zilean into an early Zhonya's."
) {
  throw new Error('Multi-message revision assembly failed.');
}

const beforeStale = store;
store = applyTranscriptMessage(
  store,
  revision(3, 0, 0, 1, 'Stale replay'),
);
if (store !== beforeStale) {
  throw new Error('Stale revision was not ignored.');
}

store = applyTranscriptMessage(store, revision(5, 1, 1, 1, null));
visible = visibleTranscriptSegments(store);
if (visible.length !== 1 || visible[0]?.text !== 'TP!') {
  throw new Error('Deletion-only revision failed.');
}

store = applyTranscriptMessage(
  createTranscriptState(),
  revision(12, 5, 5, 6, 'Recovered current suffix', 'listening'),
);
visible = visibleTranscriptSegments(store);
if (
  visible.length !== 1
  || visible[0]?.text !== 'Recovered current suffix'
) {
  throw new Error('Missed baseline revision recovery failed.');
}

const legacy = new TextEncoder().encode(JSON.stringify({
  object: 'matchcast.caption',
  text: 'The Shockwave',
  final: false,
  turn_id: 8,
  created_at: 8,
}));
const legacyMessage = decodeTranscriptMessage(legacy);
if (!legacyMessage) throw new Error('Legacy caption decoding failed.');
store = applyTranscriptMessage(createTranscriptState(), legacyMessage);
visible = visibleTranscriptSegments(store);
if (
  visible.length !== 1
  || visible[0]?.state !== 'listening'
  || visible[0]?.text !== 'The Shockwave'
) {
  throw new Error('Legacy caption compatibility failed.');
}

console.log('Revisable transcript protocol contract passed.');
