import asyncio
import audioop
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging
import re
from statistics import median
import time

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
PCM_WIDTH_BYTES = 2
# Gemini Transcribe Live accepts continuous 100 ms PCM framing.
CHUNK_MS = 100
CHUNK_BYTES = TARGET_SAMPLE_RATE * PCM_WIDTH_BYTES * CHUNK_MS // 1000
PHRASE_BOUNDARY = re.compile(r"""[.!?]["')\]]*(?=\s|$)""")
MISSING_SENTENCE_SPACE = re.compile(r"""([.!?]["')\]]*)(?=[A-Za-z0-9])""")
TRANSCRIPT_WORD = re.compile(r"""[\w'-]+""", re.UNICODE)
MAX_TRANSCRIPT_SEGMENTS = 100
MAX_TRANSCRIPT_SEGMENT_CHARACTERS = 180
RECENT_TAIL_REVISION_MS = 4_000
RECENT_TAIL_MAX_WORDS = 10
RECENT_TAIL_CHARACTER_SIMILARITY = 0.68
RECENT_TAIL_TOKEN_SIMILARITY = 0.70
MANUAL_ACTIVITY_MIN_MS = 2_400
MANUAL_ACTIVITY_MAX_MS = 3_600
MANUAL_ACTIVITY_PREFIX_MS = 600
MANUAL_ACTIVITY_LOW_ENERGY_RATIO = 0.5
MANUAL_ACTIVITY_COMMIT_GRACE_SECONDS = 0.9
INCOMPLETE_DRAFT_MAX_BOUNDARIES = 3
GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS = 5_000
GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS = 6_000
GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS = 300
GEMINI_TRANSCRIBE_ACTIVITY_HANDOFF_SECONDS = 1.5
GEMINI_TRANSCRIBE_ACTIVITY_BUFFER_MAX_MS = 3_000


@dataclass(frozen=True)
class CaptionEvent:
    text: str
    is_final: bool
    language: str = "en"


@dataclass(frozen=True)
class TranscriptSegment:
    id: str
    text: str
    state: str
    created_at: int


@dataclass(frozen=True)
class TranscriptRevisionEvent:
    revision: int
    replace_from: int
    index: int
    total: int
    segment: TranscriptSegment | None


@dataclass(frozen=True)
class TranscriptPatchEvent:
    revision: int
    drop_from_start: int
    replace_from: int
    total: int
    segments: tuple[TranscriptSegment, ...]


TranscriptionEvent = CaptionEvent | TranscriptRevisionEvent | TranscriptPatchEvent


@dataclass(frozen=True)
class ActivityAudioBatch:
    current: tuple[bytes, ...] = ()
    boundary: bool = False
    next_prefix: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class ActivityHandoffResult:
    chunks: tuple[bytes, ...]
    wait_seconds: float
    acknowledged: bool = False
    timed_out: bool = False
    buffer_limited: bool = False
    stopped: bool = False


class FinalEventGeneration:
    """Let the audio sender wait for a final observed after a boundary."""

    def __init__(self) -> None:
        self._generation = 0
        self._condition = asyncio.Condition()

    @property
    def generation(self) -> int:
        return self._generation

    async def notify(self) -> None:
        async with self._condition:
            self._generation += 1
            self._condition.notify_all()

    async def wait_after(self, generation: int) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._generation > generation,
            )


async def buffer_audio_until_final(
    audio_queue: asyncio.Queue[bytes],
    final_events: FinalEventGeneration,
    *,
    after_generation: int,
    stop: asyncio.Event,
    timeout_seconds: float,
    buffer_max_ms: int,
) -> ActivityHandoffResult:
    """Buffer live audio while Gemini finishes the preceding activity."""
    started_at = time.monotonic()
    if timeout_seconds <= 0:
        return ActivityHandoffResult(chunks=(), wait_seconds=0.0)

    max_chunks = max(1, buffer_max_ms // CHUNK_MS)
    chunks: list[bytes] = []
    deadline = started_at + timeout_seconds

    while True:
        if final_events.generation > after_generation:
            return ActivityHandoffResult(
                chunks=tuple(chunks),
                wait_seconds=time.monotonic() - started_at,
                acknowledged=True,
            )
        if stop.is_set():
            return ActivityHandoffResult(
                chunks=tuple(chunks),
                wait_seconds=time.monotonic() - started_at,
                stopped=True,
            )
        if len(chunks) >= max_chunks:
            return ActivityHandoffResult(
                chunks=tuple(chunks),
                wait_seconds=time.monotonic() - started_at,
                buffer_limited=True,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ActivityHandoffResult(
                chunks=tuple(chunks),
                wait_seconds=time.monotonic() - started_at,
                timed_out=True,
            )

        final_task = asyncio.create_task(
            final_events.wait_after(after_generation),
        )
        audio_task = asyncio.create_task(audio_queue.get())
        stop_task = asyncio.create_task(stop.wait())
        tasks = (final_task, audio_task, stop_task)
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if audio_task in done:
            chunks.append(audio_task.result())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if final_task in done:
            final_task.result()
            return ActivityHandoffResult(
                chunks=tuple(chunks),
                wait_seconds=time.monotonic() - started_at,
                acknowledged=True,
            )
        if stop_task in done and stop_task.result():
            return ActivityHandoffResult(
                chunks=tuple(chunks),
                wait_seconds=time.monotonic() - started_at,
                stopped=True,
            )
        if not done:
            return ActivityHandoffResult(
                chunks=tuple(chunks),
                wait_seconds=time.monotonic() - started_at,
                timed_out=True,
            )


class ManualActivitySegmenter:
    """Find bounded boundaries and replay a short context into the next turn."""

    def __init__(
        self,
        *,
        chunk_ms: int = CHUNK_MS,
        min_ms: int = MANUAL_ACTIVITY_MIN_MS,
        max_ms: int = MANUAL_ACTIVITY_MAX_MS,
        prefix_ms: int = MANUAL_ACTIVITY_PREFIX_MS,
        low_energy_ratio: float = MANUAL_ACTIVITY_LOW_ENERGY_RATIO,
    ) -> None:
        self._chunk_ms = chunk_ms
        self._min_chunks = max(1, min_ms // chunk_ms)
        self._max_chunks = max(self._min_chunks, max_ms // chunk_ms)
        self._prefix_chunks = max(1, prefix_ms // chunk_ms)
        self._low_energy_ratio = low_energy_ratio
        self._history: deque[bytes] = deque(maxlen=self._prefix_chunks)
        self._energies: deque[int] = deque(maxlen=self._max_chunks)
        self._activity_chunks = 0

    def push(self, chunk: bytes) -> ActivityAudioBatch:
        energy = audioop.rms(chunk, PCM_WIDTH_BYTES)
        self._energies.append(energy)
        self._history.append(chunk)
        self._activity_chunks += 1

        completed_chunks = self._activity_chunks
        can_split = completed_chunks >= self._min_chunks
        must_split = completed_chunks >= self._max_chunks
        recent = tuple(self._energies)[-4:]
        baseline = median(self._energies) if self._energies else 0
        low_energy = (
            can_split
            and energy <= min(recent)
            and energy <= max(96, baseline * self._low_energy_ratio)
        )
        if low_energy or must_split:
            prefix = tuple(self._history)
            prefix_energies = tuple(self._energies)[-len(prefix):]
            # Replayed context does not count against the next activity's
            # 2.4–3.6 second window.
            self._activity_chunks = 0
            self._energies.clear()
            self._energies.extend(prefix_energies)
            return ActivityAudioBatch(
                current=(chunk,),
                boundary=True,
                next_prefix=prefix,
            )

        return ActivityAudioBatch(current=(chunk,))

    def flush(self) -> tuple[bytes, ...]:
        return ()


class RollingTranscriptDocument:
    """Maintain the canonical document behind Gemini's revisable snapshots."""

    def __init__(self, *, language: str) -> None:
        self._language = language
        self._text = ""
        self._segments: list[TranscriptSegment] = []
        self._revision = 0

    @property
    def text(self) -> str:
        return self._text

    @property
    def segments(self) -> tuple[TranscriptSegment, ...]:
        return tuple(self._segments)

    def update(
        self,
        transcript: str,
        *,
        now_ms: int | None = None,
    ) -> list[TranscriptRevisionEvent]:
        incoming = self._normalize(transcript)
        if not incoming:
            return []
        created_at = int(time.time() * 1000) if now_ms is None else now_ms
        canonical = self._merge_snapshot(incoming, now_ms=created_at)
        if canonical == self._text:
            return []

        proposed = self._segment(canonical)
        old_signature = [(item.text, item.state) for item in self._segments]
        new_signature = [(text, state) for text, state in proposed]
        if old_signature == new_signature:
            self._text = canonical
            return []

        replace_from = 0
        shared = min(len(old_signature), len(new_signature))
        while (
            replace_from < shared
            and old_signature[replace_from] == new_signature[replace_from]
        ):
            replace_from += 1

        self._revision += 1
        unchanged = self._segments[:replace_from]
        changed = [
            TranscriptSegment(
                id=f"r{self._revision}-s{index}",
                text=text,
                state=state,
                created_at=created_at,
            )
            for index, (text, state) in enumerate(
                proposed[replace_from:],
                start=replace_from,
            )
        ]
        self._segments = unchanged + changed
        self._text = canonical
        total = len(self._segments)
        if not changed:
            return [
                TranscriptRevisionEvent(
                    revision=self._revision,
                    replace_from=replace_from,
                    index=replace_from,
                    total=total,
                    segment=None,
                )
            ]
        return [
            TranscriptRevisionEvent(
                revision=self._revision,
                replace_from=replace_from,
                index=index,
                total=total,
                segment=segment,
            )
            for index, segment in enumerate(changed, start=replace_from)
        ]

    @staticmethod
    def _normalize(transcript: str) -> str:
        collapsed = " ".join(transcript.strip().split())
        return MISSING_SENTENCE_SPACE.sub(r"\1 ", collapsed)

    def _merge_snapshot(self, incoming: str, *, now_ms: int) -> str:
        current = self._text
        if not current:
            return incoming
        if incoming == current:
            return current

        current_folded = current.casefold()
        incoming_folded = incoming.casefold()
        if incoming_folded.startswith(current_folded):
            return incoming
        if current_folded.startswith(incoming_folded):
            return incoming

        if self._segments:
            tail = self._segments[-1].text
            tail_tokens = [
                token.casefold()
                for token in TRANSCRIPT_WORD.findall(tail)
            ]
            incoming_tail_tokens = [
                token.casefold()
                for token in TRANSCRIPT_WORD.findall(incoming)
            ]
            if (
                0 < len(tail_tokens) <= 4
                and incoming_tail_tokens[:len(tail_tokens)] == tail_tokens
            ):
                tail_start = current.rfind(tail)
                if tail_start >= 0:
                    return self._join(current[:tail_start], incoming)
            if self._is_recent_tail_revision(
                tail,
                incoming,
                tail_created_at=self._segments[-1].created_at,
                now_ms=now_ms,
            ):
                tail_start = current.rfind(tail)
                if tail_start >= 0:
                    return self._join(current[:tail_start], incoming)

        leading_phrase = PHRASE_BOUNDARY.search(incoming)
        if leading_phrase is not None:
            phrase = incoming[:leading_phrase.end()].strip()
            if current_folded.rstrip().endswith(phrase.casefold()):
                return self._join(current, incoming[leading_phrase.end():])

        current_words = list(TRANSCRIPT_WORD.finditer(current))
        incoming_words = list(TRANSCRIPT_WORD.finditer(incoming))
        current_tokens = [match.group(0).casefold() for match in current_words]
        incoming_tokens = [match.group(0).casefold() for match in incoming_words]
        matcher = SequenceMatcher(
            None,
            current_tokens,
            incoming_tokens,
            autojunk=False,
        )
        anchors = [
            block
            for block in matcher.get_matching_blocks()
            if block.size >= 4
        ]
        if anchors:
            # Prefer the earliest context in the new snapshot. Backtracking the
            # same number of words in the old document lets revised lead-in
            # words replace their previous versions as well.
            block = min(anchors, key=lambda item: (item.b, item.a))
            replace_word = max(0, block.a - block.b)
            current_anchor = current_words[replace_word].start()
            return self._join(current[:current_anchor], incoming)

        # Gemini occasionally rolls its snapshot window forward. Exact suffix
        # overlap is sufficient to preserve history without replaying the tail.
        max_overlap = min(len(current_tokens), len(incoming_tokens), 24)
        for size in range(max_overlap, 1, -1):
            if current_tokens[-size:] == incoming_tokens[:size]:
                suffix_start = incoming_words[size].start() if size < len(incoming_words) else len(incoming)
                return self._join(current, incoming[suffix_start:])

        # No shared context means this is a genuinely new utterance window.
        # Replace an abandoned listening tail instead of gluing the new speech
        # onto a sentence Gemini never completed.
        boundaries = list(PHRASE_BOUNDARY.finditer(current))
        if boundaries and current[boundaries[-1].end():].strip():
            return self._join(current[:boundaries[-1].end()], incoming)
        if not boundaries:
            return incoming
        return self._join(current, incoming)

    @staticmethod
    def _is_recent_tail_revision(
        tail: str,
        incoming: str,
        *,
        tail_created_at: int,
        now_ms: int,
    ) -> bool:
        """Recognize a fast correction of the most recent short caption."""
        age_ms = now_ms - tail_created_at
        if age_ms < 0 or age_ms > RECENT_TAIL_REVISION_MS:
            return False

        tail_tokens = [
            token.casefold()
            for token in TRANSCRIPT_WORD.findall(tail)
        ]
        incoming_tokens = [
            token.casefold()
            for token in TRANSCRIPT_WORD.findall(incoming)
        ]
        if (
            not tail_tokens
            or not incoming_tokens
            or len(tail_tokens) > RECENT_TAIL_MAX_WORDS
        ):
            return False

        character_similarity = SequenceMatcher(
            None,
            tail.casefold(),
            incoming.casefold(),
            autojunk=False,
        ).ratio()
        comparison_size = min(len(tail_tokens), len(incoming_tokens))
        token_similarity = SequenceMatcher(
            None,
            tail_tokens[:comparison_size],
            incoming_tokens[:comparison_size],
            autojunk=False,
        ).ratio()
        return (
            character_similarity >= RECENT_TAIL_CHARACTER_SIMILARITY
            or token_similarity >= RECENT_TAIL_TOKEN_SIMILARITY
        )

    @staticmethod
    def _join(prefix: str, suffix: str) -> str:
        return " ".join(part.strip() for part in (prefix, suffix) if part.strip())

    def _segment(self, transcript: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        cursor = 0
        for match in PHRASE_BOUNDARY.finditer(transcript):
            text = transcript[cursor:match.end()].strip()
            if text:
                segments.extend(self._bounded(text, "caption"))
            cursor = match.end()
        trailing = transcript[cursor:].strip()
        if trailing:
            segments.extend(self._bounded(trailing, "listening"))
        collapsed: list[tuple[str, str]] = []
        for text, state in segments:
            if (
                collapsed
                and collapsed[-1][0].casefold() == text.casefold()
            ):
                if collapsed[-1][1] == "listening" and state == "caption":
                    collapsed[-1] = (text, state)
                continue
            collapsed.append((text, state))
        return collapsed[-MAX_TRANSCRIPT_SEGMENTS:]

    @staticmethod
    def _bounded(text: str, state: str) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        remaining = text
        while len(remaining) > MAX_TRANSCRIPT_SEGMENT_CHARACTERS:
            boundary = remaining.rfind(
                " ",
                0,
                MAX_TRANSCRIPT_SEGMENT_CHARACTERS + 1,
            )
            if boundary < MAX_TRANSCRIPT_SEGMENT_CHARACTERS // 2:
                boundary = MAX_TRANSCRIPT_SEGMENT_CHARACTERS
            parts.append((remaining[:boundary].strip(), "caption"))
            remaining = remaining[boundary:].strip()
        if remaining:
            parts.append((remaining, state))
        return parts


class StableTranscriptDocument:
    """Keep committed turns immutable and expose one mutable draft."""

    def __init__(self, *, language: str) -> None:
        self._language = language
        self._committed: list[TranscriptSegment] = []
        self._draft_text = ""
        self._draft_boundaries = 0
        self._revision = 0

    @property
    def segments(self) -> tuple[TranscriptSegment, ...]:
        return tuple(self._committed)

    @property
    def draft_text(self) -> str:
        return self._draft_text

    def preview(
        self,
        transcript: str,
        *,
        now_ms: int | None = None,
    ) -> TranscriptPatchEvent | None:
        text = self._prepare_turn_text(transcript)
        if text == self._draft_text:
            return None
        self._draft_text = text
        created_at = int(time.time() * 1000) if now_ms is None else now_ms
        drop_from_start = max(
            0,
            len(self._committed) - MAX_TRANSCRIPT_SEGMENTS + 1,
        )
        if drop_from_start:
            self._committed = self._committed[drop_from_start:]
        self._revision += 1
        replacement = (
            (
                TranscriptSegment(
                    id=f"r{self._revision}-draft",
                    text=text,
                    state="listening",
                    created_at=created_at,
                ),
            )
            if text
            else ()
        )
        return TranscriptPatchEvent(
            revision=self._revision,
            drop_from_start=drop_from_start,
            replace_from=len(self._committed),
            total=len(self._committed) + len(replacement),
            segments=replacement,
        )

    def commit(
        self,
        transcript: str,
        *,
        now_ms: int | None = None,
        force: bool = True,
    ) -> TranscriptPatchEvent | None:
        text = self._prepare_turn_text(transcript)
        previous_draft = self._draft_text
        created_at = int(time.time() * 1000) if now_ms is None else now_ms
        pieces, trailing = self._split_turn(text)
        if force and trailing:
            pieces.extend(
                item
                for item, _state in RollingTranscriptDocument._bounded(
                    trailing,
                    "caption",
                )
            )
            trailing = ""
        elif trailing:
            self._draft_boundaries = (
                1
                if pieces
                else self._draft_boundaries + 1
            )
            if (
                self._draft_boundaries
                >= INCOMPLETE_DRAFT_MAX_BOUNDARIES
            ):
                pieces.extend(
                    item
                    for item, _state in RollingTranscriptDocument._bounded(
                        trailing,
                        "caption",
                    )
                )
                trailing = ""
        if not trailing:
            self._draft_boundaries = 0
        self._draft_text = trailing
        pieces = self._dedupe_adjacent_pieces(pieces)
        previous_caption_key = (
            self._spoken_key(pieces[-1])
            if pieces
            else self._spoken_key(self._committed[-1].text)
            if self._committed
            else ""
        )
        if (
            trailing
            and self._spoken_key(trailing) == previous_caption_key
        ):
            trailing = ""
            self._draft_text = ""
            self._draft_boundaries = 0
        replace_from = len(self._committed)
        appended = [
            TranscriptSegment(
                id="",
                text=piece,
                state="caption",
                created_at=created_at,
            )
            for index, piece in enumerate(pieces)
        ]
        changed = bool(appended) or trailing != previous_draft
        if not changed:
            return None
        self._revision += 1
        appended = [
            TranscriptSegment(
                id=f"r{self._revision}-s{replace_from + index}",
                text=segment.text,
                state=segment.state,
                created_at=segment.created_at,
            )
            for index, segment in enumerate(appended)
        ]
        self._committed.extend(appended)
        replacement: list[TranscriptSegment] = list(appended)
        if trailing:
            replacement.append(
                TranscriptSegment(
                    id=f"r{self._revision}-draft",
                    text=trailing,
                    state="listening",
                    created_at=created_at,
                )
            )
        drop_from_start = max(
            0,
            len(self._committed)
            + (1 if trailing else 0)
            - MAX_TRANSCRIPT_SEGMENTS,
        )
        if drop_from_start:
            self._committed = self._committed[drop_from_start:]
            replace_from -= drop_from_start
            replace_from = max(0, replace_from)
        return TranscriptPatchEvent(
            revision=self._revision,
            drop_from_start=drop_from_start,
            replace_from=replace_from,
            total=len(self._committed) + (1 if trailing else 0),
            segments=tuple(replacement),
        )

    def _prepare_turn_text(self, transcript: str) -> str:
        incoming = RollingTranscriptDocument._normalize(transcript)
        if not incoming or not self._committed:
            return incoming
        recent = " ".join(segment.text for segment in self._committed[-3:])
        recent_words = list(TRANSCRIPT_WORD.finditer(recent))
        incoming_words = list(TRANSCRIPT_WORD.finditer(incoming))
        recent_tokens = [
            match.group(0).casefold()
            for match in recent_words
        ]
        incoming_tokens = [
            match.group(0).casefold()
            for match in incoming_words
        ]
        max_overlap = min(len(recent_tokens), len(incoming_tokens), 18)
        for size in range(max_overlap, 2, -1):
            if recent_tokens[-size:] != incoming_tokens[:size]:
                continue
            if size >= len(incoming_words):
                return ""
            return incoming[incoming_words[size].start():].strip()
        return incoming

    def _dedupe_adjacent_pieces(self, pieces: list[str]) -> list[str]:
        """Drop only exact adjacent repeats without revising stable history."""
        previous_key = (
            self._caption_key(self._committed[-1].text)
            if self._committed
            else ""
        )
        deduplicated: list[str] = []
        for piece in pieces:
            key = self._caption_key(piece)
            if not key or key == previous_key:
                continue
            deduplicated.append(piece)
            previous_key = key
        return deduplicated

    @staticmethod
    def _caption_key(text: str) -> str:
        return " ".join(text.split()).casefold()

    @staticmethod
    def _spoken_key(text: str) -> str:
        return " ".join(
            token.casefold()
            for token in TRANSCRIPT_WORD.findall(text)
        )

    @staticmethod
    def _split_turn(text: str) -> tuple[list[str], str]:
        if not text:
            return [], ""
        pieces: list[str] = []
        cursor = 0
        for match in PHRASE_BOUNDARY.finditer(text):
            phrase = text[cursor:match.end()].strip()
            if phrase:
                pieces.extend(
                    item
                    for item, _state in RollingTranscriptDocument._bounded(
                        phrase,
                        "caption",
                    )
                )
            cursor = match.end()
        trailing = text[cursor:].strip()
        return pieces, trailing


class StableActivityTranscript:
    """Keep an incomplete tail across manually bounded Gemini activities."""

    def __init__(self, *, language: str) -> None:
        self._language = language
        self._document = StableTranscriptDocument(language=language)
        self._activity = RollingTranscriptDocument(language=language)
        self._carried_draft = ""

    @property
    def document(self) -> StableTranscriptDocument:
        return self._document

    @property
    def activity_text(self) -> str:
        return self._activity.text

    @property
    def carried_draft(self) -> str:
        return self._carried_draft

    def update(self, transcript: str) -> TranscriptPatchEvent | None:
        changed = self._activity.update(transcript)
        if not changed:
            return None
        return self._document.preview(self._combined_text())

    def finalize(
        self,
        transcript: str,
        *,
        force: bool = False,
    ) -> tuple[TranscriptPatchEvent | None, TranscriptPatchEvent | None]:
        # input_transcription is Gemini's authoritative final snapshot for the
        # activity. Replace stale interim hypotheses before committing it.
        self._activity = RollingTranscriptDocument(language=self._language)
        self._activity.update(transcript)
        preview = self._document.preview(self._combined_text())
        committed = self.commit(force=force)
        return preview, committed

    def commit(self, *, force: bool) -> TranscriptPatchEvent | None:
        patch = self._document.commit(
            self._combined_text(),
            force=force,
        )
        self._carried_draft = self._document.draft_text
        self._activity = RollingTranscriptDocument(language=self._language)
        return patch

    def _combined_text(self) -> str:
        return merge_activity_context(
            self._carried_draft,
            self._activity.text,
        )


class GeminiTranscriptDocument:
    """Append Gemini finals without revising previously committed speech."""

    def __init__(self, *, language: str) -> None:
        self._language = language
        self._committed: list[TranscriptSegment] = []
        self._draft_text = ""
        self._last_final_text = ""
        self._revision = 0
        self._raw_final_words = 0
        self._committed_final_words = 0
        self._overlap_words_removed = 0

    @property
    def segments(self) -> tuple[TranscriptSegment, ...]:
        return tuple(self._committed)

    @property
    def draft_text(self) -> str:
        return self._draft_text

    @property
    def raw_final_words(self) -> int:
        return self._raw_final_words

    @property
    def committed_final_words(self) -> int:
        return self._committed_final_words

    @property
    def overlap_words_removed(self) -> int:
        return self._overlap_words_removed

    def preview(
        self,
        transcript: str,
        *,
        now_ms: int | None = None,
    ) -> TranscriptPatchEvent | None:
        text, _overlap = self._trim_replayed_prefix(transcript)
        if text == self._draft_text:
            return None
        self._draft_text = text
        created_at = int(time.time() * 1000) if now_ms is None else now_ms
        drop_from_start = max(
            0,
            len(self._committed) + (1 if text else 0)
            - MAX_TRANSCRIPT_SEGMENTS,
        )
        if drop_from_start:
            self._committed = self._committed[drop_from_start:]
        self._revision += 1
        replacement = (
            (
                TranscriptSegment(
                    id=f"g{self._revision}-draft",
                    text=text,
                    state="listening",
                    created_at=created_at,
                ),
            )
            if text
            else ()
        )
        return TranscriptPatchEvent(
            revision=self._revision,
            drop_from_start=drop_from_start,
            replace_from=len(self._committed),
            total=len(self._committed) + len(replacement),
            segments=replacement,
        )

    def commit(
        self,
        transcript: str,
        *,
        now_ms: int | None = None,
    ) -> TranscriptPatchEvent | None:
        normalized = RollingTranscriptDocument._normalize(transcript)
        self._raw_final_words += len(TRANSCRIPT_WORD.findall(normalized))
        text, overlap_words = self._trim_replayed_prefix(normalized)
        self._overlap_words_removed += overlap_words
        return self._append_final(
            text,
            now_ms=now_ms,
            count_committed=True,
        )

    def finalize_draft(
        self,
        *,
        now_ms: int | None = None,
    ) -> TranscriptPatchEvent | None:
        """Preserve the last interim only when the live session is closing."""
        return self._append_final(
            self._draft_text,
            now_ms=now_ms,
            count_committed=False,
        )

    def _append_final(
        self,
        text: str,
        *,
        now_ms: int | None,
        count_committed: bool,
    ) -> TranscriptPatchEvent | None:
        previous_draft = self._draft_text
        self._draft_text = ""
        pieces = self._caption_pieces(text)
        if not pieces and not previous_draft:
            return None
        created_at = int(time.time() * 1000) if now_ms is None else now_ms
        replace_from = len(self._committed)
        self._revision += 1
        appended = [
            TranscriptSegment(
                id=f"g{self._revision}-s{replace_from + index}",
                text=piece,
                state="caption",
                created_at=created_at,
            )
            for index, piece in enumerate(pieces)
        ]
        self._committed.extend(appended)
        if text:
            self._last_final_text = text
            if count_committed:
                self._committed_final_words += len(
                    TRANSCRIPT_WORD.findall(text)
                )
        drop_from_start = max(
            0,
            len(self._committed) - MAX_TRANSCRIPT_SEGMENTS,
        )
        if drop_from_start:
            self._committed = self._committed[drop_from_start:]
            replace_from = max(0, replace_from - drop_from_start)
        return TranscriptPatchEvent(
            revision=self._revision,
            drop_from_start=drop_from_start,
            replace_from=replace_from,
            total=len(self._committed),
            segments=tuple(appended),
        )

    def _trim_replayed_prefix(self, transcript: str) -> tuple[str, int]:
        incoming = RollingTranscriptDocument._normalize(transcript)
        previous = self._last_final_text
        if not incoming or not previous:
            return incoming, 0
        previous_words = list(TRANSCRIPT_WORD.finditer(previous))
        incoming_words = list(TRANSCRIPT_WORD.finditer(incoming))
        previous_tokens = [
            item.group(0).casefold()
            for item in previous_words
        ]
        incoming_tokens = [
            item.group(0).casefold()
            for item in incoming_words
        ]
        max_overlap = min(
            len(previous_tokens),
            len(incoming_tokens),
            12,
        )
        for size in range(max_overlap, 0, -1):
            if previous_tokens[-size:] != incoming_tokens[:size]:
                continue
            exact_duplicate = (
                size == len(previous_tokens) == len(incoming_tokens)
            )
            # One repeated word can be intentional speech. Only trim it when
            # the entire final is an exact replay; otherwise require two words.
            if size < 2 and not exact_duplicate:
                continue
            if size == len(incoming_words):
                return "", size
            return incoming[incoming_words[size].start():].strip(), size
        return incoming, 0

    @staticmethod
    def _caption_pieces(text: str) -> list[str]:
        if not text:
            return []
        pieces: list[str] = []
        cursor = 0
        for match in PHRASE_BOUNDARY.finditer(text):
            phrase = text[cursor:match.end()].strip()
            if phrase:
                pieces.extend(
                    item
                    for item, _state in RollingTranscriptDocument._bounded(
                        phrase,
                        "caption",
                    )
                )
            cursor = match.end()
        trailing = text[cursor:].strip()
        if trailing:
            pieces.extend(
                item
                for item, _state in RollingTranscriptDocument._bounded(
                    trailing,
                    "caption",
                )
            )
        return pieces


def merge_activity_context(carried: str, current: str) -> str:
    """Reconcile a previous draft with a context-overlapped new activity."""
    prefix = RollingTranscriptDocument._normalize(carried)
    incoming = RollingTranscriptDocument._normalize(current)
    if not prefix:
        return incoming
    if not incoming:
        return prefix
    prefix_words = list(TRANSCRIPT_WORD.finditer(prefix))
    incoming_words = list(TRANSCRIPT_WORD.finditer(incoming))
    prefix_tokens = [item.group(0).casefold() for item in prefix_words]
    incoming_tokens = [item.group(0).casefold() for item in incoming_words]
    max_overlap = min(len(prefix_tokens), len(incoming_tokens), 12)
    for size in range(max_overlap, 0, -1):
        if prefix_tokens[-size:] != incoming_tokens[:size]:
            continue
        suffix_start = (
            incoming_words[size].start()
            if size < len(incoming_words)
            else len(incoming)
        )
        return RollingTranscriptDocument._join(
            prefix,
            incoming[suffix_start:],
        )

    # The replayed context can revise a misheard short tail. A distinctive
    # first word anchors that correction even when the surrounding words
    # changed (for example "Faker maybe" -> "Faker made the play").
    if incoming_tokens and len(incoming_tokens[0]) >= 4:
        start = max(0, len(prefix_tokens) - 8)
        for index in range(len(prefix_tokens) - 1, start - 1, -1):
            if prefix_tokens[index] != incoming_tokens[0]:
                continue
            return RollingTranscriptDocument._join(
                prefix[:prefix_words[index].start()],
                incoming,
            )
    return RollingTranscriptDocument._join(prefix, incoming)


class Pcm16Normalizer:
    """Convert Agora PCM frames to Gemini's 16 kHz mono signed PCM stream."""

    def __init__(self) -> None:
        self._rate_state: tuple | None = None
        self._buffer = bytearray()

    def push(self, pcm: bytes, *, sample_rate: int, channels: int) -> list[bytes]:
        if not pcm or sample_rate <= 0 or channels <= 0:
            return []
        mono = pcm
        if channels == 2:
            mono = audioop.tomono(pcm, PCM_WIDTH_BYTES, 0.5, 0.5)
        elif channels != 1:
            raise ValueError(f"Unsupported PCM channel count: {channels}")
        if sample_rate != TARGET_SAMPLE_RATE:
            mono, self._rate_state = audioop.ratecv(
                mono,
                PCM_WIDTH_BYTES,
                1,
                sample_rate,
                TARGET_SAMPLE_RATE,
                self._rate_state,
            )
        self._buffer.extend(mono)
        chunks: list[bytes] = []
        while len(self._buffer) >= CHUNK_BYTES:
            chunks.append(bytes(self._buffer[:CHUNK_BYTES]))
            del self._buffer[:CHUNK_BYTES]
        return chunks


def _read_text(value: object | None) -> str:
    text = getattr(value, "text", None)
    return text.strip() if isinstance(text, str) else ""


def caption_events_from_response(response: object, language: str) -> list[CaptionEvent]:
    """Normalize current and EAP Live API response shapes at one boundary."""
    content = getattr(response, "server_content", None)
    if content is None:
        return []
    events: list[CaptionEvent] = []
    interim = _read_text(getattr(content, "interim_input_transcription", None))
    final = _read_text(getattr(content, "input_transcription", None))
    if interim:
        events.append(CaptionEvent(interim, False, language))
    if final:
        # The EAP contract uses input_transcription itself as the finalized
        # event. Its optional `finished` field is not a turn-finality flag and
        # is false in current responses; the official EAP demo likewise treats
        # every inputTranscription message as final.
        events.append(CaptionEvent(final, True, language))
    return events


def build_audio_transcription_config(
    types_module,
    *,
    language: str,
    vocabulary: list[str],
    dedicated_transcribe: bool,
    vocabulary_mode: str = "custom",
    transcription_mode: str = "smart",
):
    """Build transcription options without coupling tests to google-genai."""
    if not dedicated_transcribe:
        return types_module.AudioTranscriptionConfig()
    options: dict[str, object] = {
        "mode": transcription_mode.upper(),
    }
    selected_vocabulary = vocabulary[:1000]
    if selected_vocabulary and vocabulary_mode == "custom":
        options["custom_vocabulary"] = selected_vocabulary
    normalized_language = language.strip()
    if normalized_language.casefold() == "auto":
        # The launch API removed `language_auto`. Automatic multi-language
        # identification is requested by sending an empty code list.
        options["language_codes"] = []
    elif normalized_language:
        options["language_codes"] = [normalized_language]
    config_type = types_module.AudioTranscriptionConfig
    model_fields = getattr(config_type, "model_fields", None)
    if model_fields is not None and "mode" not in model_fields:
        raise RuntimeError(
            "The installed google-genai SDK does not support "
            "AudioTranscriptionConfig.mode. Upgrade google-genai to a "
            "Gemini 3.5 Transcribe launch-compatible version."
        )
    try:
        return config_type(**options)
    except Exception as exc:
        if "mode" in str(exc).casefold():
            raise RuntimeError(
                "The installed google-genai SDK rejected "
                "AudioTranscriptionConfig.mode. Upgrade google-genai to a "
                "Gemini 3.5 Transcribe launch-compatible version."
            ) from exc
        raise


class GeminiLiveTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        language: str,
        vocabulary: list[str],
        activity_min_ms: int = GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS,
        activity_max_ms: int = GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS,
        activity_prefix_ms: int = GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS,
        activity_low_energy_ratio: float = MANUAL_ACTIVITY_LOW_ENERGY_RATIO,
        activity_commit_grace_seconds: float = (
            MANUAL_ACTIVITY_COMMIT_GRACE_SECONDS
        ),
        activity_handoff_seconds: float = (
            GEMINI_TRANSCRIBE_ACTIVITY_HANDOFF_SECONDS
        ),
        activity_buffer_max_ms: int = (
            GEMINI_TRANSCRIBE_ACTIVITY_BUFFER_MAX_MS
        ),
        vocabulary_mode: str = "custom",
        transcription_mode: str = "smart",
        raw_event_callback: (
            Callable[[CaptionEvent], Awaitable[None]] | None
        ) = None,
        manual_endpointing: bool = True,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language
        self._vocabulary = vocabulary
        self._activity_min_ms = activity_min_ms
        self._activity_max_ms = activity_max_ms
        self._activity_prefix_ms = activity_prefix_ms
        self._activity_low_energy_ratio = activity_low_energy_ratio
        self._activity_commit_grace_seconds = activity_commit_grace_seconds
        self._activity_handoff_seconds = max(0.0, activity_handoff_seconds)
        self._activity_buffer_max_ms = max(
            CHUNK_MS,
            activity_buffer_max_ms,
        )
        self._vocabulary_mode = vocabulary_mode
        self._transcription_mode = transcription_mode
        self._raw_event_callback = raw_event_callback
        self._manual_endpointing = manual_endpointing
        self._raw_final_words = 0
        self._committed_final_words = 0
        self._overlap_words_removed = 0
        self._activity_boundaries = 0
        self._activity_final_acks = 0
        self._activity_handoff_timeouts = 0
        self._activity_handoff_buffer_limits = 0
        self._activity_handoff_peak_buffer_ms = 0
        self._activity_handoff_total_wait_seconds = 0.0
        self._activity_handoff_max_wait_seconds = 0.0

    @property
    def transcript_metrics(self) -> dict[str, int | float]:
        expected_words = max(
            0,
            self._raw_final_words - self._overlap_words_removed,
        )
        retention_ratio = (
            self._committed_final_words / expected_words
            if expected_words
            else 1.0
        )
        return {
            "raw_final_words": self._raw_final_words,
            "committed_final_words": self._committed_final_words,
            "overlap_words_removed": self._overlap_words_removed,
            "final_retention_ratio": retention_ratio,
            "activity_boundaries": self._activity_boundaries,
            "activity_final_acks": self._activity_final_acks,
            "activity_handoff_timeouts": self._activity_handoff_timeouts,
            "activity_handoff_buffer_limits": (
                self._activity_handoff_buffer_limits
            ),
            "activity_handoff_peak_buffer_ms": (
                self._activity_handoff_peak_buffer_ms
            ),
            "activity_handoff_total_wait_seconds": (
                self._activity_handoff_total_wait_seconds
            ),
            "activity_handoff_max_wait_seconds": (
                self._activity_handoff_max_wait_seconds
            ),
        }

    def _capture_transcript_metrics(
        self,
        transcript: GeminiTranscriptDocument,
    ) -> None:
        self._raw_final_words = transcript.raw_final_words
        self._committed_final_words = transcript.committed_final_words
        self._overlap_words_removed = transcript.overlap_words_removed

    def _capture_handoff_metrics(
        self,
        result: ActivityHandoffResult,
    ) -> None:
        self._activity_boundaries += 1
        if result.acknowledged:
            self._activity_final_acks += 1
        if result.timed_out:
            self._activity_handoff_timeouts += 1
        if result.buffer_limited:
            self._activity_handoff_buffer_limits += 1
        self._activity_handoff_peak_buffer_ms = max(
            self._activity_handoff_peak_buffer_ms,
            len(result.chunks) * CHUNK_MS,
        )
        self._activity_handoff_total_wait_seconds += result.wait_seconds
        self._activity_handoff_max_wait_seconds = max(
            self._activity_handoff_max_wait_seconds,
            result.wait_seconds,
        )

    async def run(
        self,
        audio_queue: asyncio.Queue[bytes],
        emit: Callable[[TranscriptionEvent], Awaitable[None]],
        stop: asyncio.Event,
    ) -> None:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(api_version="v1alpha"),
        )
        is_dedicated_transcribe = "transcribe" in self._model.lower()
        transcription_config = build_audio_transcription_config(
            types,
            language=self._language,
            vocabulary=self._vocabulary,
            dedicated_transcribe=is_dedicated_transcribe,
            vocabulary_mode=self._vocabulary_mode,
            transcription_mode=self._transcription_mode,
        )
        config = types.LiveConnectConfig(
            # Native-audio Live models only accept AUDIO here. We discard their
            # generated audio and consume input_transcription. The dedicated
            # Gemini 3.5 Transcribe Live returns text directly.
            response_modalities=["TEXT" if is_dedicated_transcribe else "AUDIO"],
            input_audio_transcription=transcription_config,
            realtime_input_config=(
                types.RealtimeInputConfig(
                    automatic_activity_detection=(
                        types.AutomaticActivityDetection(disabled=True)
                    ),
                )
                if is_dedicated_transcribe and self._manual_endpointing
                else None
            ),
            system_instruction=(
                None
                if is_dedicated_transcribe
                else (
                    "Act as a passive transcription monitor. Do not answer, "
                    "comment on, translate, or summarize the incoming audio."
                )
            ),
        )
        transcript = GeminiTranscriptDocument(language=self._language)
        async with client.aio.live.connect(
            model=self._model,
            config=config,
        ) as session:
            activity_lock = asyncio.Lock()
            boundary_lock = asyncio.Lock()
            final_events = FinalEventGeneration()

            async def send_audio() -> None:
                segmenter = (
                    ManualActivitySegmenter(
                        min_ms=self._activity_min_ms,
                        max_ms=self._activity_max_ms,
                        prefix_ms=self._activity_prefix_ms,
                        low_energy_ratio=self._activity_low_energy_ratio,
                    )
                    if is_dedicated_transcribe and self._manual_endpointing
                    else None
                )
                activity_open = False
                pending_chunks: deque[bytes] = deque()

                async def start_activity() -> None:
                    nonlocal activity_open
                    if segmenter is None or activity_open:
                        return
                    await session.send_realtime_input(
                        activity_start=types.ActivityStart(),
                    )
                    activity_open = True

                async def send_chunks(chunks: tuple[bytes, ...]) -> None:
                    if not chunks:
                        return
                    await start_activity()
                    for pcm in chunks:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=pcm,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )

                while pending_chunks or not stop.is_set():
                    if pending_chunks:
                        chunk = pending_chunks.popleft()
                    else:
                        try:
                            chunk = await asyncio.wait_for(
                                audio_queue.get(),
                                timeout=0.25,
                            )
                        except asyncio.TimeoutError:
                            continue
                    if segmenter is None:
                        await send_chunks((chunk,))
                        continue
                    batch = segmenter.push(chunk)
                    await send_chunks(batch.current)
                    if batch.boundary:
                        if activity_open:
                            async with boundary_lock:
                                await session.send_realtime_input(
                                    activity_end=types.ActivityEnd(),
                                )
                                after_generation = final_events.generation
                            activity_open = False
                            result = await buffer_audio_until_final(
                                audio_queue,
                                final_events,
                                after_generation=after_generation,
                                stop=stop,
                                timeout_seconds=(
                                    self._activity_handoff_seconds
                                ),
                                buffer_max_ms=self._activity_buffer_max_ms,
                            )
                            self._capture_handoff_metrics(result)
                            if result.stopped:
                                if result.chunks:
                                    await send_chunks(
                                        batch.next_prefix + result.chunks,
                                    )
                                break
                            await send_chunks(batch.next_prefix)
                            pending_chunks.extend(result.chunks)

                if segmenter is not None:
                    await send_chunks(segmenter.flush())
                    if activity_open:
                        await session.send_realtime_input(
                            activity_end=types.ActivityEnd(),
                        )
                else:
                    await session.send_realtime_input(audio_stream_end=True)

            async def receive_captions() -> None:
                while not stop.is_set():
                    turn = session.receive()
                    async for response in turn:
                        for event in caption_events_from_response(
                            response,
                            self._language,
                        ):
                            if event.is_final:
                                async with boundary_lock:
                                    await final_events.notify()
                            if self._raw_event_callback is not None:
                                await self._raw_event_callback(event)
                            async with activity_lock:
                                if event.is_final:
                                    patch = transcript.commit(
                                        event.text,
                                    )
                                    self._capture_transcript_metrics(
                                        transcript,
                                    )
                                else:
                                    patch = transcript.preview(event.text)
                            if patch is not None:
                                await emit(patch)

            sender = asyncio.create_task(send_audio(), name="gemini-audio-sender")
            receiver = asyncio.create_task(
                receive_captions(),
                name="gemini-caption-receiver",
            )
            stop_waiter = asyncio.create_task(
                stop.wait(),
                name="gemini-caption-stop-waiter",
            )
            tasks = (sender, receiver, stop_waiter)
            try:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_waiter in done:
                    await sender
                    # Let the final manual activity boundary flush its response.
                    await asyncio.sleep(
                        self._activity_commit_grace_seconds + 0.6,
                    )
                    async with activity_lock:
                        patch = transcript.finalize_draft()
                        self._capture_transcript_metrics(transcript)
                    if patch is not None:
                        await emit(patch)
                    return
                if receiver in done:
                    await receiver
                    raise RuntimeError("Gemini Transcribe session closed")
                if sender in done:
                    await sender
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
