from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import wave

from .gemini_transcriber import (
    CaptionEvent,
    MAX_TRANSCRIPT_SEGMENTS,
    PCM_WIDTH_BYTES,
    TARGET_SAMPLE_RATE,
    TranscriptPatchEvent,
    TranscriptRevisionEvent,
    TranscriptSegment,
    TranscriptionEvent,
)


@dataclass(frozen=True)
class DiagnosticTranscriptSegment:
    id: str
    text: str
    state: str
    created_at: int
    emitted_at: int


class DiagnosticTranscriptDocument:
    """Mirror the browser's bounded atomic transcript state."""

    def __init__(self, *, max_segments: int = MAX_TRANSCRIPT_SEGMENTS) -> None:
        self._max_segments = max_segments
        self._segments: list[DiagnosticTranscriptSegment | None] = []
        self._revision = 0
        self._legacy_turn = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def segments(self) -> tuple[DiagnosticTranscriptSegment, ...]:
        return tuple(item for item in self._segments if item is not None)

    @property
    def stable_text(self) -> str:
        return " ".join(
            item.text.strip()
            for item in self.segments
            if item.state == "caption" and item.text.strip()
        )

    @property
    def visible_text(self) -> str:
        return " ".join(
            item.text.strip()
            for item in self.segments
            if item.text.strip()
        )

    def apply(self, event: TranscriptionEvent, *, emitted_at: int) -> bool:
        if isinstance(event, TranscriptPatchEvent):
            return self._apply_patch(event, emitted_at=emitted_at)
        if isinstance(event, TranscriptRevisionEvent):
            return self._apply_revision(event, emitted_at=emitted_at)
        return self._apply_caption(event, emitted_at=emitted_at)

    def _apply_patch(
        self,
        event: TranscriptPatchEvent,
        *,
        emitted_at: int,
    ) -> bool:
        if (
            event.revision <= self._revision
            or event.drop_from_start < 0
            or event.drop_from_start > len(self._segments)
            or event.replace_from < 0
            or event.total < 0
            or event.total > self._max_segments
            or len(event.segments) > self._max_segments
            or event.replace_from + len(event.segments) != event.total
        ):
            return False
        after_drop = self._segments[event.drop_from_start:]
        if event.replace_from > len(after_drop):
            return False
        replacements = [
            self._diagnostic_segment(segment, emitted_at=emitted_at)
            for segment in event.segments
        ]
        updated = after_drop[:event.replace_from] + replacements
        if len(updated) != event.total:
            return False
        self._segments = updated
        self._revision = event.revision
        return True

    def _apply_revision(
        self,
        event: TranscriptRevisionEvent,
        *,
        emitted_at: int,
    ) -> bool:
        if (
            event.revision < self._revision
            or event.replace_from < 0
            or event.replace_from > event.total
            or event.index < event.replace_from
            or event.total < 0
            or event.total > self._max_segments
            or (
                event.segment is not None
                and event.index >= event.total
            )
            or (
                event.segment is None
                and event.index > event.total
            )
        ):
            return False
        if event.revision > self._revision:
            prefix = self._segments[
                : min(event.replace_from, len(self._segments))
            ]
            updated: list[DiagnosticTranscriptSegment | None] = list(prefix)
            updated.extend(None for _ in range(event.total - len(updated)))
            for index in range(event.replace_from, event.total):
                updated[index] = None
        else:
            updated = list(self._segments[:event.total])
            updated.extend(None for _ in range(event.total - len(updated)))
        if event.segment is not None and event.index < event.total:
            updated[event.index] = self._diagnostic_segment(
                event.segment,
                emitted_at=emitted_at,
            )
        self._segments = updated
        self._revision = event.revision
        return True

    def _apply_caption(self, event: CaptionEvent, *, emitted_at: int) -> bool:
        text = " ".join(event.text.strip().split())
        if not text:
            return False
        state = "caption" if event.is_final else "listening"
        item = DiagnosticTranscriptSegment(
            id=f"legacy-{self._legacy_turn}",
            text=text,
            state=state,
            created_at=emitted_at,
            emitted_at=emitted_at,
        )
        if self._segments and self._segments[-1] is not None:
            previous = self._segments[-1]
            if previous.id == item.id and previous.state == "listening":
                self._segments[-1] = item
            else:
                self._segments.append(item)
        else:
            self._segments.append(item)
        if event.is_final:
            self._legacy_turn += 1
        self._segments = self._segments[-self._max_segments:]
        self._revision += 1
        return True

    @staticmethod
    def _diagnostic_segment(
        segment: TranscriptSegment,
        *,
        emitted_at: int,
    ) -> DiagnosticTranscriptSegment:
        return DiagnosticTranscriptSegment(
            id=segment.id,
            text=segment.text,
            state=segment.state,
            created_at=segment.created_at,
            emitted_at=emitted_at,
        )


class BoundedPcmWaveCapture:
    """Thread-safe, duration-bounded capture of normalized Gemini input PCM."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_seconds: float = 60.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
        sample_width: int = PCM_WIDTH_BYTES,
        channels: int = 1,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError("PCM capture max_seconds must be positive.")
        self._path = Path(path)
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._channels = channels
        self._max_bytes = int(
            max_seconds * sample_rate * sample_width * channels
        )
        self._buffer = bytearray()
        self._closed = False
        self._lock = threading.Lock()

    @property
    def captured_bytes(self) -> int:
        with self._lock:
            return len(self._buffer)

    def write(self, pcm: bytes) -> int:
        if not pcm:
            return 0
        with self._lock:
            if self._closed:
                return 0
            remaining = self._max_bytes - len(self._buffer)
            if remaining <= 0:
                return 0
            accepted = pcm[:remaining]
            self._buffer.extend(accepted)
            return len(accepted)

    def close(self) -> Path | None:
        with self._lock:
            if self._closed:
                return self._path if self._path.exists() else None
            self._closed = True
            pcm = bytes(self._buffer)
        if not pcm:
            return None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self._path), "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(self._sample_width)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(pcm)
        return self._path
