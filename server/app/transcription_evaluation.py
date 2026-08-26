from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
from typing import Any

from dotenv import load_dotenv

from .gemini_transcriber import (
    CaptionEvent,
    CHUNK_BYTES,
    CHUNK_MS,
    GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS,
    GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS,
    GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS,
    GEMINI_TRANSCRIBE_ACTIVITY_HANDOFF_SECONDS,
    GEMINI_TRANSCRIBE_ACTIVITY_BUFFER_MAX_MS,
    GeminiLiveTranscriber,
    TranscriptPatchEvent,
    TranscriptRevisionEvent,
    TranscriptionEvent,
)
from .transcription_diagnostics import DiagnosticTranscriptDocument
from .vocabulary import GEMINI_TRANSCRIBE_VOCABULARY, merge_vocabulary

EVALUATION_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def normalize_words(text: str) -> list[str]:
    normalized = (
        text.casefold()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    return EVALUATION_WORD.findall(normalized)


def word_error_stats(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(
            hypothesis_words,
            start=1,
        ):
            substitution = previous[hypothesis_index - 1] + (
                reference_word != hypothesis_word
            )
            deletion = previous[hypothesis_index] + 1
            insertion = current[hypothesis_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    edits = previous[-1]
    return {
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "edits": edits,
        "word_error_rate": (
            edits / len(reference_words)
            if reference_words
            else 0.0 if not hypothesis_words else 1.0
        ),
    }


def vocabulary_hit_stats(
    transcript: str,
    vocabulary: list[str],
    *,
    reference: str | None = None,
) -> list[dict[str, Any]]:
    transcript_words = normalize_words(transcript)
    reference_words = normalize_words(reference or "")

    def count_phrase(words: list[str], phrase: list[str]) -> int:
        if not phrase or len(phrase) > len(words):
            return 0
        return sum(
            words[index:index + len(phrase)] == phrase
            for index in range(len(words) - len(phrase) + 1)
        )

    results: list[dict[str, Any]] = []
    for term in vocabulary:
        phrase = normalize_words(term)
        transcript_count = count_phrase(transcript_words, phrase)
        reference_count = count_phrase(reference_words, phrase)
        if transcript_count or reference_count:
            results.append(
                {
                    "term": term,
                    "transcript_count": transcript_count,
                    "reference_count": reference_count,
                    "matched_expected": (
                        min(transcript_count, reference_count)
                        if reference is not None
                        else None
                    ),
                }
            )
    return results


def _mean_interval(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.fmean(
        later - earlier
        for earlier, later in zip(values, values[1:])
    )


def summarize_timeline(
    timeline: list[dict[str, Any]],
    *,
    audio_duration_seconds: float,
    stable_transcript: str,
    visible_transcript: str,
) -> dict[str, Any]:
    draft_times = [
        item["elapsed_seconds"]
        for item in timeline
        if item["has_listening"]
    ]
    stable_times = [
        item["elapsed_seconds"]
        for item in timeline
        if item["new_stable_segments"] > 0
    ]
    last_event = (
        timeline[-1]["elapsed_seconds"]
        if timeline
        else None
    )
    return {
        "audio_duration_seconds": audio_duration_seconds,
        "event_count": len(timeline),
        "draft_event_count": len(draft_times),
        "stable_event_count": len(stable_times),
        "first_draft_latency_seconds": draft_times[0] if draft_times else None,
        "first_stable_caption_latency_seconds": (
            stable_times[0] if stable_times else None
        ),
        "mean_draft_update_interval_seconds": _mean_interval(draft_times),
        "mean_stable_caption_interval_seconds": _mean_interval(stable_times),
        "last_event_elapsed_seconds": last_event,
        "finalization_lag_seconds": (
            max(0.0, last_event - audio_duration_seconds)
            if last_event is not None
            else None
        ),
        "stable_transcript": stable_transcript,
        "visible_transcript": visible_transcript,
    }


def event_payload(event: TranscriptionEvent) -> dict[str, Any]:
    if isinstance(event, TranscriptPatchEvent):
        return {
            "type": "patch",
            "revision": event.revision,
            "drop_from_start": event.drop_from_start,
            "replace_from": event.replace_from,
            "total": event.total,
            "segments": [asdict(segment) for segment in event.segments],
        }
    if isinstance(event, TranscriptRevisionEvent):
        return {
            "type": "revision",
            "revision": event.revision,
            "replace_from": event.replace_from,
            "index": event.index,
            "total": event.total,
            "segment": (
                asdict(event.segment)
                if event.segment is not None
                else None
            ),
        }
    return {
        "type": "caption",
        "text": event.text,
        "is_final": event.is_final,
        "language": event.language,
    }


def load_vocabulary_file(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise ValueError("Vocabulary JSON must be an array of strings.")
        return [item.strip() for item in parsed if item.strip()]
    return [
        item.strip()
        for line in raw.splitlines()
        for item in line.split(",")
        if item.strip() and not item.lstrip().startswith("#")
    ]


def validate_evaluation_inputs(
    *,
    input_path: Path,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_path: str | None,
    api_key: str | None,
) -> None:
    if not input_path.is_file():
        raise ValueError(f"Input media does not exist: {input_path}")
    if start_seconds < 0:
        raise ValueError("Start offset must be zero or greater.")
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive.")
    if not ffmpeg_path:
        raise ValueError("ffmpeg is required but was not found on PATH.")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required.")


class EvaluationRecorder:
    def __init__(self, *, started_at: float) -> None:
        self._started_at = started_at
        self.document = DiagnosticTranscriptDocument()
        self.timeline: list[dict[str, Any]] = []
        self.raw_timeline: list[dict[str, Any]] = []

    async def record_raw(self, event: CaptionEvent) -> None:
        self.raw_timeline.append(
            {
                "elapsed_seconds": round(
                    time.monotonic() - self._started_at,
                    4,
                ),
                "text": event.text,
                "is_final": event.is_final,
                "language": event.language,
            }
        )

    async def emit(self, event: TranscriptionEvent) -> None:
        elapsed = time.monotonic() - self._started_at
        emitted_at = int(time.time() * 1000)
        stable_before = sum(
            item.state == "caption"
            for item in self.document.segments
        )
        applied = self.document.apply(event, emitted_at=emitted_at)
        stable_after = sum(
            item.state == "caption"
            for item in self.document.segments
        )
        payload = event_payload(event)
        payload.update(
            {
                "elapsed_seconds": round(elapsed, 4),
                "emitted_at": emitted_at,
                "applied": applied,
                "new_stable_segments": max(0, stable_after - stable_before),
                "has_listening": any(
                    item.state == "listening"
                    for item in self.document.segments
                ),
                "document_text": self.document.visible_text,
            }
        )
        self.timeline.append(payload)


async def enqueue_audio_chunk(
    queue: asyncio.Queue[bytes],
    chunk: bytes,
    transcriber_task: asyncio.Task[None],
) -> None:
    """Stop feeding promptly when the live connection has already failed."""
    put_task = asyncio.create_task(queue.put(chunk))
    done, _pending = await asyncio.wait(
        (put_task, transcriber_task),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if transcriber_task in done:
        if not put_task.done():
            put_task.cancel()
        await asyncio.gather(put_task, return_exceptions=True)
        await transcriber_task
    await put_task


async def evaluate_live_transcription(
    *,
    input_path: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    language: str,
    vocabulary: list[str],
    start_seconds: float,
    duration_seconds: float,
    activity_min_ms: int,
    activity_max_ms: int,
    activity_prefix_ms: int,
    activity_low_energy_ratio: float,
    activity_commit_grace_seconds: float,
    activity_handoff_seconds: float,
    activity_buffer_max_ms: int,
    vocabulary_mode: str,
    transcription_mode: str,
    manual_endpointing: bool,
    reference: str | None,
    ffmpeg_path: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
    stop = asyncio.Event()
    started_at = time.monotonic()
    recorder = EvaluationRecorder(started_at=started_at)
    transcriber = GeminiLiveTranscriber(
        api_key=api_key,
        model=model,
        language=language,
        vocabulary=vocabulary,
        activity_min_ms=activity_min_ms,
        activity_max_ms=activity_max_ms,
        activity_prefix_ms=activity_prefix_ms,
        activity_low_energy_ratio=activity_low_energy_ratio,
        activity_commit_grace_seconds=activity_commit_grace_seconds,
        activity_handoff_seconds=activity_handoff_seconds,
        activity_buffer_max_ms=activity_buffer_max_ms,
        vocabulary_mode=vocabulary_mode,
        transcription_mode=transcription_mode,
        raw_event_callback=recorder.record_raw,
        manual_endpointing=manual_endpointing,
    )
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-i",
        str(input_path),
        "-t",
        str(duration_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    transcriber_task = asyncio.create_task(
        transcriber.run(queue, recorder.emit, stop),
        name="evaluation-gemini-transcriber",
    )
    chunks_sent = 0
    failure: BaseException | None = None
    try:
        while True:
            chunk = await process.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            if len(chunk) < CHUNK_BYTES:
                break
            target = started_at + chunks_sent * (CHUNK_MS / 1000)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await enqueue_audio_chunk(
                queue,
                chunk,
                transcriber_task,
            )
            chunks_sent += 1
        return_code = await process.wait()
        if return_code != 0:
            stderr = (
                await process.stderr.read()
                if process.stderr is not None
                else b""
            )
            raise RuntimeError(
                f"ffmpeg failed ({return_code}): "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
        while not queue.empty():
            if transcriber_task.done():
                await transcriber_task
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.15)
        stop.set()
        await asyncio.wait_for(transcriber_task, timeout=10)
    except BaseException as exc:
        failure = exc
        stop.set()
        if process.returncode is None:
            process.terminate()
            await process.wait()
        if not transcriber_task.done():
            transcriber_task.cancel()
        await asyncio.gather(transcriber_task, return_exceptions=True)
    audio_duration = chunks_sent * CHUNK_MS / 1000
    summary = summarize_timeline(
        recorder.timeline,
        audio_duration_seconds=audio_duration,
        stable_transcript=recorder.document.stable_text,
        visible_transcript=recorder.document.visible_text,
    )
    summary["gemini_final_retention"] = transcriber.transcript_metrics
    summary["audio_chunks_sent"] = chunks_sent
    summary["audio_bytes_sent"] = chunks_sent * CHUNK_BYTES
    summary["config"] = {
        "input": str(input_path),
        "start_seconds": start_seconds,
        "requested_duration_seconds": duration_seconds,
        "model": model,
        "language": language,
        "vocabulary": vocabulary,
        "activity_min_ms": activity_min_ms,
        "activity_max_ms": activity_max_ms,
        "activity_prefix_ms": activity_prefix_ms,
        "activity_low_energy_ratio": activity_low_energy_ratio,
        "activity_commit_grace_seconds": activity_commit_grace_seconds,
        "activity_handoff_seconds": activity_handoff_seconds,
        "activity_buffer_max_ms": activity_buffer_max_ms,
        "vocabulary_mode": vocabulary_mode,
        "transcription_mode": transcription_mode,
        "manual_endpointing": manual_endpointing,
    }
    if reference is not None:
        summary["accuracy"] = word_error_stats(
            reference,
            recorder.document.stable_text,
        )
    summary["vocabulary_hits"] = vocabulary_hit_stats(
        recorder.document.stable_text,
        vocabulary,
        reference=reference,
    )
    if failure is not None:
        summary["error"] = f"{type(failure).__name__}: {failure}"
    (output_dir / "events.json").write_text(
        json.dumps(recorder.timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "raw-events.json").write_text(
        json.dumps(recorder.raw_timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "transcript.txt").write_text(
        recorder.document.stable_text + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failure is not None:
        raise failure
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feed a media interval to the production Gemini Live transcriber "
            "at real-time speed and write diagnostic artifacts."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=float, default=212.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        default=None,
    )
    parser.add_argument("--language", default=None)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--vocabulary-file", type=Path)
    parser.add_argument(
        "--vocabulary-term",
        action="append",
        default=[],
    )
    parser.add_argument("--no-default-vocabulary", action="store_true")
    parser.add_argument(
        "--activity-min-ms",
        type=int,
        default=GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS,
    )
    parser.add_argument(
        "--activity-max-ms",
        type=int,
        default=GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS,
    )
    parser.add_argument(
        "--activity-prefix-ms",
        type=int,
        default=GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS,
    )
    parser.add_argument("--activity-low-energy-ratio", type=float, default=0.5)
    parser.add_argument(
        "--activity-commit-grace-seconds",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--activity-handoff-seconds",
        type=float,
        default=GEMINI_TRANSCRIBE_ACTIVITY_HANDOFF_SECONDS,
    )
    parser.add_argument(
        "--activity-buffer-max-ms",
        type=int,
        default=GEMINI_TRANSCRIBE_ACTIVITY_BUFFER_MAX_MS,
    )
    parser.add_argument(
        "--vocabulary-mode",
        choices=("custom", "none"),
        default="custom",
    )
    parser.add_argument(
        "--transcription-mode",
        choices=("smart", "verbatim"),
        default=os.getenv("GEMINI_TRANSCRIPTION_MODE", "smart").lower(),
    )
    parser.add_argument(
        "--automatic-vad",
        action="store_true",
        help="Use Gemini automatic activity detection instead of manual turns.",
    )
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[1] / "evaluations" / timestamp


def main(argv: list[str] | None = None) -> int:
    server_dir = Path(__file__).resolve().parents[1]
    repo_dir = server_dir.parent
    load_dotenv(server_dir / ".env")
    load_dotenv(server_dir / ".env.local")
    load_dotenv(repo_dir / ".env.local")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    args = build_parser().parse_args(arguments)
    ffmpeg_path = shutil.which("ffmpeg")
    api_key = os.getenv("GEMINI_API_KEY")
    validate_evaluation_inputs(
        input_path=args.input,
        start_seconds=args.start,
        duration_seconds=args.duration,
        ffmpeg_path=ffmpeg_path,
        api_key=api_key,
    )
    vocabulary = [] if args.no_default_vocabulary else list(
        GEMINI_TRANSCRIBE_VOCABULARY
    )
    if args.vocabulary_file:
        vocabulary = merge_vocabulary(
            vocabulary,
            load_vocabulary_file(args.vocabulary_file),
        )
    vocabulary = merge_vocabulary(vocabulary, args.vocabulary_term)
    reference = (
        args.reference.read_text(encoding="utf-8")
        if args.reference
        else None
    )
    output_dir = args.output_dir or default_output_dir()
    summary = asyncio.run(
        evaluate_live_transcription(
            input_path=args.input,
            output_dir=output_dir,
            api_key=api_key or "",
            model=args.model
            or os.getenv(
                "GEMINI_MODEL",
                "models/gemini-3.5-transcribe-live",
            ),
            language=args.language or os.getenv("GEMINI_LANGUAGE", "en-US"),
            vocabulary=vocabulary,
            start_seconds=args.start,
            duration_seconds=args.duration,
            activity_min_ms=args.activity_min_ms,
            activity_max_ms=args.activity_max_ms,
            activity_prefix_ms=args.activity_prefix_ms,
            activity_low_energy_ratio=args.activity_low_energy_ratio,
            activity_commit_grace_seconds=(
                args.activity_commit_grace_seconds
            ),
            activity_handoff_seconds=args.activity_handoff_seconds,
            activity_buffer_max_ms=args.activity_buffer_max_ms,
            vocabulary_mode=args.vocabulary_mode,
            transcription_mode=args.transcription_mode,
            manual_endpointing=not args.automatic_vad,
            reference=reference,
            ffmpeg_path=ffmpeg_path or "ffmpeg",
        )
    )
    print(f"Evaluation artifacts: {output_dir}")
    print(
        json.dumps(
            {
                "first_draft_latency_seconds": summary[
                    "first_draft_latency_seconds"
                ],
                "first_stable_caption_latency_seconds": summary[
                    "first_stable_caption_latency_seconds"
                ],
                "stable_transcript": summary["stable_transcript"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
