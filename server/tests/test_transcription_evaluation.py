import asyncio
from pathlib import Path

import pytest

from app.transcription_evaluation import (
    build_parser,
    enqueue_audio_chunk,
    normalize_words,
    summarize_timeline,
    validate_evaluation_inputs,
    vocabulary_hit_stats,
    word_error_stats,
)


def test_evaluator_defaults_to_smart_transcription():
    args = build_parser().parse_args(["clip.mp4"])

    assert args.transcription_mode == "smart"


def test_evaluator_accepts_verbatim_transcription():
    args = build_parser().parse_args(
        ["clip.mp4", "--transcription-mode", "verbatim"]
    )

    assert args.transcription_mode == "verbatim"


def test_reference_normalization_and_word_error_rate():
    assert normalize_words("Kai’Sa’s Shockwave—lands!") == [
        "kai'sa's",
        "shockwave",
        "lands",
    ]

    stats = word_error_stats(
        "Faker finds the Shockwave",
        "Faker finds Shockwave",
    )

    assert stats["reference_words"] == 4
    assert stats["edits"] == 1
    assert stats["word_error_rate"] == 0.25


def test_vocabulary_hits_use_exact_normalized_phrases():
    hits = vocabulary_hit_stats(
        "Kai'Sa takes Baron Nashor.",
        ["Kai'Sa", "Baron", "Baron Nashor", "Faker"],
        reference="Kai'Sa takes the Baron Nashor.",
    )

    assert hits == [
        {
            "term": "Kai'Sa",
            "transcript_count": 1,
            "reference_count": 1,
            "matched_expected": 1,
        },
        {
            "term": "Baron",
            "transcript_count": 1,
            "reference_count": 1,
            "matched_expected": 1,
        },
        {
            "term": "Baron Nashor",
            "transcript_count": 1,
            "reference_count": 1,
            "matched_expected": 1,
        },
    ]


def test_summary_calculates_update_cadence_and_finalization_lag():
    timeline = [
        {
            "elapsed_seconds": 0.8,
            "has_listening": True,
            "new_stable_segments": 0,
        },
        {
            "elapsed_seconds": 1.8,
            "has_listening": True,
            "new_stable_segments": 1,
        },
        {
            "elapsed_seconds": 3.2,
            "has_listening": False,
            "new_stable_segments": 1,
        },
    ]

    summary = summarize_timeline(
        timeline,
        audio_duration_seconds=3.0,
        stable_transcript="One. Two.",
        visible_transcript="One. Two.",
    )

    assert summary["first_draft_latency_seconds"] == 0.8
    assert summary["mean_draft_update_interval_seconds"] == 1.0
    assert summary["first_stable_caption_latency_seconds"] == 1.8
    assert summary["mean_stable_caption_interval_seconds"] == pytest.approx(1.4)
    assert summary["finalization_lag_seconds"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "input_path": Path("/missing.mp4"),
                "start_seconds": 0,
                "duration_seconds": 10,
                "ffmpeg_path": "/usr/bin/ffmpeg",
                "api_key": "key",
            },
            "does not exist",
        ),
    ],
)
def test_evaluator_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_evaluation_inputs(**kwargs)


def test_evaluator_requires_ffmpeg_and_api_key(tmp_path):
    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="ffmpeg"):
        validate_evaluation_inputs(
            input_path=source,
            start_seconds=0,
            duration_seconds=10,
            ffmpeg_path=None,
            api_key="key",
        )
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        validate_evaluation_inputs(
            input_path=source,
            start_seconds=0,
            duration_seconds=10,
            ffmpeg_path="/usr/bin/ffmpeg",
            api_key=None,
        )


@pytest.mark.asyncio
async def test_evaluator_stops_when_transcriber_fails_while_queue_is_full():
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    queue.put_nowait(b"queued")

    async def fail_connection() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("invalid credentials")

    transcriber_task = asyncio.create_task(fail_connection())

    with pytest.raises(RuntimeError, match="invalid credentials"):
        await asyncio.wait_for(
            enqueue_audio_chunk(queue, b"blocked", transcriber_task),
            timeout=0.2,
        )
