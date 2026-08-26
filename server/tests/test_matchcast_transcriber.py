import asyncio
import json

from app.gemini_transcriber import (
    CaptionEvent,
    GeminiLiveTranscriber,
    TranscriptPatchEvent,
    TranscriptRevisionEvent,
    TranscriptSegment,
)
from app.matchcast_transcriber import (
    _RemoteAudioObserver,
    _gemini_transcriber_type_for_mode,
    caption_payload,
    transcript_patch_payload,
    transcript_revision_payload,
)


def test_caption_payload_is_small_and_browser_readable():
    payload = caption_payload(
        CaptionEvent("The Shockwave catches them all!", True, "en"),
        uid=123456,
        turn_id=7,
    )
    decoded = json.loads(payload)

    assert decoded["object"] == "matchcast.caption"
    assert decoded["final"] is True
    assert decoded["turn_id"] == 7
    assert decoded["user_id"] == "123456"
    assert len(payload) < 1024


def test_revision_payload_is_small_and_carries_indexed_replacement():
    payload = transcript_revision_payload(
        TranscriptRevisionEvent(
            revision=12,
            replace_from=3,
            index=3,
            total=4,
            segment=TranscriptSegment(
                id="r12-s3",
                text="召" * 180,
                state="listening",
                created_at=1234,
            ),
        ),
        uid=123456,
    )
    decoded = json.loads(payload)

    assert decoded["object"] == "matchcast.transcript.segment"
    assert decoded["revision"] == 12
    assert decoded["replace_from"] == 3
    assert decoded["segment"]["state"] == "listening"
    assert len(payload) < 1024


def test_patch_payload_is_atomic_and_below_agora_limit():
    payload = transcript_patch_payload(
        TranscriptPatchEvent(
            revision=13,
            drop_from_start=1,
            replace_from=2,
            total=4,
            segments=(
                TranscriptSegment(
                    id="r13-s2",
                    text="Rekkles goes down to Tai.",
                    state="caption",
                    created_at=1234,
                ),
                TranscriptSegment(
                    id="r13-s3",
                    text="Gets a Baron steal!",
                    state="caption",
                    created_at=1234,
                ),
            ),
        ),
        uid=123456,
    )
    decoded = json.loads(payload)

    assert decoded["object"] == "matchcast.transcript.patch"
    assert decoded["revision"] == 13
    assert decoded["drop_from_start"] == 1
    assert decoded["replace_from"] == 2
    assert [segment["text"] for segment in decoded["segments"]] == [
        "Rekkles goes down to Tai.",
        "Gets a Baron steal!",
    ]
    assert len(payload) < 1024


def test_full_audio_queue_drops_stale_chunk_and_keeps_live_edge():
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    queue.put_nowait(b"stale")
    frame_states: list[bool] = []
    observer = _RemoteAudioObserver(
        loop=loop,
        queue=queue,
        media_uid=234567,
        on_frame=frame_states.append,
    )

    observer._offer(b"live")

    assert queue.get_nowait() == b"live"
    assert frame_states == [True, False]
    loop.close()


def test_gemini_mode_selects_exact_adapter():
    assert _gemini_transcriber_type_for_mode("gemini-live") is (
        GeminiLiveTranscriber
    )
