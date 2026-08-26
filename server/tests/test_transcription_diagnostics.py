import wave

from app.gemini_transcriber import (
    TranscriptPatchEvent,
    TranscriptSegment,
)
from app.transcription_diagnostics import (
    BoundedPcmWaveCapture,
    DiagnosticTranscriptDocument,
)


def segment(identifier: str, text: str, state: str = "caption"):
    return TranscriptSegment(
        id=identifier,
        text=text,
        state=state,
        created_at=1234,
    )


def test_diagnostic_document_applies_atomic_patch_like_browser():
    document = DiagnosticTranscriptDocument()

    assert document.apply(
        TranscriptPatchEvent(
            revision=1,
            drop_from_start=0,
            replace_from=0,
            total=2,
            segments=(
                segment("a", "Faker finds the angle."),
                segment("b", "The fight continues", "listening"),
            ),
        ),
        emitted_at=2000,
    )
    assert document.apply(
        TranscriptPatchEvent(
            revision=2,
            drop_from_start=0,
            replace_from=1,
            total=2,
            segments=(
                segment("c", "The fight continues in river.", "caption"),
            ),
        ),
        emitted_at=2100,
    )

    assert document.revision == 2
    assert document.stable_text == (
        "Faker finds the angle. The fight continues in river."
    )
    assert document.segments[-1].emitted_at == 2100


def test_diagnostic_document_applies_rolling_drop_and_rejects_bad_patch():
    document = DiagnosticTranscriptDocument(max_segments=2)
    document.apply(
        TranscriptPatchEvent(
            revision=1,
            drop_from_start=0,
            replace_from=0,
            total=2,
            segments=(segment("a", "One."), segment("b", "Two.")),
        ),
        emitted_at=2000,
    )

    assert document.apply(
        TranscriptPatchEvent(
            revision=2,
            drop_from_start=1,
            replace_from=1,
            total=2,
            segments=(segment("c", "Three."),),
        ),
        emitted_at=2100,
    )
    assert [item.text for item in document.segments] == ["Two.", "Three."]
    assert not document.apply(
        TranscriptPatchEvent(
            revision=3,
            drop_from_start=0,
            replace_from=2,
            total=1,
            segments=(),
        ),
        emitted_at=2200,
    )
    assert [item.text for item in document.segments] == ["Two.", "Three."]


def test_pcm_capture_writes_valid_bounded_wav(tmp_path):
    output = tmp_path / "nested" / "capture.wav"
    capture = BoundedPcmWaveCapture(
        output,
        max_seconds=0.2,
        sample_rate=16_000,
    )

    assert capture.write(b"\x01\x00" * 4_000) == 6_400
    assert capture.write(b"\x02\x00" * 100) == 0
    assert capture.close() == output
    assert capture.close() == output

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == 3_200


def test_empty_pcm_capture_does_not_create_file(tmp_path):
    output = tmp_path / "empty.wav"
    capture = BoundedPcmWaveCapture(output)

    assert capture.close() is None
    assert not output.exists()
