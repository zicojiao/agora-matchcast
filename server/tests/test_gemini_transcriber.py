import asyncio
from types import SimpleNamespace

import pytest

from app.gemini_transcriber import (
    CHUNK_BYTES,
    ActivityHandoffResult,
    FinalEventGeneration,
    GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS,
    GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS,
    GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS,
    GeminiLiveTranscriber,
    GeminiTranscriptDocument,
    ManualActivitySegmenter,
    Pcm16Normalizer,
    RollingTranscriptDocument,
    StableActivityTranscript,
    StableTranscriptDocument,
    caption_events_from_response,
    merge_activity_context,
    build_audio_transcription_config,
    buffer_audio_until_final,
)


def pcm_chunk(level: int) -> bytes:
    sample = int(level).to_bytes(2, "little", signed=True)
    return sample * (CHUNK_BYTES // 2)


class FakeAudioTranscriptionConfig:
    def __init__(self, **kwargs):
        self.options = kwargs


class FakeTypes:
    """The launch API is flat, so no nested language helper types exist.

    Leaving `LanguageHints` and `LanguageAuto` off this fake makes any
    reintroduction of the removed EAP options fail loudly.
    """

    AudioTranscriptionConfig = FakeAudioTranscriptionConfig


def test_dedicated_transcribe_config_sends_language_code_and_vocabulary():
    config = build_audio_transcription_config(
        FakeTypes,
        language="en-US",
        vocabulary=["Faker", "Kai'Sa"],
        dedicated_transcribe=True,
    )

    assert config.options["custom_vocabulary"] == ["Faker", "Kai'Sa"]
    assert config.options["language_codes"] == ["en-US"]
    assert config.options["mode"] == "SMART"
    assert "language_hints" not in config.options
    assert "language_auto" not in config.options
    assert "adaptation_phrases" not in config.options


def test_dedicated_transcribe_config_supports_auto_language():
    config = build_audio_transcription_config(
        FakeTypes,
        language="auto",
        vocabulary=[],
        dedicated_transcribe=True,
    )

    assert config.options["language_codes"] == []
    assert "language_auto" not in config.options


def test_dedicated_transcribe_config_supports_verbatim_mode():
    config = build_audio_transcription_config(
        FakeTypes,
        language="en-US",
        vocabulary=[],
        dedicated_transcribe=True,
        transcription_mode="verbatim",
    )

    assert config.options["mode"] == "VERBATIM"


def test_non_dedicated_config_does_not_send_eap_only_options():
    config = build_audio_transcription_config(
        FakeTypes,
        language="en-US",
        vocabulary=["Faker"],
        dedicated_transcribe=False,
    )

    assert config.options == {}


def test_dedicated_transcribe_config_rejects_sdk_without_mode():
    class OldAudioTranscriptionConfig:
        model_fields = {"custom_vocabulary": object()}

    old_types = SimpleNamespace(
        AudioTranscriptionConfig=OldAudioTranscriptionConfig,
    )

    with pytest.raises(RuntimeError, match="does not support.*mode"):
        build_audio_transcription_config(
            old_types,
            language="en-US",
            vocabulary=[],
            dedicated_transcribe=True,
        )


def test_dedicated_transcribe_config_can_disable_custom_vocabulary():
    config = build_audio_transcription_config(
        FakeTypes,
        language="en-US",
        vocabulary=["Death Mark", "Ryu"],
        dedicated_transcribe=True,
        vocabulary_mode="none",
    )

    assert "custom_vocabulary" not in config.options
    assert "adaptation_phrases" not in config.options
    assert config.options["language_codes"] == ["en-US"]


def test_pcm_normalizer_makes_100ms_16khz_chunks():
    normalizer = Pcm16Normalizer()
    stereo_48khz_250ms = b"\x00\x00\x00\x00" * 12_000

    chunks = normalizer.push(
        stereo_48khz_250ms,
        sample_rate=48_000,
        channels=2,
    )

    assert len(chunks) == 2
    assert all(len(chunk) == CHUNK_BYTES for chunk in chunks)

    # The remaining 50 ms is retained and combines with the next 50 ms frame.
    next_chunks = normalizer.push(
        b"\x00\x00\x00\x00" * 2_400,
        sample_rate=48_000,
        channels=2,
    )
    assert len(next_chunks) == 1
    assert len(next_chunks[0]) == CHUNK_BYTES


def test_manual_activity_segmenter_prefers_low_energy_after_minimum():
    segmenter = ManualActivitySegmenter()
    batches = [
        segmenter.push(pcm_chunk(1_000))
        for _ in range(23)
    ]

    boundary = segmenter.push(pcm_chunk(0))

    assert not any(batch.boundary for batch in batches)
    assert boundary.boundary is True
    assert len(boundary.next_prefix) == 6


def test_manual_activity_segmenter_forces_constant_audio_by_maximum():
    segmenter = ManualActivitySegmenter()
    boundary_at = None

    for index in range(1, 50):
        batch = segmenter.push(pcm_chunk(1_000))
        if batch.boundary:
            boundary_at = index
            break

    assert boundary_at == 36
    assert len(batch.next_prefix) == 6


def test_manual_activity_segmenter_replays_context_without_dropping_source():
    segmenter = ManualActivitySegmenter()
    chunks = [
        pcm_chunk(500 + index)
        for index in range(100)
    ]
    activities: list[list[bytes]] = []
    current_activity: list[bytes] = []
    boundary_count = 0

    for chunk in chunks:
        batch = segmenter.push(chunk)
        current_activity.extend(batch.current)
        if batch.boundary:
            boundary_count += 1
            activities.append(current_activity)
            current_activity = list(batch.next_prefix)
    current_activity.extend(segmenter.flush())
    activities.append(current_activity)

    assert boundary_count >= 2
    for previous, current in zip(activities, activities[1:]):
        assert current[:6] == previous[-6:]
    reconstructed: list[bytes] = list(activities[0])
    for activity in activities[1:]:
        reconstructed.extend(activity[6:])
    assert reconstructed == chunks


def test_gemini_endpointing_uses_longer_turns_and_smaller_prefix():
    segmenter = ManualActivitySegmenter(
        min_ms=GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS,
        max_ms=GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS,
        prefix_ms=GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS,
    )

    before_boundary = [
        segmenter.push(pcm_chunk(1_000))
        for _ in range(49)
    ]
    low_energy_boundary = segmenter.push(pcm_chunk(0))

    assert not any(batch.boundary for batch in before_boundary)
    assert low_energy_boundary.boundary is True
    assert len(low_energy_boundary.next_prefix) == 3


def test_gemini_endpointing_forces_continuous_audio_at_six_seconds():
    segmenter = ManualActivitySegmenter(
        min_ms=GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS,
        max_ms=GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS,
        prefix_ms=GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS,
    )

    boundary_at = None
    for index in range(1, 80):
        batch = segmenter.push(pcm_chunk(1_000))
        if batch.boundary:
            boundary_at = index
            break

    assert boundary_at == 60
    assert len(batch.next_prefix) == 3


@pytest.mark.asyncio
async def test_activity_handoff_buffers_audio_until_new_final():
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    chunks = (pcm_chunk(100), pcm_chunk(200))
    for chunk in chunks:
        queue.put_nowait(chunk)
    final_events = FinalEventGeneration()
    stop = asyncio.Event()

    handoff = asyncio.create_task(
        buffer_audio_until_final(
            queue,
            final_events,
            after_generation=0,
            stop=stop,
            timeout_seconds=0.5,
            buffer_max_ms=3_000,
        )
    )
    while not queue.empty():
        await asyncio.sleep(0)
    await final_events.notify()

    result = await handoff

    assert result.acknowledged is True
    assert result.chunks == chunks
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_activity_handoff_ignores_final_before_boundary():
    final_events = FinalEventGeneration()
    await final_events.notify()

    result = await buffer_audio_until_final(
        asyncio.Queue(),
        final_events,
        after_generation=final_events.generation,
        stop=asyncio.Event(),
        timeout_seconds=0.01,
        buffer_max_ms=3_000,
    )

    assert result.acknowledged is False
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_activity_handoff_resumes_at_buffer_limit_without_dropping():
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    chunks = (pcm_chunk(100), pcm_chunk(200), pcm_chunk(300))
    for chunk in chunks:
        queue.put_nowait(chunk)

    result = await buffer_audio_until_final(
        queue,
        FinalEventGeneration(),
        after_generation=0,
        stop=asyncio.Event(),
        timeout_seconds=0.5,
        buffer_max_ms=200,
    )

    assert result.buffer_limited is True
    assert result.chunks == chunks[:2]
    assert queue.get_nowait() == chunks[2]


@pytest.mark.asyncio
async def test_activity_handoff_zero_seconds_restores_immediate_restart():
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    queue.put_nowait(pcm_chunk(100))

    result = await buffer_audio_until_final(
        queue,
        FinalEventGeneration(),
        after_generation=0,
        stop=asyncio.Event(),
        timeout_seconds=0,
        buffer_max_ms=3_000,
    )

    assert result == ActivityHandoffResult(chunks=(), wait_seconds=0.0)
    assert not queue.empty()


@pytest.mark.asyncio
async def test_activity_handoff_stop_returns_buffered_audio():
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    chunk = pcm_chunk(100)
    queue.put_nowait(chunk)
    stop = asyncio.Event()

    handoff = asyncio.create_task(
        buffer_audio_until_final(
            queue,
            FinalEventGeneration(),
            after_generation=0,
            stop=stop,
            timeout_seconds=0.5,
            buffer_max_ms=3_000,
        )
    )
    while not queue.empty():
        await asyncio.sleep(0)
    stop.set()

    result = await handoff

    assert result.stopped is True
    assert result.chunks == (chunk,)


def test_gemini_handoff_metrics_are_reported():
    transcriber = GeminiLiveTranscriber(
        api_key="key",
        model="models/gemini-3.5-transcribe-live",
        language="en-US",
        vocabulary=[],
    )

    transcriber._capture_handoff_metrics(
        ActivityHandoffResult(
            chunks=(pcm_chunk(100), pcm_chunk(200)),
            wait_seconds=0.25,
            acknowledged=True,
        )
    )

    metrics = transcriber.transcript_metrics
    assert metrics["activity_boundaries"] == 1
    assert metrics["activity_final_acks"] == 1
    assert metrics["activity_handoff_peak_buffer_ms"] == 200
    assert metrics["activity_handoff_max_wait_seconds"] == 0.25


def test_response_normalization_keeps_interim_and_final():
    response = SimpleNamespace(
        server_content=SimpleNamespace(
            interim_input_transcription=SimpleNamespace(text="Faker finds"),
            input_transcription=SimpleNamespace(
                text="Faker finds the Shockwave.",
                finished=True,
            ),
        )
    )

    events = caption_events_from_response(response, "en")

    assert [(event.text, event.is_final) for event in events] == [
        ("Faker finds", False),
        ("Faker finds the Shockwave.", True),
    ]


def test_input_transcription_field_is_final_even_when_finished_is_false():
    response = SimpleNamespace(
        server_content=SimpleNamespace(
            input_transcription=SimpleNamespace(
                text="Faker finds the Shockwave.",
                finished=False,
            ),
        )
    )

    events = caption_events_from_response(response, "en")

    assert [(event.text, event.is_final) for event in events] == [
        ("Faker finds the Shockwave.", True),
    ]


def test_rolling_document_replaces_a_growing_listening_tail():
    document = RollingTranscriptDocument(language="en-US")

    first = document.update("Bang is hiding", now_ms=1)
    second = document.update("Bang is hiding. He's coming", now_ms=2)
    third = document.update(
        "Bang is hiding. He's coming. Bang looking to come in",
        now_ms=3,
    )

    assert first[0].replace_from == 0
    assert second[0].replace_from == 0
    assert third[0].replace_from == 1
    assert [(item.text, item.state) for item in document.segments] == [
        ("Bang is hiding.", "caption"),
        ("He's coming.", "caption"),
        ("Bang looking to come in", "listening"),
    ]


def test_rolling_document_revises_shockwave_chain_in_place():
    document = RollingTranscriptDocument(language="en-US")
    snapshots = [
        "TP!",
        "TP! Oh, Shockwave!",
        "TP! Oh, Shockwave! It lands, and that's going to force Doom into an early Zhonya's.",
        "TP! Oh, Shockwave! It lands, and that's going to force Doom be into an early Zhonya's.",
        (
            "TP! Oh, Shockwave! It lands, and that's going to force Zilean "
            "into an early Zhonya's. The Shockwave is enough to only catch one. "
            "The Olaf is dead."
        ),
    ]

    for index, snapshot in enumerate(snapshots):
        document.update(snapshot, now_ms=index)

    rendered = [item.text for item in document.segments]
    assert rendered == [
        "TP!",
        "Oh, Shockwave!",
        "It lands, and that's going to force Zilean into an early Zhonya's.",
        "The Shockwave is enough to only catch one.",
        "The Olaf is dead.",
    ]
    assert all(rendered.count(text) == 1 for text in rendered)


def test_rolling_document_aligns_a_replayed_window_without_appending_it():
    document = RollingTranscriptDocument(language="en-US")
    document.update(
        "Orianna never really gained the opportunity never getting the TP! "
        "Oh, Shockwave! It lands, and that's going to force Doom into an early Zhonya's.",
        now_ms=1,
    )

    events = document.update(
        "opportunity. Never get the TP. Oh, Shockwave! It lands, and that's "
        "going to force Zilean into an early Zhonya's. The Shockwave is enough "
        "to only catch one. The Olaf is dead.",
        now_ms=2,
    )

    text = " ".join(item.text for item in document.segments)
    assert events
    assert text.count("Oh, Shockwave!") == 1
    assert text.count("It lands") == 1
    assert "Zilean" in text
    assert "Doom" not in text


def test_rolling_document_keeps_cutoff_tail_listening_until_corrected():
    document = RollingTranscriptDocument(language="en-US")
    document.update("The Olaf is dead. It's turning t", now_ms=1)

    assert document.segments[-1].text == "It's turning t"
    assert document.segments[-1].state == "listening"

    events = document.update(
        "The Olaf is dead. It's turning toward the river.",
        now_ms=2,
    )

    assert events[0].replace_from == 1
    assert document.segments[-1].text == "It's turning toward the river."
    assert document.segments[-1].state == "caption"


def test_rolling_document_does_not_rewind_on_generic_two_word_overlap():
    document = RollingTranscriptDocument(language="en-US")
    document.update(
        "And that's insane! Whoo! Perkz is going low, but he Chronobreaks back. "
        "A double kill for Huni! Get kidding me? They're going to take down the",
        now_ms=1,
    )
    document.update(
        "A slaughter of T1, 12-0 and 6.",
        now_ms=2,
    )
    document.update(
        "And RNG going to Faker Busan Busan",
        now_ms=3,
    )

    rendered = [item.text for item in document.segments]
    assert rendered[:4] == [
        "And that's insane!",
        "Whoo!",
        "Perkz is going low, but he Chronobreaks back.",
        "A double kill for Huni!",
    ]
    assert rendered[-2:] == [
        "A slaughter of T1, 12-0 and 6.",
        "And RNG going to Faker Busan Busan",
    ]
    assert len(rendered) == 7
    assert document.segments[-1].state == "listening"


def test_rolling_document_collapses_adjacent_window_replay():
    document = RollingTranscriptDocument(language="en-US")
    document.update(
        "unbelievable Baron steal. It's game five. unfinished tail",
        now_ms=1,
    )
    document.update("It's game five.", now_ms=2)

    rendered = [item.text for item in document.segments]
    assert rendered == [
        "unbelievable Baron steal.",
        "It's game five.",
    ]


def test_rolling_document_replaces_short_tail_across_manual_activities():
    document = RollingTranscriptDocument(language="en-US")
    document.update("The fight starts. Teleport!", now_ms=1)
    document.update("Teleport.", now_ms=2)
    document.update("Dwindling.", now_ms=3)
    document.update("Dwindling and taking down BLG.", now_ms=4)

    assert [item.text for item in document.segments] == [
        "The fight starts.",
        "Teleport.",
        "Dwindling and taking down BLG.",
    ]


def test_rolling_document_revises_recent_fuzzy_tail_in_place():
    document = RollingTranscriptDocument(language="en-US")
    document.update("Love you to death, Mark.", now_ms=1_000)
    document.update("Brother or death, Mark.", now_ms=1_400)
    events = document.update(
        "Double or death mark tries to clean it up for Ryu.",
        now_ms=1_800,
    )

    assert events[0].replace_from == 0
    assert [item.text for item in document.segments] == [
        "Double or death mark tries to clean it up for Ryu.",
    ]


def test_rolling_document_keeps_distinct_recent_short_captions():
    document = RollingTranscriptDocument(language="en-US")
    document.update("What the hell?", now_ms=1_000)
    document.update("Gets a triple kill.", now_ms=1_300)

    assert [item.text for item in document.segments] == [
        "What the hell?",
        "Gets a triple kill.",
    ]


def test_rolling_document_does_not_revise_an_old_similar_tail():
    document = RollingTranscriptDocument(language="en-US")
    document.update("Turtle on the quad.", now_ms=1_000)
    document.update("Turtle on the quadra.", now_ms=5_100)

    assert [item.text for item in document.segments] == [
        "Turtle on the quad.",
        "Turtle on the quadra.",
    ]


def test_stable_document_only_revises_current_turn_draft():
    document = StableTranscriptDocument(language="en-US")

    first = document.preview("Reckles goes down", now_ms=1_000)
    corrected = document.preview("Rekkles goes down to Tai.", now_ms=1_100)
    committed = document.commit("Rekkles goes down to Tai.", now_ms=1_200)
    next_turn = document.preview("Gets a Baron steal", now_ms=1_300)

    assert first is not None
    assert first.segments[0].state == "listening"
    assert corrected is not None
    assert corrected.replace_from == 0
    assert corrected.segments[0].text == "Rekkles goes down to Tai."
    assert committed is not None
    assert committed.segments[0].state == "caption"
    assert next_turn is not None
    assert next_turn.replace_from == 1
    assert [segment.text for segment in document.segments] == [
        "Rekkles goes down to Tai.",
    ]


def test_stable_document_never_rewrites_a_committed_caption():
    document = StableTranscriptDocument(language="en-US")
    document.preview("Kidding me?", now_ms=1_000)
    document.commit("Kidding me?", now_ms=1_100)

    patch = document.preview("Santorin fights early.", now_ms=1_200)

    assert patch is not None
    assert patch.replace_from == 1
    assert patch.segments[0].text == "Santorin fights early."
    assert [segment.text for segment in document.segments] == ["Kidding me?"]


def test_stable_document_drops_repeated_incomplete_tail_after_same_caption():
    document = StableTranscriptDocument(language="en-US")

    patch = document.commit("Faker! What? What", now_ms=1_000, force=False)

    assert patch is not None
    assert [item.text for item in document.segments] == ["Faker!", "What?"]
    assert document.draft_text == ""


def test_activity_transcript_preserves_unfinished_tail_across_boundaries():
    transcript = StableActivityTranscript(language="en-US")
    transcript.update("Oh, Faker maybe")

    transcript.commit(force=False)
    assert transcript.carried_draft == "Oh, Faker maybe"

    preview = transcript.update("the cleanse, looking for the moves.")
    assert preview is not None
    assert preview.segments[-1].text == (
        "Oh, Faker maybe the cleanse, looking for the moves."
    )

    transcript.commit(force=False)
    assert [item.text for item in transcript.document.segments] == [
        "Oh, Faker maybe the cleanse, looking for the moves.",
    ]
    assert transcript.carried_draft == ""


def test_activity_transcript_revises_current_activity_without_duplicating_carry():
    transcript = StableActivityTranscript(language="en-US")
    transcript.update("Play the QSS")
    transcript.commit(force=False)

    transcript.update("to cleanse")
    transcript.update("to cleanse the Death Mark.")

    assert transcript.document.draft_text == (
        "Play the QSS to cleanse the Death Mark."
    )


def test_activity_final_replaces_stale_interim_snapshot_before_commit():
    transcript = StableActivityTranscript(language="en-US")
    transcript.update("Faker nightmares. nightmares, you wake up")

    preview, committed = transcript.finalize(
        "Faker nightmares, you wake up in a cold sweat.",
    )

    assert preview is not None
    assert committed is not None
    assert [item.text for item in transcript.document.segments] == [
        "Faker nightmares, you wake up in a cold sweat.",
    ]


def test_activity_context_deduplicates_exact_overlap():
    assert merge_activity_context(
        "Play the QSS to cleanse",
        "to cleanse the Death Mark.",
    ) == "Play the QSS to cleanse the Death Mark."


def test_activity_context_uses_distinctive_word_as_revision_anchor():
    assert merge_activity_context(
        "Oh, Faker maybe",
        "Faker made the play. Look at the cleanse.",
    ) == "Oh, Faker made the play. Look at the cleanse."


def test_stable_document_trims_exact_replayed_turn_context():
    document = StableTranscriptDocument(language="en-US")
    document.commit("The quadra kill for Huni.", now_ms=1_000)

    preview = document.preview(
        "quadra kill for Huni. Dwindling.",
        now_ms=1_100,
    )
    committed = document.commit(
        "quadra kill for Huni. Dwindling.",
        now_ms=1_200,
    )

    assert preview is not None
    assert preview.segments[0].text == "Dwindling."
    assert committed is not None
    assert [segment.text for segment in document.segments] == [
        "The quadra kill for Huni.",
        "Dwindling.",
    ]


def test_stable_document_commit_atomically_replaces_draft_with_sentences():
    document = StableTranscriptDocument(language="en-US")
    document.preview(
        "The Shockwave lands. Faker finds the angle",
        now_ms=1_000,
    )

    patch = document.commit(
        "The Shockwave lands. Faker finds the angle",
        now_ms=1_100,
    )

    assert patch is not None
    assert patch.drop_from_start == 0
    assert patch.replace_from == 0
    assert patch.total == 2
    assert [segment.text for segment in patch.segments] == [
        "The Shockwave lands.",
        "Faker finds the angle",
    ]
    assert all(segment.state == "caption" for segment in patch.segments)


def test_stable_document_carries_incomplete_phrase_across_boundary():
    document = StableTranscriptDocument(language="en-US")
    document.preview("JDG never really gained the", now_ms=1_000)

    first_boundary = document.commit(
        "JDG never really gained the",
        now_ms=1_100,
        force=False,
    )
    document.preview(
        "JDG never really gained the opportunity.",
        now_ms=1_200,
    )
    second_boundary = document.commit(
        "JDG never really gained the opportunity.",
        now_ms=1_300,
        force=False,
    )

    assert first_boundary is None
    assert document.draft_text == ""
    assert second_boundary is not None
    assert [segment.text for segment in document.segments] == [
        "JDG never really gained the opportunity.",
    ]


def test_stable_document_forces_unpunctuated_phrase_after_three_boundaries():
    document = StableTranscriptDocument(language="en-US")
    document.preview("Faker finds the angle", now_ms=1_000)
    first_boundary = document.commit(
        "Faker finds the angle",
        now_ms=1_100,
        force=False,
    )
    second_boundary = document.commit(
        "Faker finds the angle",
        now_ms=1_200,
        force=False,
    )

    forced = document.commit(
        "Faker finds the angle",
        now_ms=1_300,
        force=False,
    )

    assert first_boundary is None
    assert second_boundary is None
    assert forced is not None
    assert forced.segments[0].text == "Faker finds the angle"
    assert forced.segments[0].state == "caption"
    assert document.draft_text == ""


def test_stable_document_collapses_exact_adjacent_repeated_sentences():
    document = StableTranscriptDocument(language="en-US")

    patch = document.commit(
        "Done. done. Done. He cannot do it.",
        now_ms=1_000,
    )

    assert patch is not None
    assert [segment.text for segment in document.segments] == [
        "Done.",
        "He cannot do it.",
    ]


def test_stable_document_collapses_exact_repeat_across_turns():
    document = StableTranscriptDocument(language="en-US")
    document.commit("Done.", now_ms=1_000)
    document.preview("done.", now_ms=1_100)

    patch = document.commit("done.", now_ms=1_200)

    assert patch is not None
    assert patch.replace_from == 1
    assert patch.segments == ()
    assert [segment.text for segment in document.segments] == ["Done."]


def test_gemini_document_commits_every_unpunctuated_final_immediately():
    document = GeminiTranscriptDocument(language="en-US")
    document.preview("Faker finds the angle", now_ms=1_000)

    patch = document.commit("Faker finds the angle", now_ms=1_100)

    assert patch is not None
    assert patch.replace_from == 0
    assert patch.total == 1
    assert [(item.text, item.state) for item in document.segments] == [
        ("Faker finds the angle", "caption"),
    ]
    assert document.draft_text == ""
    assert document.raw_final_words == 4
    assert document.committed_final_words == 4


def test_gemini_document_trims_only_exact_adjacent_replay():
    document = GeminiTranscriptDocument(language="en-US")
    document.commit(
        "The quadra kill for Huni.",
        now_ms=1_000,
    )

    patch = document.commit(
        "kill for Huni. Cloud 9 reach the semifinals.",
        now_ms=1_100,
    )

    assert patch is not None
    assert [item.text for item in document.segments] == [
        "The quadra kill for Huni.",
        "Cloud 9 reach the semifinals.",
    ]
    assert document.raw_final_words == 13
    assert document.committed_final_words == 10
    assert document.overlap_words_removed == 3


def test_gemini_document_does_not_fuzzy_rewrite_committed_history():
    document = GeminiTranscriptDocument(language="en-US")
    document.commit("Love you to death, Mark.", now_ms=1_000)

    document.commit(
        "Double or Death Mark tries to clean it up for Ryu.",
        now_ms=1_100,
    )

    assert [item.text for item in document.segments] == [
        "Love you to death, Mark.",
        "Double or Death Mark tries to clean it up for Ryu.",
    ]


def test_gemini_document_ignores_exact_duplicate_and_clears_draft():
    document = GeminiTranscriptDocument(language="en-US")
    document.commit("Faker finds the angle.", now_ms=1_000)
    document.preview(
        "Faker finds the angle and wins",
        now_ms=1_100,
    )

    patch = document.commit("Faker finds the angle.", now_ms=1_200)

    assert patch is not None
    assert patch.segments == ()
    assert patch.replace_from == 1
    assert document.draft_text == ""
    assert [item.text for item in document.segments] == [
        "Faker finds the angle.",
    ]


def test_gemini_document_preserves_last_interim_when_session_closes():
    document = GeminiTranscriptDocument(language="en-US")
    document.preview("Cloud 9 in the top four", now_ms=1_000)

    patch = document.finalize_draft(now_ms=1_100)

    assert patch is not None
    assert [item.text for item in document.segments] == [
        "Cloud 9 in the top four",
    ]
    assert document.draft_text == ""
