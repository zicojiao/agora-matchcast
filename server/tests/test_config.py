import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.vocabulary import (
    GEMINI_TRANSCRIBE_VOCABULARY,
    MATCHCAST_CLIP_VOCABULARY,
)


AMBIGUOUS_VOCABULARY = {
    "bang",
    "blank",
    "wolf",
    "crown",
    "ruler",
    "flash",
    "river",
    "peel",
    "kite",
    "poke",
}


def test_settings_loads_default_matchcast_vocabulary(monkeypatch):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    get_settings.cache_clear()

    settings = get_settings()

    assert "Faker" in settings.gemini_vocabulary
    assert "Baron Nashor" in settings.gemini_vocabulary
    assert "Perkz" in settings.gemini_vocabulary
    assert "Rekkles" in settings.gemini_vocabulary
    assert "Chronobreak" in settings.gemini_vocabulary
    assert "Zhonya's Hourglass" in settings.gemini_vocabulary
    assert settings.gemini_vocabulary_mode == "custom"
    assert settings.gemini_transcription_mode == "smart"
    assert settings.gemini_transcribe_vocabulary == list(
        GEMINI_TRANSCRIBE_VOCABULARY
    )
    assert settings.gemini_transcribe_activity_min_ms == 5_000
    assert settings.gemini_transcribe_activity_max_ms == 6_000
    assert settings.gemini_transcribe_activity_prefix_ms == 300
    assert settings.gemini_transcribe_activity_handoff_seconds == 1.5
    assert settings.gemini_transcribe_activity_buffer_max_ms == 3_000


def test_matchcast_preset_is_small_and_excludes_ambiguous_english():
    normalized = {term.casefold() for term in MATCHCAST_CLIP_VOCABULARY}

    assert len(MATCHCAST_CLIP_VOCABULARY) <= 50
    assert AMBIGUOUS_VOCABULARY.isdisjoint(normalized)


def test_settings_merge_vocabulary_additions_without_duplicates(monkeypatch):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    monkeypatch.setenv(
        "GEMINI_VOCABULARY",
        "faker, Hextech Soul, BARON NASHOR, Gen.G",
    )
    get_settings.cache_clear()

    settings = get_settings()

    assert "Hextech Soul" in settings.gemini_vocabulary
    assert "Gen.G" in settings.gemini_vocabulary
    assert sum(
        term.casefold() == "faker"
        for term in settings.gemini_vocabulary
    ) == 1
    assert sum(
        term.casefold() == "baron nashor"
        for term in settings.gemini_vocabulary
    ) == 1
    assert "Hextech Soul" not in settings.gemini_transcribe_vocabulary


def test_settings_loads_independent_gemini_transcribe_tuning(monkeypatch):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    monkeypatch.setenv(
        "GEMINI_TRANSCRIBE_VOCABULARY",
        "Faker,Keria,Cloud9,Orianna,Shockwave,QSS,Ryu,T1,Huni,Rekkles,ignored",
    )
    monkeypatch.setenv("GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS", "5200")
    monkeypatch.setenv("GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS", "6300")
    monkeypatch.setenv("GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS", "200")
    monkeypatch.setenv("GEMINI_TRANSCRIBE_ACTIVITY_HANDOFF_SECONDS", "1.25")
    monkeypatch.setenv("GEMINI_TRANSCRIBE_ACTIVITY_BUFFER_MAX_MS", "2500")
    monkeypatch.setenv("GEMINI_TRANSCRIPTION_MODE", "verbatim")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.gemini_transcribe_vocabulary == [
        "Faker",
        "Keria",
        "Cloud9",
        "Orianna",
        "Shockwave",
        "QSS",
        "Ryu",
        "T1",
        "Huni",
        "Rekkles",
    ]
    assert settings.gemini_transcribe_activity_min_ms == 5_200
    assert settings.gemini_transcribe_activity_max_ms == 6_300
    assert settings.gemini_transcribe_activity_prefix_ms == 200
    assert settings.gemini_transcribe_activity_handoff_seconds == 1.25
    assert settings.gemini_transcribe_activity_buffer_max_ms == 2_500
    assert settings.gemini_transcription_mode == "verbatim"


def test_settings_require_agora_credentials():
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="Missing Agora credentials"):
        get_settings()


def test_settings_rejects_unknown_gemini_transcription_mode(monkeypatch):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    monkeypatch.setenv("GEMINI_TRANSCRIPTION_MODE", "summarize")
    get_settings.cache_clear()

    with pytest.raises(ValidationError, match="gemini_transcription_mode"):
        get_settings()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("models/gemini-3.5-transcribe-live", "gemini-live"),
    ],
)
def test_settings_select_transcriber_mode(monkeypatch, model, expected):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    monkeypatch.setenv("GEMINI_MODEL", model)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.transcriber_mode == expected


@pytest.mark.parametrize(
    "model",
    [
        "models/gemini-3.1-flash-lite",
        "models/gemini-transcribe-batch",
    ],
)
def test_settings_rejects_non_live_gemini_model(monkeypatch, model):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    monkeypatch.setenv("GEMINI_MODEL", model)
    get_settings.cache_clear()

    settings = get_settings()

    with pytest.raises(ValueError, match="Unsupported GEMINI_MODEL"):
        _ = settings.transcriber_mode


def test_settings_loads_explicit_gemini_models(monkeypatch):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    monkeypatch.setenv(
        "GEMINI_TRANSCRIBE_MODEL",
        "models/private-transcribe",
    )
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.transcriber_mode_for_id("gemini-transcribe") == "gemini-live"
    assert settings.transcriber_model_for_id("gemini-transcribe") == (
        "models/private-transcribe"
    )


def test_settings_load_optional_bounded_pcm_capture(monkeypatch):
    monkeypatch.setenv("AGORA_APP_ID", "app")
    monkeypatch.setenv("AGORA_APP_CERTIFICATE", "certificate")
    monkeypatch.setenv(
        "TRANSCRIPTION_PCM_CAPTURE_PATH",
        "/tmp/matchcast.capture.wav",
    )
    monkeypatch.setenv("TRANSCRIPTION_PCM_CAPTURE_MAX_SECONDS", "12.5")
    get_settings.cache_clear()

    settings = get_settings()

    assert (
        settings.transcription_pcm_capture_path
        == "/tmp/matchcast.capture.wav"
    )
    assert settings.transcription_pcm_capture_max_seconds == 12.5
