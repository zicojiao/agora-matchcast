from functools import lru_cache
import os
from typing import Literal

from pydantic import BaseModel, Field

from .transcriber_selection import (
    TranscriberId,
    TranscriptionMode,
)
from .vocabulary import (
    GEMINI_TRANSCRIBE_VOCABULARY,
    MATCHCAST_CLIP_VOCABULARY,
    merge_vocabulary,
)


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class Settings(BaseModel):
    agora_app_id: str = Field(min_length=1)
    agora_app_certificate: str = Field(min_length=1)
    agora_area_code: str = "global"
    transcriber_uid: int = 123456
    media_uid: int = 234567
    token_expire_seconds: int = 3600
    gemini_api_key: str | None = None
    gemini_model: str = "models/gemini-3.5-transcribe-live"
    gemini_transcribe_model: str = "models/gemini-3.5-transcribe-live"
    gemini_language: str = "en-US"
    gemini_vocabulary: list[str] = []
    gemini_transcribe_vocabulary: list[str] = []
    gemini_vocabulary_mode: Literal["custom", "none"] = "custom"
    gemini_transcription_mode: Literal["smart", "verbatim"] = "smart"
    gemini_transcribe_activity_min_ms: int = 5_000
    gemini_transcribe_activity_max_ms: int = 6_000
    gemini_transcribe_activity_prefix_ms: int = 300
    gemini_transcribe_activity_low_energy_ratio: float = 0.5
    gemini_transcribe_activity_commit_grace_seconds: float = 0.9
    gemini_transcribe_activity_handoff_seconds: float = 1.5
    gemini_transcribe_activity_buffer_max_ms: int = 3_000
    live_session_max_seconds: float = 540.0
    viewer_heartbeat_timeout_seconds: float = 45.0
    transcription_pcm_capture_path: str | None = None
    transcription_pcm_capture_max_seconds: float = 60.0
    backend_api_secret: str | None = None
    log_dir: str = "./agora_rtc_log"

    @property
    def transcriber_mode(self) -> str:
        return self.transcriber_mode_for()

    def transcriber_mode_for(self) -> TranscriptionMode:
        normalized_model = self.gemini_model.lower()
        if "transcribe" in normalized_model and "live" in normalized_model:
            return "gemini-live"
        raise ValueError(
            f"Unsupported GEMINI_MODEL {self.gemini_model!r}. Configure a "
            "Gemini Transcribe Live model."
        )

    def transcriber_model_for(self) -> str:
        return self.gemini_model

    @staticmethod
    def transcriber_mode_for_id(
        transcriber_id: TranscriberId,
    ) -> TranscriptionMode:
        return "gemini-live"

    def transcriber_model_for_id(self, transcriber_id: TranscriberId) -> str:
        return self.gemini_transcribe_model

    def infer_transcriber_id(self) -> TranscriberId:
        self.transcriber_mode_for()
        return "gemini-transcribe"


@lru_cache
def get_settings() -> Settings:
    app_id = _first_env("AGORA_APP_ID", "NEXT_PUBLIC_AGORA_APP_ID")
    app_certificate = _first_env(
        "AGORA_APP_CERTIFICATE",
        "NEXT_AGORA_APP_CERTIFICATE",
    )
    if not app_id or not app_certificate:
        raise RuntimeError(
            "Missing Agora credentials. Set AGORA_APP_ID and AGORA_APP_CERTIFICATE."
        )

    vocabulary_additions = [
        item.strip()
        for item in os.getenv("GEMINI_VOCABULARY", "").split(",")
        if item.strip()
    ]
    vocabulary = merge_vocabulary(
        MATCHCAST_CLIP_VOCABULARY,
        vocabulary_additions,
    )
    transcribe_vocabulary_override = [
        item.strip()
        for item in os.getenv(
            "GEMINI_TRANSCRIBE_VOCABULARY",
            "",
        ).split(",")
        if item.strip()
    ]
    transcribe_vocabulary = list(
        transcribe_vocabulary_override
        or GEMINI_TRANSCRIBE_VOCABULARY
    )[:10]

    gemini_model = os.getenv(
        "GEMINI_MODEL",
        "models/gemini-3.5-transcribe-live",
    )

    return Settings(
        agora_app_id=app_id,
        agora_app_certificate=app_certificate,
        agora_area_code=os.getenv("AGORA_AREA_CODE", "global").lower(),
        transcriber_uid=int(
            os.getenv("TRANSCRIBER_UID", os.getenv("NEXT_PUBLIC_AGENT_UID", "123456"))
        ),
        media_uid=int(
            os.getenv("MEDIA_UID", os.getenv("NEXT_PUBLIC_MATCH_FEED_UID", "234567"))
        ),
        token_expire_seconds=int(os.getenv("AGORA_TOKEN_EXPIRE_SECONDS", "3600")),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=gemini_model,
        gemini_transcribe_model=_first_env(
            "GEMINI_TRANSCRIBE_MODEL",
            default=gemini_model,
        ),
        gemini_language=os.getenv("GEMINI_LANGUAGE", "en-US"),
        gemini_vocabulary=vocabulary,
        gemini_transcribe_vocabulary=transcribe_vocabulary,
        gemini_vocabulary_mode=os.getenv(
            "GEMINI_VOCABULARY_MODE",
            "custom",
        ).lower(),
        gemini_transcription_mode=os.getenv(
            "GEMINI_TRANSCRIPTION_MODE",
            "smart",
        ).lower(),
        gemini_transcribe_activity_min_ms=int(
            os.getenv("GEMINI_TRANSCRIBE_ACTIVITY_MIN_MS", "5000")
        ),
        gemini_transcribe_activity_max_ms=int(
            os.getenv("GEMINI_TRANSCRIBE_ACTIVITY_MAX_MS", "6000")
        ),
        gemini_transcribe_activity_prefix_ms=int(
            os.getenv("GEMINI_TRANSCRIBE_ACTIVITY_PREFIX_MS", "300")
        ),
        gemini_transcribe_activity_low_energy_ratio=float(
            os.getenv(
                "GEMINI_TRANSCRIBE_ACTIVITY_LOW_ENERGY_RATIO",
                "0.5",
            )
        ),
        gemini_transcribe_activity_commit_grace_seconds=float(
            os.getenv(
                "GEMINI_TRANSCRIBE_ACTIVITY_COMMIT_GRACE_SECONDS",
                "0.9",
            )
        ),
        gemini_transcribe_activity_handoff_seconds=float(
            os.getenv(
                "GEMINI_TRANSCRIBE_ACTIVITY_HANDOFF_SECONDS",
                "1.5",
            )
        ),
        gemini_transcribe_activity_buffer_max_ms=int(
            os.getenv(
                "GEMINI_TRANSCRIBE_ACTIVITY_BUFFER_MAX_MS",
                "3000",
            )
        ),
        live_session_max_seconds=float(os.getenv("LIVE_SESSION_MAX_SECONDS", "540")),
        viewer_heartbeat_timeout_seconds=float(
            os.getenv("VIEWER_HEARTBEAT_TIMEOUT_SECONDS", "45")
        ),
        transcription_pcm_capture_path=(
            os.getenv("TRANSCRIPTION_PCM_CAPTURE_PATH") or None
        ),
        transcription_pcm_capture_max_seconds=float(
            os.getenv("TRANSCRIPTION_PCM_CAPTURE_MAX_SECONDS", "60")
        ),
        backend_api_secret=os.getenv("BACKEND_API_SECRET"),
        log_dir=os.getenv("AGORA_RTC_LOG_DIR", "./agora_rtc_log"),
    )
