import sys
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

ENV_KEYS = {
    "AGORA_APP_ID",
    "AGORA_APP_CERTIFICATE",
    "NEXT_PUBLIC_AGORA_APP_ID",
    "NEXT_AGORA_APP_CERTIFICATE",
    "AGORA_AREA_CODE",
    "AGORA_TOKEN_EXPIRE_SECONDS",
    "TRANSCRIBER_UID",
    "MEDIA_UID",
    "NEXT_PUBLIC_AGENT_UID",
    "NEXT_PUBLIC_MATCH_FEED_UID",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_TRANSCRIBE_MODEL",
    "GEMINI_LANGUAGE",
    "GEMINI_VOCABULARY",
    "GEMINI_TRANSCRIPTION_MODE",
    "GEMINI_TRANSCRIBE_ACTIVITY_HANDOFF_SECONDS",
    "GEMINI_TRANSCRIBE_ACTIVITY_BUFFER_MAX_MS",
    "LIVE_SESSION_MAX_SECONDS",
    "VIEWER_HEARTBEAT_TIMEOUT_SECONDS",
    "BACKEND_API_SECRET",
    "AGORA_RTC_LOG_DIR",
}


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
