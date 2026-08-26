import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models import StartSessionRequest, TranscriberStats
import app.session_manager as session_manager_module
from app.session_manager import SessionManager


class FakeTranscriber:
    instances = []

    def __init__(self, **options):
        self.options = options
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def stats(self):
        return TranscriberStats()

    def transcript_diagnostics(self):
        return 0, ()


@pytest.fixture
def fake_transcriber(monkeypatch):
    FakeTranscriber.instances.clear()
    monkeypatch.setattr(
        session_manager_module,
        "MatchCastTranscriber",
        FakeTranscriber,
    )
    return FakeTranscriber


@pytest.mark.asyncio
async def test_start_requires_gemini_api_key():
    manager = SessionManager(
        Settings(
            agora_app_id="app",
            agora_app_certificate="certificate",
        )
    )

    with pytest.raises(RuntimeError, match="Set GEMINI_API_KEY"):
        await manager.start(
            StartSessionRequest(
                requester_id="viewer",
                channel_name="matchcast",
            )
        )


def test_start_request_rejects_unknown_transcriber_id():
    with pytest.raises(ValidationError):
        StartSessionRequest(
            requester_id="viewer",
            channel_name="matchcast",
            transcriber_id="arbitrary-model",
        )


@pytest.mark.asyncio
async def test_start_selects_dedicated_gemini_model_by_id(fake_transcriber):
    manager = SessionManager(
        Settings(
            agora_app_id="app",
            agora_app_certificate="certificate",
            gemini_api_key="gemini-key",
            gemini_model="models/legacy-live-translate",
            gemini_transcribe_model="models/dedicated-transcribe",
        )
    )
    try:
        response = await manager.start(
            StartSessionRequest(
                requester_id="viewer",
                channel_name="matchcast",
                transcriber_id="gemini-transcribe",
            )
        )

        assert response.transcriber_id == "gemini-transcribe"
        assert response.transcription_mode == "gemini-live"
        assert response.model == "models/dedicated-transcribe"
        assert fake_transcriber.instances[-1].options["model"] == (
            "models/dedicated-transcribe"
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_invalid_replacement_keeps_existing_session(fake_transcriber):
    manager = SessionManager(
        Settings(
            agora_app_id="app",
            agora_app_certificate="certificate",
            gemini_api_key="gemini-key",
        )
    )
    try:
        active = await manager.start(
            StartSessionRequest(
                requester_id="viewer",
                channel_name="matchcast",
            )
        )

        status = await manager.status(
            session_id=active.session_id,
            agent_id=None,
        )
        assert status.state == "running"
        assert status.transcription_mode == "gemini-live"
        assert not fake_transcriber.instances[0].stopped
    finally:
        await manager.close()
