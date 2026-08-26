import asyncio
import logging
import time
from collections import deque

from .config import Settings
from .matchcast_transcriber import MatchCastTranscriber
from .models import (
    SessionLifecycleEvent,
    SessionStatusResponse,
    StartSessionRequest,
    StartSessionResponse,
)

logger = logging.getLogger(__name__)


class LiveSession:
    def __init__(
        self,
        *,
        session_id: str,
        agent_id: str,
        channel_name: str,
        transcriber: MatchCastTranscriber,
        transcriber_id: str | None,
        transcription_mode: str,
        model: str,
        created_at: int,
        created_at_monotonic: float,
    ) -> None:
        self.session_id = session_id
        self.agent_id = agent_id
        self.channel_name = channel_name
        self.transcriber = transcriber
        self.transcriber_id = transcriber_id
        self.transcription_mode = transcription_mode
        self.model = model
        self.created_at = created_at
        self.created_at_monotonic = created_at_monotonic
        self.last_viewer_heartbeat_at = created_at_monotonic
        self.last_viewer_heartbeat_ts = created_at
        self.stopped_at: int | None = None
        self.stop_reason: str | None = None
        self.events: deque[SessionLifecycleEvent] = deque(maxlen=40)
        self.monitor_task: asyncio.Task[None] | None = None


class SessionManager:
    RECORD_LIMIT = 100

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, LiveSession] = {}
        self._records: dict[str, LiveSession] = {}
        self._lock = asyncio.Lock()

    async def start(self, request: StartSessionRequest) -> StartSessionResponse:
        async with self._lock:
            if request.transcriber_id:
                transcriber_id = request.transcriber_id
                transcription_mode = self._settings.transcriber_mode_for_id(
                    transcriber_id
                )
                model = self._settings.transcriber_model_for_id(transcriber_id)
            else:
                transcriber_id = self._settings.infer_transcriber_id()
                transcription_mode = self._settings.transcriber_mode_for()
                model = self._settings.transcriber_model_for()
            if not self._settings.gemini_api_key:
                raise RuntimeError("No Gemini credential. Set GEMINI_API_KEY.")
            existing = list(self._sessions.values())
            self._sessions.clear()
            await self._stop_sessions(existing, "replaced by newer session")

            transcriber_uid = request.transcriber_uid or self._settings.transcriber_uid
            media_uid = request.media_uid or self._settings.media_uid
            now = int(time.time())
            session_id = f"{request.channel_name}:{media_uid}:{now}"
            agent_id = f"matchcast-transcriber-{request.channel_name}-{now}"
            transcriber = MatchCastTranscriber(
                settings=self._settings,
                channel_name=request.channel_name,
                transcriber_uid=transcriber_uid,
                media_uid=media_uid,
                transcription_mode=transcription_mode,
                model=model,
            )
            transcriber.start()
            session = LiveSession(
                session_id=session_id,
                agent_id=agent_id,
                channel_name=request.channel_name,
                transcriber=transcriber,
                transcriber_id=transcriber_id,
                transcription_mode=transcription_mode,
                model=model,
                created_at=now,
                created_at_monotonic=time.monotonic(),
            )
            self._sessions[session_id] = session
            self._records[session_id] = session
            while len(self._records) > self.RECORD_LIMIT:
                self._records.pop(next(iter(self._records)))
            self._emit(
                session,
                "session_started",
                f"Listening for Media Gateway audio in {transcription_mode} mode.",
            )
            session.monitor_task = asyncio.create_task(
                self._monitor(session_id),
                name=f"matchcast-monitor-{request.channel_name}",
            )
            return StartSessionResponse(
                session_id=session_id,
                agent_id=agent_id,
                create_ts=now,
                state="RUNNING",
                channel_name=request.channel_name,
                agent_uid=str(transcriber_uid),
                media_uid=str(media_uid),
                transcriber_id=transcriber_id,
                transcription_mode=transcription_mode,
                model=model,
            )

    async def stop(self, *, session_id: str | None, agent_id: str | None) -> None:
        async with self._lock:
            session = self._find(self._sessions, session_id, agent_id, None)
            if session:
                self._sessions.pop(session.session_id, None)
        await self._stop_sessions(
            [session] if session else [],
            "explicit stop requested",
        )

    async def heartbeat(self, *, session_id: str | None, agent_id: str | None) -> bool:
        async with self._lock:
            session = self._find(self._sessions, session_id, agent_id, None)
            if not session:
                return False
            session.last_viewer_heartbeat_at = time.monotonic()
            session.last_viewer_heartbeat_ts = int(time.time())
            return True

    async def status(
        self,
        *,
        session_id: str | None,
        agent_id: str | None,
        channel_name: str | None = None,
    ) -> SessionStatusResponse:
        async with self._lock:
            session = self._find(
                self._sessions,
                session_id,
                agent_id,
                channel_name,
            ) or self._find(self._records, session_id, agent_id, channel_name)
            if not session:
                return SessionStatusResponse(
                    success=False,
                    state="missing",
                    transcription_state="missing",
                    live_session_max_seconds=self._settings.live_session_max_seconds,
                    viewer_heartbeat_timeout_seconds=(
                        self._settings.viewer_heartbeat_timeout_seconds
                    ),
                )
            running = session.session_id in self._sessions
            stats = session.transcriber.stats()
            transcript_revision, recent_transcript = (
                session.transcriber.transcript_diagnostics()
            )
            return SessionStatusResponse(
                success=True,
                state="running" if running else "stopped",
                transcription_state=(
                    "active"
                    if running and stats.audio_frames
                    else "waiting_for_audio"
                    if running
                    else "stopped"
                ),
                session_id=session.session_id,
                agent_id=session.agent_id,
                channel_name=session.channel_name,
                transcriber_id=session.transcriber_id,
                transcription_mode=session.transcription_mode,
                model=session.model,
                created_at=session.created_at,
                stopped_at=session.stopped_at,
                stop_reason=session.stop_reason,
                last_viewer_heartbeat_at=session.last_viewer_heartbeat_ts,
                last_viewer_heartbeat_age_seconds=max(
                    0.0,
                    time.monotonic() - session.last_viewer_heartbeat_at,
                ),
                live_session_max_seconds=self._settings.live_session_max_seconds,
                viewer_heartbeat_timeout_seconds=(
                    self._settings.viewer_heartbeat_timeout_seconds
                ),
                transcript_revision=transcript_revision,
                recent_transcript=[
                    {
                        "id": item.id,
                        "text": item.text,
                        "state": item.state,
                        "created_at": item.created_at,
                        "emitted_at": item.emitted_at,
                    }
                    for item in recent_transcript
                ],
                stats=stats,
                events=list(session.events)[-12:],
            )

    async def close(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await self._stop_sessions(sessions, "backend shutdown")

    async def _monitor(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(1)
            async with self._lock:
                session = self._sessions.get(session_id)
                if not session:
                    return
                age = time.monotonic() - session.created_at_monotonic
                heartbeat_age = time.monotonic() - session.last_viewer_heartbeat_at
                reason = None
                if age >= self._settings.live_session_max_seconds:
                    reason = "Live transcription session limit reached"
                elif heartbeat_age >= self._settings.viewer_heartbeat_timeout_seconds:
                    reason = "viewer heartbeat timed out"
                if reason:
                    self._sessions.pop(session_id, None)
            if reason:
                await self._stop_sessions([session], reason)
                return

    async def _stop_sessions(
        self,
        sessions: list[LiveSession],
        reason: str,
    ) -> None:
        current = asyncio.current_task()
        monitors = [
            item.monitor_task
            for item in sessions
            if item.monitor_task and item.monitor_task is not current
        ]
        for task in monitors:
            task.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)
        results = await asyncio.gather(
            *(item.transcriber.stop() for item in sessions),
            return_exceptions=True,
        )
        for session, result in zip(sessions, results):
            session.stopped_at = int(time.time())
            session.stop_reason = reason
            if isinstance(result, Exception):
                self._emit(session, "stop_failed", type(result).__name__)
            else:
                self._emit(session, "session_stopped", reason)

    @staticmethod
    def _find(
        sessions: dict[str, LiveSession],
        session_id: str | None,
        agent_id: str | None,
        channel_name: str | None,
    ) -> LiveSession | None:
        if session_id and session_id in sessions:
            return sessions[session_id]
        values = list(sessions.values())
        if agent_id:
            return next((item for item in values if item.agent_id == agent_id), None)
        if channel_name:
            return next(
                (item for item in reversed(values) if item.channel_name == channel_name),
                None,
            )
        return None

    @staticmethod
    def _emit(session: LiveSession, event: str, message: str) -> None:
        session.events.append(
            SessionLifecycleEvent(
                event=event,
                message=message,
                created_at=int(time.time()),
            )
        )
