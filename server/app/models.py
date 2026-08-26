from typing import Literal

from pydantic import BaseModel, Field

from .transcriber_selection import (
    TranscriberId,
    TranscriptionMode,
)


class StartSessionRequest(BaseModel):
    requester_id: str = Field(min_length=1)
    channel_name: str = Field(min_length=1)
    source_mode: Literal["agora-gateway"] = "agora-gateway"
    transcriber_uid: int | None = None
    media_uid: int | None = None
    transcriber_id: TranscriberId | None = None


class StartSessionResponse(BaseModel):
    session_id: str
    agent_id: str
    create_ts: int
    state: Literal["RUNNING"]
    channel_name: str
    source_mode: Literal["agora-gateway"] = "agora-gateway"
    agent_uid: str
    media_uid: str
    transcriber_id: TranscriberId | None = None
    transcription_mode: TranscriptionMode
    model: str


class StopSessionRequest(BaseModel):
    session_id: str | None = None
    agent_id: str | None = None


class StopSessionResponse(BaseModel):
    success: bool
    state: str = "stopped"


class HeartbeatSessionRequest(BaseModel):
    session_id: str | None = None
    agent_id: str | None = None


class HeartbeatSessionResponse(BaseModel):
    success: bool
    state: Literal["running", "missing"]


class SessionStatusRequest(BaseModel):
    session_id: str | None = None
    agent_id: str | None = None
    channel_name: str | None = None


class SessionLifecycleEvent(BaseModel):
    event: str
    message: str
    created_at: int


class TranscriberStats(BaseModel):
    audio_frames: int = 0
    audio_ms: int = 0
    interim_captions: int = 0
    final_captions: int = 0
    dropped_frames: int = 0
    last_audio_at: int | None = None
    last_caption_at: int | None = None


class DiagnosticTranscriptSegment(BaseModel):
    id: str
    text: str
    state: Literal["caption", "listening"]
    created_at: int
    emitted_at: int


class SessionStatusResponse(BaseModel):
    success: bool
    state: Literal["running", "stopped", "missing"]
    transcription_state: Literal[
        "active", "waiting_for_audio", "stopped", "missing"
    ]
    session_id: str | None = None
    agent_id: str | None = None
    channel_name: str | None = None
    transcriber_id: TranscriberId | None = None
    transcription_mode: str | None = None
    model: str | None = None
    created_at: int | None = None
    stopped_at: int | None = None
    stop_reason: str | None = None
    last_viewer_heartbeat_at: int | None = None
    last_viewer_heartbeat_age_seconds: float | None = None
    live_session_max_seconds: float | None = None
    viewer_heartbeat_timeout_seconds: float | None = None
    transcript_revision: int = 0
    recent_transcript: list[DiagnosticTranscriptSegment] = Field(
        default_factory=list
    )
    stats: TranscriberStats = Field(default_factory=TranscriberStats)
    events: list[SessionLifecycleEvent] = Field(default_factory=list)
