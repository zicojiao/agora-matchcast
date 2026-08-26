import asyncio
from dataclasses import dataclass
import json
import logging
import os
from types import SimpleNamespace
import time

from agora_agent.agentkit.token import ROLE_PUBLISHER, generate_rtc_token

from .agora_region import area_code_value
from .config import Settings
from .gemini_transcriber import (
    CHUNK_MS,
    CaptionEvent,
    GeminiLiveTranscriber,
    Pcm16Normalizer,
    TranscriptPatchEvent,
    TranscriptRevisionEvent,
    TranscriptionEvent,
)
from .models import TranscriberStats
from .transcriber_selection import TranscriptionMode
from .transcription_diagnostics import (
    BoundedPcmWaveCapture,
    DiagnosticTranscriptDocument,
)

logger = logging.getLogger(__name__)
AUDIO_QUEUE_MAX_CHUNKS = 3_000 // CHUNK_MS


def _gemini_transcriber_type_for_mode(mode: TranscriptionMode):
    if mode == "gemini-live":
        return GeminiLiveTranscriber
    raise ValueError(f"Mode {mode!r} is not a Gemini transcription mode.")


def caption_payload(
    event: CaptionEvent,
    *,
    uid: int,
    turn_id: int,
) -> bytes:
    return json.dumps(
        {
            "object": "matchcast.caption",
            "text": event.text,
            "final": event.is_final,
            "language": event.language,
            "turn_id": turn_id,
            "user_id": str(uid),
            "created_at": int(time.time() * 1000),
        },
        ensure_ascii=False,
    ).encode("utf-8")


def transcript_revision_payload(
    event: TranscriptRevisionEvent,
    *,
    uid: int,
) -> bytes:
    segment = None
    if event.segment is not None:
        segment = {
            "id": event.segment.id,
            "text": event.segment.text,
            "state": event.segment.state,
            "created_at": event.segment.created_at,
        }
    return json.dumps(
        {
            "object": "matchcast.transcript.segment",
            "revision": event.revision,
            "replace_from": event.replace_from,
            "index": event.index,
            "total": event.total,
            "segment": segment,
            "user_id": str(uid),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def transcript_patch_payload(
    event: TranscriptPatchEvent,
    *,
    uid: int,
) -> bytes:
    return json.dumps(
        {
            "object": "matchcast.transcript.patch",
            "revision": event.revision,
            "drop_from_start": event.drop_from_start,
            "replace_from": event.replace_from,
            "total": event.total,
            "segments": [
                {
                    "id": segment.id,
                    "text": segment.text,
                    "state": segment.state,
                    "created_at": segment.created_at,
                }
                for segment in event.segments
            ],
            "user_id": str(uid),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class _RemoteAudioObserver:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[bytes],
        media_uid: int,
        on_frame,
        on_pcm=None,
    ) -> None:
        self._loop = loop
        self._queue = queue
        self._media_uid = str(media_uid)
        self._normalizer = Pcm16Normalizer()
        self._on_frame = on_frame
        self._on_pcm = on_pcm

    def on_playback_audio_frame_before_mixing(
        self,
        _local_user,
        _channel_id,
        uid,
        frame,
        _vad_state,
        _vad_bytes,
    ) -> int:
        if str(uid) != self._media_uid:
            return 1
        pcm = bytes(frame.buffer)
        try:
            chunks = self._normalizer.push(
                pcm,
                sample_rate=frame.samples_per_sec,
                channels=frame.channels,
            )
        except ValueError:
            logger.warning("Ignored unsupported Agora audio frame channels=%s", frame.channels)
            return 1
        for chunk in chunks:
            if self._on_pcm is not None:
                self._on_pcm(chunk)
            self._loop.call_soon_threadsafe(self._offer, chunk)
        return 1

    def _offer(self, chunk: bytes) -> None:
        try:
            self._queue.put_nowait(chunk)
            self._on_frame(False)
        except asyncio.QueueFull:
            # Live captions should skip stale audio instead of building delay.
            self._queue.get_nowait()
            self._on_frame(True)
            self._queue.put_nowait(chunk)
            self._on_frame(False)

    def on_get_audio_frame_position(self, _local_user) -> int:
        # Playback-before-mixing gives us an isolated frame for the Media Gateway UID.
        return 0x08

    def on_get_playback_audio_frame_param(self, _local_user):
        from agora.rtc.agora_base import AudioParams

        return AudioParams(
            sample_rate=16_000,
            channels=1,
            mode=0,
            samples_per_call=1600,
        )

    def on_record_audio_frame(self, *_args) -> int:
        return 1

    def on_playback_audio_frame(self, *_args) -> int:
        return 1

    def on_mixed_audio_frame(self, *_args) -> int:
        return 1

    def on_ear_monitoring_audio_frame(self, *_args) -> int:
        return 1


class MatchCastTranscriber:
    def __init__(
        self,
        *,
        settings: Settings,
        channel_name: str,
        transcriber_uid: int,
        media_uid: int,
        transcription_mode: TranscriptionMode,
        model: str,
    ) -> None:
        self._settings = settings
        self._channel_name = channel_name
        self._transcriber_uid = transcriber_uid
        self._media_uid = media_uid
        self._transcription_mode = transcription_mode
        self._model = model
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=AUDIO_QUEUE_MAX_CHUNKS
        )
        self._connection = None
        self._audio_frames = 0
        self._dropped_frames = 0
        self._interim_captions = 0
        self._final_captions = 0
        self._last_audio_at: int | None = None
        self._last_caption_at: int | None = None
        self._turn_id = 1
        self._diagnostic_document = DiagnosticTranscriptDocument()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(),
                name=f"matchcast-{self._channel_name}",
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=8)
            except asyncio.TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    def stats(self) -> TranscriberStats:
        return TranscriberStats(
            audio_frames=self._audio_frames,
            audio_ms=self._audio_frames * CHUNK_MS,
            interim_captions=self._interim_captions,
            final_captions=self._final_captions,
            dropped_frames=self._dropped_frames,
            last_audio_at=self._last_audio_at,
            last_caption_at=self._last_caption_at,
        )

    def transcript_diagnostics(self):
        return (
            self._diagnostic_document.revision,
            self._diagnostic_document.segments,
        )

    def _record_frame(self, dropped: bool) -> None:
        if dropped:
            self._dropped_frames += 1
        else:
            self._audio_frames += 1
            self._last_audio_at = int(time.time())

    async def _emit_transcription(self, event: TranscriptionEvent) -> None:
        if self._connection is None:
            return
        emitted_at_ms = int(time.time() * 1000)
        if not self._diagnostic_document.apply(
            event,
            emitted_at=emitted_at_ms,
        ):
            logger.warning(
                "Ignored malformed or stale transcript event type=%s",
                type(event).__name__,
            )
            return
        if isinstance(event, TranscriptPatchEvent):
            payload = transcript_patch_payload(
                event,
                uid=self._transcriber_uid,
            )
            self._final_captions += sum(
                segment.state == "caption"
                for segment in event.segments
            )
            self._interim_captions += sum(
                segment.state == "listening"
                for segment in event.segments
            )
        elif isinstance(event, TranscriptRevisionEvent):
            payload = transcript_revision_payload(
                event,
                uid=self._transcriber_uid,
            )
            if event.segment is not None and event.segment.state == "caption":
                self._final_captions += 1
            else:
                self._interim_captions += 1
        else:
            if not event.text.strip():
                return
            payload = caption_payload(
                event,
                uid=self._transcriber_uid,
                turn_id=self._turn_id,
            )
            if event.is_final:
                self._final_captions += 1
                self._turn_id += 1
            else:
                self._interim_captions += 1
        if len(payload) >= 1024:
            logger.error(
                "Agora transcript stream message exceeds 1024 bytes size=%s",
                len(payload),
            )
            return
        ret = self._connection.send_stream_message(payload)
        if ret != 0:
            logger.warning("Agora caption stream message failed ret=%s", ret)
        self._last_caption_at = int(time.time())

    async def _run(self) -> None:
        sdk = self._load_sdk()
        os.makedirs(self._settings.log_dir, exist_ok=True)
        service_config = sdk.AgoraServiceConfig()
        service_config.appid = self._settings.agora_app_id
        service_config.enable_audio_processor = 1
        service_config.enable_audio_device = 0
        service_config.enable_video = 0
        service_config.log_path = os.path.join(self._settings.log_dir, "agorasdk.log")
        service_config.data_dir = self._settings.log_dir
        service_config.config_dir = self._settings.log_dir
        service_config.area_code = area_code_value(
            self._settings.agora_area_code,
            sdk.AreaCode,
        )

        service = sdk.AgoraService()
        connection = None
        observer = None
        pcm_capture = (
            BoundedPcmWaveCapture(
                self._settings.transcription_pcm_capture_path,
                max_seconds=(
                    self._settings.transcription_pcm_capture_max_seconds
                ),
            )
            if self._settings.transcription_pcm_capture_path
            else None
        )
        try:
            service.initialize(service_config)
            connection = service.create_rtc_connection(
                sdk.RTCConnConfig(
                    auto_subscribe_audio=0,
                    auto_subscribe_video=0,
                    client_role_type=sdk.ClientRoleType.CLIENT_ROLE_BROADCASTER,
                    channel_profile=sdk.ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
                    enable_audio_recording_or_playout=1,
                ),
                sdk.RtcConnectionPublishConfig(
                    audio_profile=sdk.AudioProfileType.AUDIO_PROFILE_DEFAULT,
                    audio_scenario=sdk.AudioScenarioType.AUDIO_SCENARIO_AI_SERVER,
                    audio_publish_type=sdk.AudioPublishType.AUDIO_PUBLISH_TYPE_NONE,
                    video_publish_type=sdk.VideoPublishType.VIDEO_PUBLISH_TYPE_NONE,
                    is_publish_audio=False,
                    is_publish_video=False,
                ),
            )
            token = generate_rtc_token(
                app_id=self._settings.agora_app_id,
                app_certificate=self._settings.agora_app_certificate,
                channel=self._channel_name,
                uid=self._transcriber_uid,
                role=ROLE_PUBLISHER,
                expiry_seconds=self._settings.token_expire_seconds,
            )
            connection.connect(token, self._channel_name, str(self._transcriber_uid))
            self._connection = connection
            local_user = connection.get_local_user()
            local_user.set_playback_audio_frame_before_mixing_parameters(1, 16_000)
            observer = _RemoteAudioObserver(
                loop=asyncio.get_running_loop(),
                queue=self._audio_queue,
                media_uid=self._media_uid,
                on_frame=self._record_frame,
                on_pcm=pcm_capture.write if pcm_capture is not None else None,
            )
            register_ret = connection.register_audio_frame_observer(observer, 0, None)
            subscribe_ret = local_user.subscribe_audio(str(self._media_uid))
            logger.info(
                "MatchCast listening channel=%s media_uid=%s mode=%s register=%s subscribe=%s",
                self._channel_name,
                self._media_uid,
                self._transcription_mode,
                register_ret,
                subscribe_ret,
            )

            if not self._settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required.")
            transcriber_type = _gemini_transcriber_type_for_mode(
                self._transcription_mode
            )
            transcriber_options = dict(
                api_key=self._settings.gemini_api_key,
                model=self._model,
                language=self._settings.gemini_language,
            )
            transcriber_options["vocabulary"] = (
                self._settings.gemini_transcribe_vocabulary
            )
            transcriber_options["vocabulary_mode"] = self._settings.gemini_vocabulary_mode
            transcriber_options["transcription_mode"] = self._settings.gemini_transcription_mode
            transcriber_options["activity_min_ms"] = self._settings.gemini_transcribe_activity_min_ms
            transcriber_options["activity_max_ms"] = self._settings.gemini_transcribe_activity_max_ms
            transcriber_options["activity_prefix_ms"] = self._settings.gemini_transcribe_activity_prefix_ms
            transcriber_options["activity_low_energy_ratio"] = self._settings.gemini_transcribe_activity_low_energy_ratio
            transcriber_options["activity_commit_grace_seconds"] = self._settings.gemini_transcribe_activity_commit_grace_seconds
            transcriber_options["activity_handoff_seconds"] = self._settings.gemini_transcribe_activity_handoff_seconds
            transcriber_options["activity_buffer_max_ms"] = self._settings.gemini_transcribe_activity_buffer_max_ms
            transcriber = transcriber_type(**transcriber_options)
            await transcriber.run(
                self._audio_queue,
                self._emit_transcription,
                self._stop,
            )
        finally:
            self._connection = None
            if pcm_capture is not None:
                captured_path = pcm_capture.close()
                if captured_path is not None:
                    logger.info("Captured normalized Agora PCM path=%s", captured_path)
            if connection is not None:
                try:
                    local_user = connection.get_local_user()
                    local_user.unsubscribe_audio(str(self._media_uid))
                    connection._unregister_audio_frame_observer()
                    connection.disconnect()
                    connection.release()
                except Exception:
                    logger.warning("Failed to release MatchCast RTC connection", exc_info=True)
            try:
                service.release()
            except Exception:
                logger.warning("Failed to release Agora service", exc_info=True)

    @staticmethod
    def _load_sdk() -> SimpleNamespace:
        from agora.rtc.agora_base import (
            AreaCode,
            AudioProfileType,
            AudioPublishType,
            AudioScenarioType,
            ChannelProfileType,
            ClientRoleType,
            RtcConnectionPublishConfig,
            VideoPublishType,
        )
        from agora.rtc.agora_service import AgoraService, AgoraServiceConfig, RTCConnConfig

        return SimpleNamespace(
            AgoraService=AgoraService,
            AgoraServiceConfig=AgoraServiceConfig,
            RTCConnConfig=RTCConnConfig,
            RtcConnectionPublishConfig=RtcConnectionPublishConfig,
            AreaCode=AreaCode,
            AudioProfileType=AudioProfileType,
            AudioPublishType=AudioPublishType,
            AudioScenarioType=AudioScenarioType,
            ChannelProfileType=ChannelProfileType,
            ClientRoleType=ClientRoleType,
            VideoPublishType=VideoPublishType,
        )
