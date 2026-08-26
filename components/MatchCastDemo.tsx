'use client';

import Image from 'next/image';
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type {
  IAgoraRTCClient,
  IAgoraRTCRemoteUser,
} from 'agora-rtc-sdk-ng';
import {
  Captions,
  ChevronRight,
  Download,
  Radio,
  Signal,
  Square,
} from 'lucide-react';
import {
  applyTranscriptMessage,
  createTranscriptState,
  decodeTranscriptMessage,
  visibleTranscriptSegments,
} from '@/lib/captions';
import {
  completeSessionCapture,
  createSessionCapture,
  identifySession,
  markSessionTiming,
  recordCaptionMessage,
  sessionCaptureFilename,
  sessionCaptureToCsv,
  type SessionCapture,
} from '@/lib/session-export';
import { scrollTranscriptToLatest } from '@/lib/transcript-scroll';
import {
  type TranscriberId,
  type TranscriberOption,
} from '@/lib/transcribers';
import TranscriberSelect from '@/components/TranscriberSelect';

type AgentSession = {
  session_id: string;
  agent_id: string;
  transcriber_id: TranscriberId | null;
  transcription_mode: 'gemini-live';
  model: string;
};

type Stage = 'idle' | 'joining' | 'live' | 'error';

function displayModel(model?: string) {
  if (!model) return 'WAITING FOR SESSION';
  return model.replace(/^models\//, '').replaceAll('-', ' ').toUpperCase();
}

type MatchCastDemoProps = {
  transcribers: readonly TranscriberOption[];
};

export default function MatchCastDemo({ transcribers }: MatchCastDemoProps) {
  const [stage, setStage] = useState<Stage>('idle');
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<AgentSession | null>(null);
  const [signal, setSignal] = useState<'waiting' | 'receiving'>('waiting');
  const [selectedTranscriberId, setSelectedTranscriberId] =
    useState<TranscriberId>(transcribers[0]?.value ?? 'gemini-transcribe');
  const [transcriptState, setTranscriptState] = useState(
    createTranscriptState,
  );
  const [hasSessionCapture, setHasSessionCapture] = useState(false);
  const videoRef = useRef<HTMLDivElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const clientRef = useRef<IAgoraRTCClient | null>(null);
  const sessionRef = useRef<AgentSession | null>(null);
  const captureRef = useRef<SessionCapture | null>(null);
  const captureStartedAtRef = useRef<number | null>(null);
  const acceptingCaptionsRef = useRef(false);
  const transcript = useMemo(
    () => visibleTranscriptSegments(transcriptState),
    [transcriptState],
  );

  useLayoutEffect(() => {
    if (transcriptRef.current) {
      scrollTranscriptToLatest(transcriptRef.current);
    }
  }, [transcriptState]);

  const captureElapsedMs = useCallback(() => {
    const startedAt = captureStartedAtRef.current;
    return startedAt === null ? 0 : performance.now() - startedAt;
  }, []);

  const stop = useCallback(async () => {
    const active = sessionRef.current;
    const client = clientRef.current;
    acceptingCaptionsRef.current = false;
    clientRef.current = null;
    sessionRef.current = null;
    client?.removeAllListeners();
    if (captureRef.current) {
      completeSessionCapture(
        captureRef.current,
        new Date().toISOString(),
        captureElapsedMs(),
      );
    }
    setStage('idle');
    setSignal('waiting');
    await Promise.all([
      active
        ? fetch('/api/stop-conversation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: active.session_id,
              agent_id: active.agent_id,
            }),
          }).catch(() => {})
        : Promise.resolve(),
      client?.leave().catch(() => {}),
    ]);
  }, [captureElapsedMs]);

  useEffect(() => {
    const handlePageHide = () => {
      const active = sessionRef.current;
      acceptingCaptionsRef.current = false;
      sessionRef.current = null;
      if (active) {
        const payload = JSON.stringify({
          session_id: active.session_id,
          agent_id: active.agent_id,
        });
        navigator.sendBeacon(
          '/api/stop-conversation',
          new Blob([payload], { type: 'application/json' }),
        );
      }
      clientRef.current?.leave().catch(() => {});
      clientRef.current = null;
    };
    window.addEventListener('pagehide', handlePageHide);
    return () => {
      window.removeEventListener('pagehide', handlePageHide);
    };
  }, []);

  useEffect(() => {
    if (!session || stage !== 'live') return;
    const timer = window.setInterval(() => {
      fetch('/api/session-heartbeat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          agent_id: session.agent_id,
        }),
      }).catch(() => {});
    }, 15_000);
    return () => window.clearInterval(timer);
  }, [session, stage]);

  const start = async () => {
    if (stage === 'joining') return;
    const captureStartedAt = performance.now();
    const sessionCapture = createSessionCapture(
      selectedTranscriberId,
      new Date().toISOString(),
      crypto.randomUUID(),
    );
    captureRef.current = sessionCapture;
    captureStartedAtRef.current = captureStartedAt;
    setHasSessionCapture(true);
    acceptingCaptionsRef.current = false;
    setStage('joining');
    setError(null);
    setSession(null);
    setTranscriptState(createTranscriptState());
    try {
      const tokenResponse = await fetch('/api/generate-agora-token');
      const tokenData = await tokenResponse.json();
      if (!tokenResponse.ok) {
        throw new Error(tokenData.error ?? 'Could not create an Agora token.');
      }
      markSessionTiming(
        sessionCapture,
        'tokenReadyMs',
        performance.now() - captureStartedAt,
      );

      const { default: AgoraRTC } = await import('agora-rtc-sdk-ng');
      const client = AgoraRTC.createClient({ mode: 'live', codec: 'h264' });
      clientRef.current = client;
      await client.setClientRole('audience');
      client.on('token-privilege-will-expire', async () => {
        try {
          const renewalResponse = await fetch(
            `/api/generate-agora-token?uid=${encodeURIComponent(String(tokenData.uid))}`,
          );
          const renewal = await renewalResponse.json();
          if (!renewalResponse.ok || !renewal.token) {
            throw new Error(renewal.error ?? 'Could not renew the Agora token.');
          }
          await client.renewToken(renewal.token);
        } catch (caught) {
          console.error('Agora token renewal failed:', caught);
          setError('The Agora session could not renew its access token.');
        }
      });
      client.on(
        'user-published',
        async (user: IAgoraRTCRemoteUser, mediaType: 'audio' | 'video') => {
          await client.subscribe(user, mediaType);
          markSessionTiming(
            sessionCapture,
            'mediaSignalMs',
            performance.now() - captureStartedAt,
          );
          setSignal('receiving');
          if (mediaType === 'video' && user.videoTrack && videoRef.current) {
            user.videoTrack.play(videoRef.current, { fit: 'contain' });
          }
          if (mediaType === 'audio' && user.audioTrack) {
            user.audioTrack.play();
          }
        },
      );
      client.on('user-unpublished', () => setSignal('waiting'));
      client.on('stream-message', (_uid: number, data: Uint8Array) => {
        if (!acceptingCaptionsRef.current) return;
        const next = decodeTranscriptMessage(data);
        if (!next) return;
        recordCaptionMessage(
          sessionCapture,
          next,
          new Date().toISOString(),
          performance.now() - captureStartedAt,
        );
        setTranscriptState((state) => applyTranscriptMessage(state, next));
      });
      await client.join(
        process.env.NEXT_PUBLIC_AGORA_APP_ID!,
        tokenData.channel,
        tokenData.token,
        tokenData.uid,
      );
      markSessionTiming(
        sessionCapture,
        'rtcJoinedMs',
        performance.now() - captureStartedAt,
      );
      // The transcriber can publish its first revision before the start API
      // response reaches the browser. Listen as soon as RTC is joined so the
      // canonical revision baseline cannot be missed.
      acceptingCaptionsRef.current = true;

      const startResponse = await fetch('/api/invite-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requester_id: String(tokenData.uid),
          channel_name: tokenData.channel,
          transcriber_id: selectedTranscriberId,
        }),
      });
      const started = await startResponse.json();
      if (!startResponse.ok) {
        throw new Error(started.error ?? 'Transcriber failed to start.');
      }
      const activeSession = started as AgentSession;
      identifySession(sessionCapture, activeSession);
      markSessionTiming(
        sessionCapture,
        'transcriberStartedMs',
        performance.now() - captureStartedAt,
      );
      sessionRef.current = activeSession;
      setSession(activeSession);
      setStage('live');
    } catch (caught) {
      acceptingCaptionsRef.current = false;
      completeSessionCapture(
        sessionCapture,
        new Date().toISOString(),
        performance.now() - captureStartedAt,
      );
      const failedClient = clientRef.current;
      failedClient?.removeAllListeners();
      await failedClient?.leave().catch(() => {});
      clientRef.current = null;
      sessionRef.current = null;
      setSignal('waiting');
      setStage('error');
      setError(caught instanceof Error ? caught.message : 'Could not start MatchCast.');
    }
  };

  const downloadSessionCsv = () => {
    const capture = captureRef.current;
    if (!capture) return;
    const csv = sessionCaptureToCsv(
      capture,
      transcript,
      new Date().toISOString(),
      captureElapsedMs(),
    );
    const url = URL.createObjectURL(
      new Blob([csv], { type: 'text/csv;charset=utf-8' }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = sessionCaptureFilename(capture);
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const latestIndex = transcript.length - 1;
  const latestCaption = transcript.at(-1);
  const transcriberSelectionDisabled = stage === 'joining' || stage === 'live';
  const activeProvider = 'Gemini';
  const modeLabel = session
    ? 'GEMINI TRANSCRIBE LIVE'
    : 'TRANSCRIPTION STANDBY';

  return (
    <main className="matchcast-shell">
      <header className="matchcast-header">
        <div className="matchcast-brand">
          <Image
            className="matchcast-mark"
            src="/agora-icon-rgb-blue.svg"
            alt="Agora"
            width={38}
            height={38}
            priority
          />
          <h1>Agora MatchCast</h1>
        </div>
        <div className="matchcast-tech" aria-label="Signal path">
          <span>AGORA MEDIA GATEWAY</span>
          <ChevronRight />
          <span>AGORA RTC</span>
          <ChevronRight />
          <span>AI TRANSCRIPTION</span>
        </div>
        <div className={`header-signal ${signal}`}>
          <i />
          <span>{signal === 'receiving' ? 'ON AIR' : 'STANDBY'}</span>
        </div>
      </header>

      <section className="matchcast-workspace" aria-label="Live match and transcript">
        <div className="broadcast-pane">
          <div className="broadcast-video">
            <div ref={videoRef} className="video-player" />
            {signal === 'receiving' && latestCaption ? (
              <div
                className={[
                  'broadcast-caption',
                  latestCaption.state === 'caption'
                    ? 'is-final'
                    : 'is-interim',
                ].join(' ')}
                aria-hidden="true"
              >
                <span>
                  <i />
                  LIVE CAPTIONS
                </span>
                <p>{latestCaption.text}</p>
              </div>
            ) : null}
            {signal === 'waiting' ? (
              <div className="broadcast-standby">
                <Signal />
                <p>
                  {stage === 'joining'
                    ? 'Joining Agora RTC…'
                    : 'Agora live feed ready'}
                </p>
                <span>
                  {stage === 'idle' || stage === 'error'
                    ? 'Enter the Agora channel to watch the live replay'
                    : 'Waiting for Agora Media Gateway to publish'}
                </span>
                {stage === 'idle' || stage === 'error' ? (
                  <button className="watch-live-button" onClick={start}>
                    <Radio />
                    {stage === 'error' ? 'TRY AGAIN' : 'WATCH LIVE'}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>

          <div className="scorebug">
            <span className="live-pill"><i /> LIVE</span>
          </div>
        </div>

        <aside className="transcript-panel" aria-label="Live transcript">
          <div className="transcript-header">
            <div>
              <span className={`status-dot ${signal}`} />
              <div>
                <p>REAL-TIME CAPTIONS</p>
                <strong>{modeLabel} VIA AGORA</strong>
              </div>
            </div>
            <Captions aria-hidden="true" />
          </div>

          <div
            ref={transcriptRef}
            className="transcript-feed"
            aria-live="polite"
          >
            {transcript.length ? (
              transcript.map((item, index) => (
                <article
                  className={[
                    'transcript-entry',
                    item.state === 'caption' ? 'is-final' : 'is-interim',
                    index === latestIndex ? 'is-current' : '',
                  ].join(' ')}
                  key={item.id}
                >
                  <div>
                    <span>
                      {item.state === 'caption' ? 'CAPTION' : 'LISTENING'}
                    </span>
                    <time>
                      {new Date(item.created_at).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}
                    </time>
                  </div>
                  <p>{item.text}</p>
                </article>
              ))
            ) : (
              <div className="transcript-empty">
                <Captions />
                <p>
                  {error
                    ? error
                    : stage === 'live'
                      ? 'Agora is carrying the live commentary…'
                      : 'Live captions will appear here as the match plays.'}
                </p>
                <span>
                  {stage === 'live'
                    ? `${activeProvider} captions are returned over the Agora RTC channel.`
                    : 'Agora delivers the match while captions scroll beside it.'}
                </span>
              </div>
            )}
          </div>

          <div className="transcript-model">
            <span>
              {stage === 'live'
                ? 'ACTIVE TRANSCRIPTION MODEL'
                : session
                  ? 'LAST SESSION MODEL'
                  : 'ACTIVE TRANSCRIPTION MODEL'}
            </span>
            <strong>{displayModel(session?.model)}</strong>
          </div>
        </aside>
      </section>

      <footer className="matchcast-controlbar">
        <div className="control-statuses">
          <div>
            <span className={`status-dot ${signal}`} />
            <span>AGORA INGEST</span>
            <strong>{signal === 'receiving' ? 'SIGNAL LOCKED' : 'WAITING'}</strong>
          </div>
          <div>
            <Radio />
            <span>AI TRANSCRIPTION</span>
            <strong>
              {stage === 'live'
                ? modeLabel
                : session
                  ? `${modeLabel} ENDED`
                  : 'NOT STARTED'}
            </strong>
          </div>
        </div>

        <div className="control-actions">
          <TranscriberSelect
            disabled={transcriberSelectionDisabled}
            onValueChange={setSelectedTranscriberId}
            options={transcribers}
            value={selectedTranscriberId}
          />
          <button
            className="download-action"
            disabled={!hasSessionCapture}
            onClick={downloadSessionCsv}
            title="Download this Watch Live session as CSV"
          >
            <Download />
            DOWNLOAD CSV
          </button>
          {stage === 'idle' || stage === 'error' ? (
            <button className="primary-action" onClick={start}>
              <Radio />
              {stage === 'error' ? 'TRY AGAIN' : 'WATCH LIVE'}
            </button>
          ) : stage === 'joining' ? (
            <button className="primary-action" disabled>
              <Signal className="pulse" />
              LOCKING SIGNAL…
            </button>
          ) : (
            <button className="end-action" onClick={stop}>
              <Square />
              END SESSION
            </button>
          )}
        </div>
      </footer>
    </main>
  );
}
