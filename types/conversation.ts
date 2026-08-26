import type { TranscriberId } from '@/lib/transcribers';

export interface ClientStartRequest {
  requester_id: string;
  channel_name: string;
  transcriber_id: TranscriberId;
}

export interface AgentResponse {
  agent_id: string;
  session_id: string;
  create_ts: number;
  state: string;
  channel_name: string;
  agent_uid: string;
  media_uid: string;
  transcriber_id: TranscriberId | null;
  transcription_mode: 'gemini-live';
  model: string;
}

export interface AgentErrorResponse {
  error: string;
  detail?: string;
  statusCode?: number;
}

export interface StopConversationRequest {
  agent_id?: string;
  session_id?: string;
}

export interface SessionHeartbeatRequest {
  agent_id?: string;
  session_id?: string;
}

export interface SessionHeartbeatResponse {
  success: boolean;
  state: 'running' | 'missing';
}

export interface SessionStatusRequest {
  agent_id?: string;
  session_id?: string;
  channel_name?: string;
}

export interface SessionStatusResponse {
  success: boolean;
  state: 'running' | 'stopped' | 'missing';
  transcription_state: 'active' | 'waiting_for_audio' | 'stopped' | 'missing';
  transcriber_id?: TranscriberId | null;
  transcription_mode?: AgentResponse['transcription_mode'];
  model?: string;
}
