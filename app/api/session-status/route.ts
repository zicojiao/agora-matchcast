import { NextRequest, NextResponse } from 'next/server';
import {
  SessionStatusRequest,
  SessionStatusResponse,
} from '@/types/conversation';
import { getBackendHeaders, getBackendUrl } from '@/lib/backend';

export async function POST(request: NextRequest) {
  try {
    const body: SessionStatusRequest = await request.json();

    if (!body.agent_id && !body.session_id && !body.channel_name) {
      return NextResponse.json(
        { error: 'agent_id, session_id, or channel_name is required' },
        { status: 400 },
      );
    }

    const backendResponse = await fetch(`${getBackendUrl()}/sessions/status`, {
      method: 'POST',
      headers: getBackendHeaders(),
      body: JSON.stringify({
        agent_id: body.agent_id,
        session_id: body.session_id,
        channel_name: body.channel_name,
      }),
      cache: 'no-store',
    });

    const payload = await backendResponse.json().catch(() => null);
    if (!backendResponse.ok) {
      return NextResponse.json(
        {
          error: payload?.detail ?? payload?.error ?? 'Failed to read session status',
          statusCode: backendResponse.status,
        },
        { status: backendResponse.status },
      );
    }

    return NextResponse.json(payload as SessionStatusResponse);
  } catch (error) {
    console.error('Error reading session status:', error);
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : 'Failed to read session status',
      },
      { status: 500 },
    );
  }
}
