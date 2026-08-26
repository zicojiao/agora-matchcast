# Agora MatchCast backend

This FastAPI service joins an Agora RTC channel, subscribes to the Media
Gateway publisher's audio, normalizes it to 16 kHz mono PCM, and sends it to
Gemini 3.5 Transcribe Live. Captions are published back to viewers through an
Agora RTC data stream.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env.local
```

Set the Agora credentials, the Media Gateway publisher UID, the shared backend
secret, and the Gemini API key in `.env.local`:

```bash
AGORA_APP_ID=
AGORA_APP_CERTIFICATE=
MEDIA_UID=234567
BACKEND_API_SECRET=

GEMINI_API_KEY=
GEMINI_MODEL=models/gemini-3.5-transcribe-live
GEMINI_LANGUAGE=en-US
GEMINI_TRANSCRIPTION_MODE=smart
```

Run the service:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Gemini Transcribe Live

The backend uses one transcription path, `gemini-transcribe`, backed by
`models/gemini-3.5-transcribe-live`. It supports custom vocabulary, automatic
language identification, smart or verbatim output, bounded audio activities,
final-result handoff buffering, and revisable transcript updates.

Useful tuning variables are documented in [`.env.example`](./.env.example).

## Routes

- `GET /health`
- `POST /sessions/start`
- `POST /sessions/heartbeat`
- `POST /sessions/status`
- `POST /sessions/stop`

When `BACKEND_API_SECRET` is configured, session routes require the
`X-Agora-MatchCast-Backend-Secret` header.

## Tests

```bash
.venv/bin/pytest -q tests
```

The test suite uses synthetic audio and mocked Gemini or Agora boundaries, so
it does not make billable transcription calls.
