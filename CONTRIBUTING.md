# Contributing to Agora MatchCast

Keep the project focused on one pipeline: live sports/esports commentary in,
low-latency captions out.

## Development setup

Follow the root README to configure local frontend and backend environments.
Never use production credentials in a development checkout.

## Before opening a pull request

```bash
pnpm install
pnpm run verify

server/.venv/bin/pytest -q server/tests
```

Keep pull requests focused and explain user-visible behavior changes. Add or
update tests for caption reconciliation, session lifecycle, provider routing,
or UI contracts when those areas change.

## Safety

- Never commit API keys, Agora certificates, backend secrets, Media Gateway
  credentials, stream keys, local `.env` files, or PCM captures.
- Do not include copyrighted broadcast footage or generated exports.
- Use synthetic PCM and mocked provider responses in unit tests.
- Do not make live provider calls from the default CI suite.

By submitting a contribution, you agree that it may be distributed under the
project's MIT License.
