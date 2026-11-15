# Vendored from agent-chat-ui

This directory is a copy of [langchain-ai/agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) at commit `0d21f7ff4c45287797e876dc6d8b150b6f277b1e` (2026-11-03), not a git submodule.

The upstream project is MIT licensed. Copyright (c) 2025 Brace Sproul. The full license is in `LICENSE`.

Local changes on top of that snapshot:

- Default graph id is `hike_planner` and the default API URL is `http://localhost:2024`, so the setup form is skipped for this hike planner.
- `next.config.mjs` sets `output: "standalone"` for the Railway image.
- `GET /health` is added for the Railway healthcheck.
- `Dockerfile` and `railway.toml` build and start the Next.js app on `$PORT`.

A submodule would pin the same commit, but it would also make `pnpm install` and the Docker build depend on a second clone. A directory in this repo is the copy Railway builds.
