#!/usr/bin/env bash
# Start the in-memory LangGraph server and the chat UI together.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi
if [[ ! -f ui/.env ]]; then
  cp ui/.env.example ui/.env
  echo "Created ui/.env from ui/.env.example"
fi

langgraph dev --host 127.0.0.1 --port 2024 --no-browser --allow-blocking &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "LangGraph API: http://127.0.0.1:2024  (graph id hike_planner)"
echo "Chat UI:       http://127.0.0.1:3000"
(cd ui && pnpm dev --port 3000)
