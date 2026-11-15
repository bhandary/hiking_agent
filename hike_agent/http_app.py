"""Routes merged into the LangGraph server.

``langgraph.json`` points ``http.app`` here. Railway uses ``GET /health``.
The rest of the Agent Server API (threads, runs, ``/info``) stays on the
in-memory server started by ``langgraph dev``.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Hike planner")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
