"""Conversational graph served to agent-chat-ui.

The LangGraph server streams the ``messages`` channel. This graph reads the
latest human message, extracts a place and a date, runs the fixed hike graph,
and returns the planner text as one assistant message.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from hike_agent.cli import plan_hike
from hike_agent.extract import extract_hike_request


class ChatState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    coordinates: dict[str, Any]


def latest_human_text(messages: list[Any] | None) -> str:
    """Text of the newest human message, or the newest message of any role."""

    chosen = ""
    for message in reversed(messages or []):
        role, text = _role_and_text(message)
        if not text.strip():
            continue
        if role in {"human", "user"}:
            return text
        if not chosen:
            chosen = text
    return chosen


def build_chat_graph(plan=None):
    """Compile the chat graph. Tests can pass a fake ``plan`` callable."""

    planner = plan or plan_hike

    def plan_node(state: ChatState) -> dict[str, Any]:
        text = latest_human_text(state.get("messages"))
        extracted = extract_hike_request(text)
        place = extracted["place"] or text
        coordinates = state.get("coordinates") or extracted.get("coordinates")
        result = planner(place, when=extracted.get("when") or {}, coordinates=coordinates)
        answer = (result or {}).get("recommendation") or "No recommendation was produced."
        return {"messages": [AIMessage(content=answer)]}

    graph = StateGraph(ChatState)
    graph.add_node("plan", plan_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", END)
    return graph.compile()


def _role_and_text(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("type") or message.get("role") or "")
        content = message.get("content", "")
    else:
        role = str(getattr(message, "type", "") or "")
        content = getattr(message, "content", "")
    return role, _content_text(content)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {None, "text"}:
                parts.append(str(block.get("text") or ""))
        return " ".join(part for part in parts if part)
    return str(content or "")


graph = build_chat_graph()
