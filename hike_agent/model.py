"""OpenAI chat model used by the planner.

``MODEL_NAME`` defaults to ``gpt-4.1-mini``. ``OPENAI_API_KEY`` is required.
``OPENAI_BASE_URL`` is optional and points at an OpenAI-compatible host.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from hike_agent.agents.planner import PLANNER_SYSTEM
from hike_agent.config import Settings

logger = logging.getLogger(__name__)


def warn_model_failure(exc: BaseException) -> str:
    """Log a warning and return the same text for the graph error list.

    The planner still returns the factual brief. This warning is what makes
    that fallback visible in the CLI and in the LangGraph server logs.
    """

    message = (
        f"Planner model call failed ({type(exc).__name__}: {exc}). "
        "Using the factual brief as the recommendation."
    )
    logger.warning(message)
    if not logger.handlers and not logging.getLogger().handlers:
        import sys

        print(f"warning: {message}", file=sys.stderr)
    return message


def build_chat_model(settings: Settings) -> Any:
    """Return a ChatOpenAI client. Raises when ``OPENAI_API_KEY`` is missing."""

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "temperature": 0,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def make_complete(settings: Settings):
    """Return a function that asks the OpenAI model to rewrite a factual brief."""

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    def complete(brief: str) -> str:
        model = build_chat_model(settings)
        message = model.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=brief),
            ]
        )
        return _message_text(message.content)

    return complete
