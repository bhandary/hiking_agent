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


class ModelUnavailable(RuntimeError):
    """The OpenAI request failed after the allowed attempts."""


def build_chat_model(settings: Settings) -> Any:
    """Return a ChatOpenAI client. Raises when ``OPENAI_API_KEY`` is missing.

    Each attempt uses ``OPENAI_TIMEOUT_SECONDS``. Retries are counted in
    ``make_complete`` so a failure becomes one error instead of a stack trace.
    The client itself does not retry again.
    """

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "temperature": 0,
        "api_key": settings.openai_api_key,
        "timeout": settings.openai_timeout,
        "max_retries": 0,
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


def make_complete(settings: Settings, call=None):
    """Return a function that asks the OpenAI model to rewrite a factual brief.

    ``call`` replaces the network request in tests. Attempts are
    ``OPENAI_MAX_RETRIES`` plus the first try. The last failure is raised as
    ``ModelUnavailable``.
    """

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    attempts = max(1, int(settings.openai_max_retries) + 1)

    def complete(brief: str) -> str:
        last: BaseException | None = None
        for _attempt in range(attempts):
            try:
                if call is not None:
                    text = call(brief)
                else:
                    model = build_chat_model(settings)
                    message = model.invoke(
                        [
                            SystemMessage(content=PLANNER_SYSTEM),
                            HumanMessage(content=brief),
                        ]
                    )
                    text = _message_text(message.content)
                if not str(text or "").strip():
                    raise RuntimeError("the model returned an empty recommendation")
                return str(text).strip()
            except Exception as exc:  # noqa: BLE001 - counted toward the retry limit
                last = exc
        detail = f"{type(last).__name__}: {last}" if last else "unknown error"
        raise ModelUnavailable(
            f"The hike model did not respond after {attempts} attempts ({detail})."
        ) from last

    return complete
