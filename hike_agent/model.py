"""Build the chat model named by ``MODEL_PROVIDER`` and ``MODEL_NAME``.

The default is the existing local setup: Ollama model ``gpt-oss:20b``.
Set ``MODEL_PROVIDER=openai`` to use the OpenAI API, or point
``OPENAI_BASE_URL`` at another OpenAI-compatible server.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from hike_agent.agents.planner import PLANNER_SYSTEM
from hike_agent.config import Settings


def build_chat_model(settings: Settings) -> Any:
    """Return a LangChain chat model. Raises if the provider is unknown or missing a key."""

    provider = settings.model_provider
    if provider == "ollama":
        import httpx
        from langchain_ollama import ChatOllama

        # Connect quickly when Ollama is not running; allow a local model time to answer.
        timeout = httpx.Timeout(connect=3.0, read=90.0, write=10.0, pool=5.0)
        return ChatOllama(
            model=settings.model_name,
            base_url=settings.ollama_base_url,
            temperature=0,
            sync_client_kwargs={"timeout": timeout},
        )
    if provider in {"openai", "openai-compatible"}:
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
    raise RuntimeError(
        f"Unsupported MODEL_PROVIDER {settings.model_provider!r}. Use 'ollama' or 'openai'."
    )


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
    """Return a function that asks the configured model to rewrite a factual brief."""

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
