"""OpenAI timeout and the limited retry around a planner call."""

from __future__ import annotations

import pytest

from hike_agent.config import Settings
from hike_agent.model import ModelUnavailable, build_chat_model, make_complete


def _settings(**overrides) -> Settings:
    values = dict(
        model_name="gpt-4.1-mini",
        openai_api_key="sk-test",
        openai_base_url="",
        openai_timeout=12,
        openai_max_retries=2,
        nps_api_key="",
        nps_mcp_url="",
        nps_mcp_enabled=False,
        nps_mcp_command="npx",
        nps_mcp_args=(),
        weather_mcp_enabled=False,
        weather_mcp_command="npx",
        weather_mcp_args=(),
        nps_mcp_timeout=1,
        weather_mcp_timeout=1,
        http_timeout=5,
        user_agent="hike-agent-test",
    )
    values.update(overrides)
    return Settings(**values)


def test_chat_model_sets_a_request_timeout(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChat)
    build_chat_model(_settings())
    assert captured["timeout"] == 12
    assert captured["max_retries"] == 0
    assert captured["model"] == "gpt-4.1-mini"


def test_complete_retries_then_returns_the_model_text():
    calls = {"n": 0}

    def call(_brief: str) -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("slow")
        return "Hike the looked-up trail."

    complete = make_complete(_settings(openai_max_retries=2), call=call)
    assert complete("brief") == "Hike the looked-up trail."
    assert calls["n"] == 2


def test_complete_stops_after_the_retry_limit():
    calls = {"n": 0}

    def call(_brief: str) -> str:
        calls["n"] += 1
        raise TimeoutError("slow")

    complete = make_complete(_settings(openai_max_retries=2), call=call)
    with pytest.raises(ModelUnavailable) as raised:
        complete("brief")
    assert calls["n"] == 3
    assert "did not respond after 3 attempts" in str(raised.value)
    assert "TimeoutError" in str(raised.value)
