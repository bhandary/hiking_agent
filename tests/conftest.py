"""Shared fixtures for agent tool tests."""

from __future__ import annotations

import warnings

import pytest

from hike_agent.config import Settings

warnings.filterwarnings("ignore", message=".*langchain.mcp.*")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        model_provider="ollama",
        model_name="gpt-oss:20b",
        ollama_base_url="http://127.0.0.1:11434",
        openai_api_key="",
        openai_base_url="",
        nps_api_key="",
        nps_mcp_url="https://nps.example.test/mcp",
        nps_mcp_enabled=True,
        weather_mcp_enabled=True,
        weather_mcp_command="npx",
        weather_mcp_args=("-y", "@dangahagan/weather-mcp@1.35.17"),
        nps_mcp_timeout=1,
        weather_mcp_timeout=1,
        http_timeout=5,
        user_agent="hike-agent-test",
    )
