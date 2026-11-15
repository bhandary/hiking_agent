"""Shared fixtures for agent tool tests."""

from __future__ import annotations

import warnings

import pytest

from hike_agent.config import Settings

warnings.filterwarnings("ignore", message=".*langchain.mcp.*")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        model_name="gpt-4.1-mini",
        openai_api_key="",
        openai_base_url="",
        openai_timeout=30,
        openai_max_retries=2,
        nps_api_key="",
        nps_mcp_url="https://nps.example.test/mcp",
        nps_mcp_enabled=True,
        nps_mcp_command="npx",
        nps_mcp_args=("-y", "@cyanheads/national-parks-mcp-server@0.3.0"),
        weather_mcp_enabled=True,
        weather_mcp_command="npx",
        weather_mcp_args=("-y", "@dangahagan/weather-mcp@1.35.17"),
        nps_mcp_timeout=1,
        weather_mcp_timeout=1,
        http_timeout=5,
        user_agent="hike-agent-test",
    )
