"""Runtime configuration loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    """Everything the agents need, with no secrets stored in code."""

    model_name: str
    openai_api_key: str
    openai_base_url: str
    openai_timeout: float
    openai_max_retries: int
    nps_api_key: str
    nps_mcp_url: str
    nps_mcp_enabled: bool
    nps_mcp_command: str
    nps_mcp_args: tuple[str, ...]
    weather_mcp_enabled: bool
    weather_mcp_command: str
    weather_mcp_args: tuple[str, ...]
    nps_mcp_timeout: float
    weather_mcp_timeout: float
    http_timeout: float
    user_agent: str


def get_settings() -> Settings:
    """Read settings from the current environment.

    Called at use time so tests can change variables without reimporting.
    """

    weather_args = os.environ.get(
        "WEATHER_MCP_ARGS",
        "-y,@dangahagan/weather-mcp@1.35.17",
    )
    nps_args = os.environ.get(
        "NPS_MCP_ARGS",
        "-y,@cyanheads/national-parks-mcp-server@0.3.0",
    )
    return Settings(
        model_name=os.environ.get("MODEL_NAME", "").strip() or "gpt-4.1-mini",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "").strip(),
        openai_timeout=_float("OPENAI_TIMEOUT_SECONDS", 30),
        openai_max_retries=_int("OPENAI_MAX_RETRIES", 2),
        nps_api_key=os.environ.get("NPS_API_KEY", "").strip(),
        nps_mcp_url=os.environ.get(
            "NPS_MCP_URL",
            "https://national-parks.caseyjhand.com/mcp",
        ).strip(),
        nps_mcp_enabled=_flag("NPS_MCP_ENABLED", True),
        nps_mcp_command=os.environ.get("NPS_MCP_COMMAND", "npx").strip(),
        nps_mcp_args=tuple(part for part in nps_args.split(",") if part),
        weather_mcp_enabled=_flag("WEATHER_MCP_ENABLED", True),
        weather_mcp_command=os.environ.get("WEATHER_MCP_COMMAND", "npx").strip(),
        weather_mcp_args=tuple(part for part in weather_args.split(",") if part),
        nps_mcp_timeout=_float("NPS_MCP_TIMEOUT_SECONDS", 12),
        weather_mcp_timeout=_float("WEATHER_MCP_TIMEOUT_SECONDS", 25),
        http_timeout=_float("HTTP_TIMEOUT_SECONDS", 20),
        user_agent=os.environ.get(
            "HIKE_USER_AGENT",
            "hike-agent/0.1 (educational hike planner)",
        ).strip(),
    )
