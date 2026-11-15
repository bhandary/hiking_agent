"""Weather and conditions agent.

Prefers the keyless ``@dangahagan/weather-mcp`` server (NOAA for the US,
Open-Meteo elsewhere, plus alerts). Falls back to the original Open-Meteo
forecast plus the National Weather Service alerts endpoint.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable

from langchain_core.tools import tool

from hike_agent.config import Settings, get_settings
from hike_agent.mcp_bridge import call_mcp_tools, weather_mcp_target
from hike_agent.services.http import get_json as default_get_json
from hike_agent.services.weather import fetch_open_meteo_and_alerts

AGENT_NAME = "weather"
AGENT_BRIEF = "Summarize daylight hiking conditions and active weather alerts."


def _forecast_horizon(today: date | None, *, anchor: date | None = None) -> int | None:
    """How many forecast days to request so ``today`` is inside the window.

    Returns ``None`` when ``today`` is 16 or more days after ``anchor``. The
    forecast APIs do not cover that day, so the caller says so instead of
    requesting a window that ends earlier.
    """

    if today is None:
        return 2
    base = anchor if anchor is not None else date.today()
    delta = (today - base).days
    if delta < 0:
        return 2
    if delta >= 16:
        return None
    if delta <= 1:
        return 2
    # Open-Meteo and the weather MCP accept 16 days. "next Monday" can be 13
    # days out; a cap of 7 asked for a window that did not contain that day.
    return min(delta + 1, 16)


def _outside_forecast(target: date, anchor: date) -> dict[str, Any]:
    end = anchor + timedelta(days=15)
    return {
        "source": "none",
        "summary": (
            f"No forecast is available for {target.isoformat()}. "
            f"Forecasts cover 16 days from {anchor.isoformat()} through {end.isoformat()}. "
            f"{target.isoformat()} is outside that window, so this is not a weather forecast."
        ),
        "suitable": False,
        "risk": "unknown",
        "safety_notes": ["No weather was looked up for this date."],
        "alerts": [],
    }


def _clip_forecast(text: str, day: date | None, limit: int = 2500) -> str:
    """Keep the requested day's section, then clip the rest.

    A blind cut at 2500 characters can drop the day the hiker asked about
    when it sits later in a multi-day report.
    """

    if len(text) <= limit:
        return text
    start = _day_section_start(text, day)
    excerpt = text[start:]
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 1].rstrip() + "…"


def _day_section_start(text: str, day: date | None) -> int:
    if day is None:
        return 0
    lowered = text.lower()
    markers = [
        day.isoformat(),
        f"{day.strftime('%B')} {day.day}".lower(),
        f"{day.strftime('%b')} {day.day}".lower(),
        day.strftime("%A").lower(),
    ]
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx == -1:
            continue
        line = text.rfind("\n", 0, idx)
        return 0 if line < 0 else line + 1
    return 0

McpCall = Callable[[Any, str, dict[str, Any], float], dict[str, Any]]

_UNSAFE_PHRASES = (
    "extreme",
    "tornado",
    "hurricane",
    "blizzard",
    "thunderstorm warning",
    "flash flood",
    "avalanche",
)
_CAUTION_PHRASES = ("warning", "watch", "advisory", "severe", "high wind")


_RISK_RANK = {"ok": 0, "caution": 1, "unsafe": 2}
_FORECAST_UNSAFE = ("thunder", "tornado", "blizzard", "hurricane", "heavy rain", "heavy snow", "ice storm")
_FORECAST_CAUTION = ("fog", "rain", "snow", "shower", "ice", "storm", "high wind")


def _worse(left: str, right: str) -> str:
    return left if _RISK_RANK[left] >= _RISK_RANK[right] else right


def _risk_from_alert_text(text: str) -> str:
    lowered = text.lower()
    if not lowered or "no active" in lowered or "no alerts" in lowered or "clear of weather warnings" in lowered:
        return "ok"
    if any(phrase in lowered for phrase in _UNSAFE_PHRASES):
        return "unsafe"
    if any(phrase in lowered for phrase in _CAUTION_PHRASES):
        return "caution"
    return "ok"


def _risk_from_forecast_text(text: str) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _FORECAST_UNSAFE):
        return "unsafe"
    if any(phrase in lowered for phrase in _FORECAST_CAUTION):
        return "caution"
    return "ok"


def _alert_note(alert_text: str, alert_ok: bool) -> str:
    if not alert_ok:
        return "Weather alerts could not be loaded from the MCP server. Do not treat the forecast as an all-clear."
    lowered = alert_text.lower()
    if not alert_text or "no active" in lowered or "no alerts" in lowered or "clear of weather warnings" in lowered:
        return "NOAA reports no active weather alerts for this point. That is not an all-clear for trail conditions."
    clipped = alert_text if len(alert_text) <= 500 else alert_text[:499].rstrip() + "…"
    return clipped


def conditions_from_mcp(
    forecast: dict[str, Any],
    alerts: dict[str, Any] | None,
    *,
    day: date | None = None,
) -> dict[str, Any] | None:
    """Build a conditions report from weather-MCP tool payloads."""

    if not forecast or not forecast.get("ok") or not (forecast.get("text") or "").strip():
        return None
    full = (forecast.get("text") or "").strip()
    summary = _clip_forecast(full, day)
    alert_text = ((alerts or {}).get("text") or "").strip()
    alert_ok = bool(alerts and alerts.get("ok"))
    risk = _worse(
        _risk_from_forecast_text(full),
        _risk_from_alert_text(alert_text) if alert_ok else "caution",
    )
    notes = [_alert_note(alert_text, alert_ok)]
    if _risk_from_forecast_text(full) != "ok":
        notes.append("The forecast mentions unsettled weather (fog, precipitation, storms, or ice).")
    return {
        "source": "weather-mcp",
        "summary": summary,
        "suitable": risk == "ok",
        "risk": risk,
        "safety_notes": notes,
        "alerts": [],
    }


def hiking_conditions(
    place: dict[str, Any],
    *,
    settings: Settings | None = None,
    mcp_call: McpCall | None = None,
    get_json: Callable[..., Any] | None = None,
    today: date | None = None,
    anchor: date | None = None,
) -> dict[str, Any]:
    """Return a conditions report for a resolved place."""

    settings = settings or get_settings()
    get_json = get_json or default_get_json
    latitude = place.get("latitude")
    longitude = place.get("longitude")
    if latitude is None or longitude is None:
        return {
            "source": "none",
            "summary": "Weather was not checked because the place has no coordinates.",
            "suitable": False,
            "risk": "caution",
            "safety_notes": ["Location failed, so conditions are unknown."],
            "alerts": [],
        }

    horizon = _forecast_horizon(today, anchor=anchor)
    if today is not None and horizon is None:
        return _outside_forecast(today, anchor if anchor is not None else date.today())
    days = horizon if horizon is not None else 2

    if settings.weather_mcp_enabled:
        target = weather_mcp_target(settings)
        forecast_arguments = {
            "latitude": latitude,
            "longitude": longitude,
            "days": days,
            "granularity": "daily",
            "include_precipitation_probability": True,
            "source": "auto",
        }
        alert_arguments = {"latitude": latitude, "longitude": longitude, "active_only": True}
        if mcp_call is None:
            forecast, alerts = call_mcp_tools(
                target,
                [("get_forecast", forecast_arguments), ("get_alerts", alert_arguments)],
                settings.weather_mcp_timeout,
            )
        else:
            forecast = mcp_call(target, "get_forecast", forecast_arguments, settings.weather_mcp_timeout)
            alerts = mcp_call(target, "get_alerts", alert_arguments, settings.weather_mcp_timeout)
        report = conditions_from_mcp(forecast, alerts, day=today)
        if report:
            return report

    try:
        return fetch_open_meteo_and_alerts(
            float(latitude),
            float(longitude),
            get_json=get_json,
            timeout=settings.http_timeout,
            user_agent=settings.user_agent,
            today=today,
            user_today=anchor,
            forecast_days=days,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "none",
            "summary": f"Weather lookup failed ({type(exc).__name__}).",
            "suitable": False,
            "risk": "caution",
            "safety_notes": ["Weather data is missing. Do not assume it is safe to hike."],
            "alerts": [],
        }


@tool
def hiking_conditions_tool(place_json: str) -> str:
    """Summarize hiking conditions for a place JSON object with latitude and longitude."""

    import json

    return json.dumps(hiking_conditions(json.loads(place_json)))
