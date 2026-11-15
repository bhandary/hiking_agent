"""Weather and conditions agent.

Prefers the keyless ``@dangahagan/weather-mcp`` server (NOAA for the US,
Open-Meteo elsewhere, plus alerts). Falls back to the original Open-Meteo
forecast plus the National Weather Service alerts endpoint.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
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


_ENGLISH_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_ENGLISH_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_ENGLISH_MONTHS_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_RELATIVE_HEADINGS = ("this afternoon", "this morning", "this evening", "tonight", "today", "overnight")
_WEEKDAY_HEADING = re.compile(
    r"^(?P<name>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def _weekday_name(day: date) -> str:
    return _ENGLISH_WEEKDAYS[day.weekday()]


def _day_label(day: date) -> str:
    return f"Forecast for {day.isoformat()} ({_weekday_name(day)})"


def _limit_text(text: str, limit: int = 2500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _line_heading(line: str) -> str | None:
    body = re.sub(r"^#+\s*", "", line).strip()
    body = re.sub(r"\*+", "", body).strip().rstrip(":").strip()
    if not body:
        return None
    lower = body.lower()
    iso = re.match(r"(\d{4}-\d{2}-\d{2})\b", lower)
    if iso:
        return f"date:{iso.group(1)}"
    weekday = _WEEKDAY_HEADING.match(lower)
    if weekday:
        return f"weekday:{weekday.group('name').lower()}"
    for relative in _RELATIVE_HEADINGS:
        if lower == relative or lower.startswith(relative + " ") or lower.startswith(relative + ":"):
            return "relative"
    month = re.match(
        r"(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}\b",
        lower,
    )
    if month:
        return f"month:{month.group(0).lower()}"
    return None


def _heading_matches(kind: str, day: date) -> bool:
    if kind == f"weekday:{_weekday_name(day).lower()}":
        return True
    if kind == f"date:{day.isoformat()}":
        return True
    if kind.startswith("month:"):
        spoken = kind.removeprefix("month:")
        long_name = f"{_ENGLISH_MONTHS[day.month - 1]} {day.day}".lower()
        short_name = f"{_ENGLISH_MONTHS_ABBR[day.month - 1]} {day.day}".lower()
        return spoken.startswith(long_name) or spoken.startswith(short_name)
    return False


def _isolate_day_text(text: str, day: date) -> str:
    """From the first heading for ``day`` through the next different day.

    ``Saturday`` and ``Saturday Night`` both stay. ``Sunday`` and a later
    ``This Afternoon`` do not. The first heading wins, not the last.
    """

    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        kind = _line_heading(line)
        if kind and kind != "relative" and _heading_matches(kind, day):
            start = index
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start + 1, len(lines)):
        kind = _line_heading(lines[index])
        if kind and kind != "relative" and not _heading_matches(kind, day):
            end = index
            break
    excerpt = "\n".join(lines[start:end]).strip()
    if not excerpt:
        return ""
    label = _day_label(day)
    if excerpt.lower().startswith(label.lower()):
        return excerpt
    return f"{label}\n{excerpt}"


def _periods_of(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    structured = forecast.get("structured")
    candidates: list[Any] = []
    if isinstance(structured, list):
        candidates = structured
    elif isinstance(structured, dict):
        if isinstance(structured.get("periods"), list):
            candidates = structured["periods"]
        else:
            properties = structured.get("properties")
            if isinstance(properties, dict) and isinstance(properties.get("periods"), list):
                candidates = properties["periods"]
    if not candidates:
        properties = forecast.get("properties")
        if isinstance(properties, dict) and isinstance(properties.get("periods"), list):
            candidates = properties["periods"]
    return [period for period in candidates if isinstance(period, dict)]


def _period_date(period: dict[str, Any]) -> date | None:
    raw = period.get("startTime") or period.get("start")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _is_daytime(period: dict[str, Any]) -> bool:
    if "isDaytime" in period:
        return bool(period.get("isDaytime"))
    name = str(period.get("name") or "").lower()
    return not name.endswith("night") and "tonight" not in name


def _render_periods(day: date, periods: list[dict[str, Any]]) -> str:
    ordered = sorted(periods, key=lambda period: (0 if _is_daytime(period) else 1, str(period.get("startTime") or "")))
    lines = [_day_label(day)]
    for period in ordered:
        name = str(period.get("name") or ("Day" if _is_daytime(period) else "Night"))
        temperature = period.get("temperature")
        unit = period.get("temperatureUnit") or "F"
        forecast = str(period.get("shortForecast") or period.get("detailedForecast") or "").strip()
        parts: list[str] = []
        if temperature is not None and temperature != "":
            parts.append(f"{temperature} {unit}")
        if forecast:
            parts.append(forecast)
        lines.append(f"{name}: {', '.join(parts)}".rstrip(": "))
    return "\n".join(lines)


def _forecast_for_day(forecast: dict[str, Any], day: date | None) -> str:
    """The requested day's periods only, from structured data when it exists."""

    full = (forecast.get("text") or "").strip()
    if day is None:
        return _limit_text(full)
    periods = [period for period in _periods_of(forecast) if _period_date(period) == day]
    if periods:
        return _limit_text(_render_periods(day, periods))
    if _periods_of(forecast):
        return _limit_text(f"{_day_label(day)}\nNo period was published for this date.")
    isolated = _isolate_day_text(full, day)
    if isolated:
        return _limit_text(isolated)
    return _limit_text(f"{_day_label(day)}\nNo period was published for this date.")


def fetch_nws_periods(
    latitude: float,
    longitude: float,
    *,
    get_json: Callable[..., Any],
    timeout: float,
    user_agent: str,
) -> list[dict[str, Any]]:
    """National Weather Service forecast periods for one point. No API key."""

    points_url = f"https://api.weather.gov/points/{float(latitude):.4f},{float(longitude):.4f}"
    points = get_json(points_url, timeout=timeout, user_agent=user_agent, headers={"User-Agent": user_agent})
    forecast_url = ((points or {}).get("properties") or {}).get("forecast")
    if not forecast_url:
        return []
    forecast = get_json(forecast_url, timeout=timeout, user_agent=user_agent, headers={"User-Agent": user_agent})
    periods = ((forecast or {}).get("properties") or {}).get("periods") or []
    return [period for period in periods if isinstance(period, dict)]

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

    if not forecast or not forecast.get("ok") or not ((forecast.get("text") or "").strip() or _periods_of(forecast)):
        return None
    summary = _forecast_for_day(forecast, day)
    alert_text = ((alerts or {}).get("text") or "").strip()
    alert_ok = bool(alerts and alerts.get("ok"))
    forecast_risk = _risk_from_forecast_text(summary)
    risk = _worse(
        forecast_risk,
        _risk_from_alert_text(alert_text) if alert_ok else "caution",
    )
    notes = [_alert_note(alert_text, alert_ok)]
    if forecast_risk != "ok":
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
        if today is not None and forecast and forecast.get("ok") and not _periods_of(forecast):
            try:
                periods = fetch_nws_periods(
                    float(latitude),
                    float(longitude),
                    get_json=get_json,
                    timeout=settings.http_timeout,
                    user_agent=settings.user_agent,
                )
            except Exception:  # noqa: BLE001 - the MCP text is still a forecast
                periods = []
            if periods:
                forecast = {**forecast, "structured": {"periods": periods}}
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
