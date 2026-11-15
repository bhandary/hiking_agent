"""Open-Meteo forecast summary and National Weather Service alerts.

The daylight summary matches the original script: local hours 08:00–17:00,
average temperature, the most common weather code, and the highest chance of
precipitation. NWS alerts are added so the planner can refuse an unsafe day.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime
from typing import Any, Callable

WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

UNSAFE_CODES = {65, 67, 75, 82, 86, 95, 96, 99}
CAUTION_CODES = {45, 48, 55, 56, 57, 61, 63, 66, 71, 73, 80, 81, 85}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"
GetJson = Callable[..., Any]


def _codes(hourly: dict[str, Any]) -> list[Any]:
    if "weather_code" in hourly:
        return hourly["weather_code"]
    return hourly.get("weathercode") or []


def summarize_daylight(weather_data: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    """Turn an Open-Meteo hourly forecast into one daylight summary.

    Returns a dict with ``summary`` text and the numbers the risk check uses.
    ``summary`` is an error string when the payload cannot be summarized; the
    ``ok`` flag distinguishes those cases.
    """

    if not weather_data or "hourly" not in weather_data:
        return {"ok": False, "summary": "Weather data is incomplete."}
    hourly = weather_data["hourly"]
    required = ("time", "temperature_2m", "precipitation_probability")
    for key in required:
        if key not in hourly or not hourly[key]:
            return {"ok": False, "summary": f"Missing or empty weather data for '{key}'."}
    if not _codes(hourly):
        return {"ok": False, "summary": "Missing or empty weather data for 'weather_code'."}

    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    precip_probs = hourly["precipitation_probability"]
    weather_codes = _codes(hourly)
    today = today or date.today()

    daylight_temps: list[float] = []
    daylight_precip: list[float] = []
    daylight_codes: list[int] = []
    for index, stamp in enumerate(times):
        try:
            moment = datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            continue
        if moment.date() != today or not 8 <= moment.hour <= 17:
            continue
        try:
            temperature = temperatures[index]
            code = weather_codes[index]
        except IndexError:
            continue
        if temperature is None or code is None:
            continue
        daylight_temps.append(float(temperature))
        precip = precip_probs[index] if index < len(precip_probs) else None
        if precip is not None:
            daylight_precip.append(float(precip))
        daylight_codes.append(int(code))

    if not daylight_temps:
        if today != date.today():
            return {"ok": False, "summary": f"No daylight forecast was available for {today.isoformat()}."}
        return {"ok": False, "summary": "Could not get a weather summary for today's daylight hours."}
    try:
        average_temp = round(statistics.mean(daylight_temps))
        common_code = statistics.mode(daylight_codes)
    except statistics.StatisticsError:
        return {"ok": False, "summary": "Could not calculate weather statistics due to insufficient data."}

    description = WMO_CODES.get(common_code, "unknown weather")
    if daylight_precip:
        max_precip = max(daylight_precip)
        precip_text = f"a maximum precipitation probability of {round(max_precip)}%"
    else:
        max_precip = None
        precip_text = "no precipitation probability in the forecast"
    if today == date.today():
        prefix = "Today's forecast"
    else:
        prefix = f"Forecast for {today.isoformat()}"
    summary = (
        f"{prefix}: {description}, with an average temperature of "
        f"{average_temp}°C and {precip_text}."
    )
    return {
        "ok": True,
        "summary": summary,
        "average_temperature_c": average_temp,
        "max_precipitation_probability": max_precip,
        "weather_code": common_code,
        "weather_description": description,
    }


def assess_risk(
    *,
    weather_code: int | None,
    average_temperature_c: float | None,
    max_precipitation_probability: float | None,
    alerts: list[dict[str, str]],
) -> tuple[str, list[str]]:
    """Return ``ok``, ``caution``, or ``unsafe`` plus the notes that justify it."""

    notes: list[str] = []
    risk = "ok"
    for alert in alerts:
        severity = (alert.get("severity") or "").lower()
        event = alert.get("event") or alert.get("headline") or "Weather alert"
        headline = alert.get("headline") or event
        if severity in {"extreme", "severe"}:
            risk = "unsafe"
        elif severity in {"moderate", "minor"} and risk != "unsafe":
            risk = "caution"
        notes.append(f"{severity or 'alert'}: {headline}".strip())
    if weather_code in UNSAFE_CODES:
        risk = "unsafe"
        notes.append(f"Daylight conditions include {WMO_CODES.get(weather_code, 'hazardous weather')}.")
    elif weather_code in CAUTION_CODES and risk != "unsafe":
        risk = "caution"
        notes.append(f"Daylight conditions include {WMO_CODES.get(weather_code, 'unsettled weather')}.")
    if max_precipitation_probability is not None:
        if max_precipitation_probability >= 70:
            risk = "unsafe"
            notes.append(f"Precipitation probability reaches {round(max_precipitation_probability)}%.")
        elif max_precipitation_probability >= 40 and risk != "unsafe":
            risk = "caution"
            notes.append(f"Precipitation probability reaches {round(max_precipitation_probability)}%.")
    if average_temperature_c is not None:
        if average_temperature_c <= -5 or average_temperature_c >= 35:
            risk = "unsafe"
            notes.append(f"Average daylight temperature is {round(average_temperature_c)}°C.")
        elif (average_temperature_c <= 0 or average_temperature_c >= 32) and risk != "unsafe":
            risk = "caution"
            notes.append(f"Average daylight temperature is {round(average_temperature_c)}°C.")
    if not notes:
        notes.append("No active weather alerts were returned for this point. That is not a substitute for a local check before you leave.")
    # De-duplicate while keeping order.
    unique: list[str] = []
    for note in notes:
        if note not in unique:
            unique.append(note)
    return risk, unique


def nws_alerts_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    for feature in payload.get("features") or []:
        props = feature.get("properties") or {}
        alerts.append(
            {
                "event": props.get("event") or "",
                "severity": props.get("severity") or "",
                "headline": props.get("headline") or props.get("event") or "",
            }
        )
    return alerts


def fetch_open_meteo_and_alerts(
    latitude: float,
    longitude: float,
    *,
    get_json: GetJson,
    timeout: float,
    user_agent: str,
    today: date | None = None,
    forecast_days: int = 2,
) -> dict[str, Any]:
    """Direct-API weather report used when the weather MCP server is down."""

    forecast = get_json(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "forecast_days": forecast_days,
            "timezone": "auto",
        },
        timeout=timeout,
        user_agent=user_agent,
    )
    summary = summarize_daylight(forecast, today=today)
    alert_error = None
    alerts: list[dict[str, str]] = []
    try:
        alert_payload = get_json(
            NWS_ALERTS_URL,
            params={"point": f"{latitude},{longitude}"},
            timeout=timeout,
            user_agent=user_agent,
            headers={"Accept": "application/geo+json"},
        )
        alerts = nws_alerts_from_payload(alert_payload)
    except Exception as exc:  # noqa: BLE001 - a forecast without alerts is still useful
        alert_error = f"NWS alerts were unavailable ({type(exc).__name__})."
    if not summary.get("ok"):
        notes = [summary["summary"]]
        if alert_error:
            notes.append(alert_error)
        return {
            "source": "open-meteo+nws",
            "summary": summary["summary"],
            "suitable": False,
            "risk": "caution",
            "safety_notes": notes,
            "alerts": alerts,
        }
    risk, notes = assess_risk(
        weather_code=summary.get("weather_code"),
        average_temperature_c=summary.get("average_temperature_c"),
        max_precipitation_probability=summary.get("max_precipitation_probability"),
        alerts=alerts,
    )
    if alert_error:
        notes.append(alert_error)
        if risk == "ok":
            risk = "caution"
    return {
        "source": "open-meteo+nws",
        "summary": summary["summary"],
        "suitable": risk == "ok",
        "risk": risk,
        "safety_notes": notes,
        "alerts": alerts,
        "average_temperature_c": summary.get("average_temperature_c"),
        "max_precipitation_probability": summary.get("max_precipitation_probability"),
        "weather_description": summary.get("weather_description"),
    }
