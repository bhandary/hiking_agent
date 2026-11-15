"""Tests for the weather agent's tools. Network and MCP calls are faked."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hike_agent.agents.weather import conditions_from_mcp, hiking_conditions
from hike_agent.config import Settings
from hike_agent.services.weather import summarize_daylight
from tests.helpers import mcp_down

TODAY = date(2026, 11, 14)
PLACE = {"latitude": 40.015, "longitude": -105.271, "label": "Boulder, Colorado"}


def _hourly(*, code: int = 2, precip: float = 10, temp: float = 10, outlier_hour: int | None = 7, day: date = TODAY):
    times, temps, precips, codes = [], [], [], []
    for hour in range(24):
        times.append(f"{day.isoformat()}T{hour:02d}:00")
        if hour == outlier_hour:
            temps.append(100)
            precips.append(0)
            codes.append(0)
        else:
            temps.append(temp)
            precips.append(precip)
            codes.append(code)
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "precipitation_probability": precips,
            "weather_code": codes,
        }
    }


def _client(forecast, alerts=None):
    def get_json(url, **_kwargs):
        if "open-meteo.com" in url:
            return forecast
        if "weather.gov" in url:
            return {"features": alerts or []}
        raise AssertionError(url)

    return get_json


def test_daylight_summary_ignores_hours_outside_8_to_17():
    summary = summarize_daylight(_hourly(code=2, precip=15, temp=10, outlier_hour=7), today=TODAY)
    assert summary["ok"] is True
    assert summary["average_temperature_c"] == 10
    assert summary["weather_description"] == "Partly cloudy"
    assert "15%" in summary["summary"]


def test_legacy_weathercode_field_is_accepted():
    payload = _hourly()
    payload["hourly"]["weathercode"] = payload["hourly"].pop("weather_code")
    summary = summarize_daylight(payload, today=TODAY)
    assert summary["ok"] is True


def test_thunderstorm_is_unsafe(settings):
    report = hiking_conditions(
        PLACE,
        settings=Settings(**{**settings.__dict__, "weather_mcp_enabled": False}),
        get_json=_client(_hourly(code=95, precip=10, temp=16)),
        today=TODAY,
    )
    assert report["source"] == "open-meteo+nws"
    assert report["risk"] == "unsafe"
    assert report["suitable"] is False
    assert any("Thunderstorm" in note for note in report["safety_notes"])


def test_severe_nws_alert_overrides_clear_skies(settings):
    alerts = [
        {
            "properties": {
                "event": "High Wind Warning",
                "severity": "Severe",
                "headline": "High Wind Warning until 8pm",
            }
        }
    ]
    report = hiking_conditions(
        PLACE,
        settings=Settings(**{**settings.__dict__, "weather_mcp_enabled": False}),
        get_json=_client(_hourly(code=1, precip=0, temp=18), alerts),
        today=TODAY,
    )
    assert report["risk"] == "unsafe"
    assert any("High Wind Warning" in note for note in report["safety_notes"])


def test_clear_day_is_suitable_and_still_says_to_check_locally(settings):
    report = hiking_conditions(
        PLACE,
        settings=Settings(**{**settings.__dict__, "weather_mcp_enabled": False}),
        get_json=_client(_hourly(code=1, precip=5, temp=18)),
        today=TODAY,
    )
    assert report["risk"] == "ok"
    assert report["suitable"] is True
    assert report["safety_notes"]


def test_mcp_forecast_is_preferred_over_direct_apis(settings):
    def mcp_call(_target, tool, arguments, _timeout):
        assert arguments["latitude"] == PLACE["latitude"]
        if tool == "get_forecast":
            return {"ok": True, "text": "Saturday: sunny, high 12C.", "structured": None, "error": None}
        if tool == "get_alerts":
            return {"ok": True, "text": "No active alerts.", "structured": None, "error": None}
        raise AssertionError(tool)

    def get_json(*_args, **_kwargs):
        raise AssertionError("direct API should not run when MCP succeeds")

    report = hiking_conditions(PLACE, settings=settings, mcp_call=mcp_call, get_json=get_json, today=TODAY)
    assert report["source"] == "weather-mcp"
    assert "sunny" in report["summary"]
    assert report["risk"] == "ok"


def test_mcp_failure_uses_open_meteo(settings):
    report = hiking_conditions(
        PLACE,
        settings=settings,
        mcp_call=mcp_down,
        get_json=_client(_hourly(code=3, precip=20, temp=9)),
        today=TODAY,
    )
    assert report["source"] == "open-meteo+nws"
    assert "Overcast" in report["summary"]


def test_forecast_fog_and_rain_are_caution_even_without_alerts():
    report = conditions_from_mcp(
        {"ok": True, "text": "Areas of fog, then a slight chance of rain showers."},
        {"ok": True, "text": "No active weather alerts for this location."},
    )
    assert report is not None
    assert report["risk"] == "caution"
    assert report["suitable"] is False
    assert "NOAA reports no active weather alerts" in report["safety_notes"][0]


def test_mcp_alert_text_can_mark_a_forecast_unsafe():
    report = conditions_from_mcp(
        {"ok": True, "text": "Cloudy."},
        {"ok": True, "text": "Tornado Warning for Boulder County."},
    )
    assert report is not None
    assert report["risk"] == "unsafe"


def test_a_future_day_is_labeled_with_its_date():
    target = date.today() + timedelta(days=6)
    summary = summarize_daylight(_hourly(day=target), today=target)
    assert summary["ok"] is True
    assert summary["summary"].startswith(f"Forecast for {target.isoformat()}")
    assert "Today's forecast" not in summary["summary"]


def test_today_keeps_the_today_label():
    target = date.today()
    summary = summarize_daylight(_hourly(day=target), today=target)
    assert summary["summary"].startswith("Today's forecast")


def test_a_date_more_than_a_week_out_is_still_requested(settings):
    seen = {}
    target = date.today() + timedelta(days=10)

    def get_json(url, **kwargs):
        if "open-meteo.com" in url:
            seen["days"] = kwargs["params"]["forecast_days"]
            return {"hourly": {}}
        return {"features": []}

    hiking_conditions(PLACE, settings=settings, mcp_call=mcp_down, get_json=get_json, today=target)
    assert seen["days"] == 11


def test_missing_coordinates_do_not_call_out():
    report = hiking_conditions({"latitude": None, "longitude": None}, mcp_call=mcp_down)
    assert report["source"] == "none"
    assert report["suitable"] is False


def test_today_label_follows_the_hikers_day_not_a_different_clock():
    hikers_today = date(2026, 11, 15)
    later = date(2026, 11, 21)
    future = summarize_daylight(_hourly(day=later), today=later, user_today=hikers_today)
    assert future["summary"].startswith("Forecast for 2026-11-21")
    current = summarize_daylight(_hourly(day=hikers_today), today=hikers_today, user_today=hikers_today)
    assert current["summary"].startswith("Today's forecast")


def test_a_day_outside_the_16_day_window_is_not_fetched(settings):
    target = date.today() + timedelta(days=20)

    def fail(*_args, **_kwargs):
        raise AssertionError("a date outside the forecast window must not be fetched")

    report = hiking_conditions(PLACE, settings=settings, mcp_call=fail, get_json=fail, today=target)
    assert "outside that window" in report["summary"]
    assert "not a weather forecast" in report["summary"]
    assert report["source"] == "none"


def test_clipping_keeps_the_requested_day():
    day = date(2026, 11, 21)
    text = ("Earlier day: cloudy.\n" * 200) + f"\n{day.isoformat()}: sunny, high 12 C.\n"
    assert len(text) > 2500
    report = conditions_from_mcp(
        {"ok": True, "text": text},
        {"ok": True, "text": "No active alerts."},
        day=day,
    )
    assert report is not None
    assert day.isoformat() in report["summary"]
    assert len(report["summary"]) <= 2500
