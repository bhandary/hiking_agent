"""Timezone default, bad retry settings, and chat queries that name no place."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from hike_agent.chat_graph import build_chat_graph
from hike_agent.config import get_settings
from hike_agent.extract import extract_hike_request


def test_default_today_uses_los_angeles(monkeypatch):
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            assert getattr(tz, "key", None) == "America/Los_Angeles"
            return datetime(2026, 11, 15, 12, tzinfo=tz)

    monkeypatch.setattr("hike_agent.clock.datetime", Fixed)
    found = extract_hike_request("hike tomorrow")
    assert found["place"] == ""
    assert found["when"]["start"] == "2026-11-16"
    assert found["timezone"] == "America/Los_Angeles"


def test_a_named_timezone_can_be_a_different_day(monkeypatch):
    instant = datetime(2026, 11, 16, 6, 30, tzinfo=ZoneInfo("UTC"))

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz)

    monkeypatch.setattr("hike_agent.clock.datetime", Fixed)
    los_angeles = extract_hike_request("hike today", timezone="America/Los_Angeles")
    new_york = extract_hike_request("hike today", timezone="America/New_York")
    assert los_angeles["when"]["start"] == "2026-11-15"
    assert new_york["when"]["start"] == "2026-11-16"


def test_bad_openai_max_retries_falls_back_once(monkeypatch, caplog):
    from hike_agent import config

    config._INT_WARNED.clear()
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "two")
    with caplog.at_level("WARNING"):
        assert get_settings().openai_max_retries == 2
        assert get_settings().openai_max_retries == 2
    assert caplog.text.count("OPENAI_MAX_RETRIES") == 1


def test_chat_does_not_geocode_the_raw_hike_sentence():
    seen = {}

    def plan(query, **_kwargs):
        seen["query"] = query
        return {"recommendation": "need a place"}

    graph = build_chat_graph(plan)
    graph.invoke({"messages": [{"role": "user", "content": "hike tomorrow"}]})
    assert seen["query"] == ""
