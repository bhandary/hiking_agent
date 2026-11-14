"""Tests for the location agent's place resolver."""

from __future__ import annotations

import pytest

from hike_agent.agents.location import resolve_place, resolve_place_tool
from hike_agent.config import get_settings
from hike_agent.services.states import state_code


def test_state_code_accepts_names_and_abbreviations():
    assert state_code("Colorado") == "CO"
    assert state_code("co") == "CO"
    assert state_code("Not a state") is None


def test_geocode_uses_the_first_hit_with_coordinates(settings):
    def get_json(url, **_kwargs):
        assert "geocoding-api.open-meteo.com" in url
        return {
            "results": [
                {"name": "Missing"},
                {
                    "name": "Boulder",
                    "latitude": 40.01499,
                    "longitude": -105.27055,
                    "admin1": "Colorado",
                    "country": "United States",
                },
            ]
        }

    place = resolve_place("Boulder, Colorado", settings=settings, get_json=get_json)
    assert place["source"] == "open-meteo"
    assert place["latitude"] == pytest.approx(40.01499)
    assert place["longitude"] == pytest.approx(-105.27055)
    assert place["state_code"] == "CO"
    assert place["label"] == "Boulder, Colorado, United States"


def test_unknown_place_has_no_coordinates(settings):
    place = resolve_place("Nowhereville", settings=settings, get_json=lambda *_a, **_k: {"results": []})
    assert place["latitude"] is None
    assert place["source"] == "none"
    assert "Nowhereville" in place["note"]


def test_geocoder_failure_is_reported(settings):
    def get_json(*_args, **_kwargs):
        raise TimeoutError("slow")

    place = resolve_place("Boulder", settings=settings, get_json=get_json)
    assert place["latitude"] is None
    assert "TimeoutError" in place["note"]


def test_me_uses_the_ip_lookup(settings):
    def ip_lookup():
        return {
            "query": "me",
            "name": "Boulder",
            "label": "Boulder, Colorado, United States",
            "latitude": 40.0,
            "longitude": -105.2,
            "state": "Colorado",
            "state_code": "CO",
            "country": "United States",
            "source": "ip",
            "note": "",
        }

    place = resolve_place("me", settings=settings, ip_lookup=ip_lookup, get_json=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network")))
    assert place["source"] == "ip"
    assert place["state_code"] == "CO"


def test_ip_lookup_failure(settings):
    def ip_lookup():
        raise OSError("offline")

    place = resolve_place("here", settings=settings, ip_lookup=ip_lookup)
    assert place["latitude"] is None
    assert place["source"] == "ip"


def test_resolve_place_tool_is_a_langchain_tool():
    assert resolve_place_tool.name == "resolve_place_tool"
    assert "coordinates" in resolve_place_tool.description


def test_settings_read_model_and_keys_from_the_environment(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-mini")
    monkeypatch.setenv("NPS_API_KEY", "test-key")
    monkeypatch.setenv("NPS_MCP_ENABLED", "false")
    loaded = get_settings()
    assert loaded.model_provider == "openai"
    assert loaded.model_name == "gpt-4.1-mini"
    assert loaded.nps_api_key == "test-key"
    assert loaded.nps_mcp_enabled is False
