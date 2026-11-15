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


def test_near_me_asks_for_a_place_instead_of_using_server_ip(settings):
    def get_json(*_args, **_kwargs):
        raise AssertionError("near me must not geocode or use the server IP")

    place = resolve_place("me", settings=settings, get_json=get_json)
    assert place["latitude"] is None
    assert place["source"] == "needs-location"
    assert "city or park name" in place["note"]

    here = resolve_place("here", settings=settings, get_json=get_json)
    assert here["source"] == "needs-location"


def test_near_me_uses_client_coordinates(settings):
    def get_json(url, **kwargs):
        assert url.endswith("/reverse")
        assert kwargs["params"]["lat"] == pytest.approx(40.0)
        assert kwargs["params"]["lon"] == pytest.approx(-105.2)
        return {"address": {"city": "Boulder", "state": "Colorado", "country": "United States"}}

    place = resolve_place(
        "me",
        settings=settings,
        get_json=get_json,
        coordinates={"latitude": 40.0, "longitude": -105.2},
    )
    assert place["source"] == "client"
    assert place["name"] == "Boulder"
    assert place["state_code"] == "CO"
    assert place["latitude"] == pytest.approx(40.0)
    assert place["longitude"] == pytest.approx(-105.2)


def test_client_coordinates_do_not_invent_a_place_name_when_reverse_fails(settings):
    def get_json(*_args, **_kwargs):
        raise TimeoutError("reverse down")

    place = resolve_place(
        "me",
        settings=settings,
        get_json=get_json,
        coordinates={"latitude": 40.0, "longitude": -105.2},
    )
    assert place["source"] == "client"
    assert place["name"] is None
    assert place["latitude"] == pytest.approx(40.0)


def test_activity_word_is_not_geocoded(settings):
    def get_json(*_args, **_kwargs):
        raise AssertionError("hike is not a place name")

    place = resolve_place("hike", settings=settings, get_json=get_json)
    assert place["latitude"] is None
    assert place["source"] == "needs-location"


def test_longitude_first_pair_is_not_a_client_point(settings):
    def get_json(*_args, **_kwargs):
        raise AssertionError("a sliced longitude must not be reverse geocoded")

    place = resolve_place("me", settings=settings, get_json=get_json, coordinates="-105.271, 40.015")
    assert place["source"] == "needs-location"
    assert place["latitude"] is None


def test_resolve_place_tool_is_a_langchain_tool():
    assert resolve_place_tool.name == "resolve_place_tool"
    assert "coordinates" in resolve_place_tool.description


def test_model_name_defaults_to_gpt_4_1_mini(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    assert get_settings().model_name == "gpt-4.1-mini"


def test_blank_model_name_uses_gpt_4_1_mini(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "  ")
    assert get_settings().model_name == "gpt-4.1-mini"


def test_settings_read_model_and_keys_from_the_environment(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1")
    monkeypatch.setenv("NPS_API_KEY", "test-key")
    monkeypatch.setenv("NPS_MCP_ENABLED", "false")
    loaded = get_settings()
    assert loaded.model_name == "gpt-4.1"
    assert loaded.nps_api_key == "test-key"
    assert loaded.nps_mcp_enabled is False
