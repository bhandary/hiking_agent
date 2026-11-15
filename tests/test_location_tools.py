"""Tests for the location agent's place resolver."""

from __future__ import annotations

import pytest

from hike_agent.agents.location import resolve_place, resolve_place_tool
from hike_agent.config import get_settings
from hike_agent.services.nps import parks_from_find_text
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


YOSEMITE_PARK = {
    "fullName": "Yosemite National Park",
    "parkCode": "yose",
    "designation": "National Park",
    "states": "CA",
    "latitude": 37.84883288,
    "longitude": -119.5571873,
}


def test_yosemite_prefers_the_national_park_over_the_first_geocode_hit(settings):
    def get_json(url, **_kwargs):
        assert "geocoding-api.open-meteo.com" in url
        return {
            "results": [
                {
                    "name": "Yosemite",
                    "latitude": 37.34674,
                    "longitude": -84.82467,
                    "admin1": "Kentucky",
                    "country": "United States",
                    "country_code": "US",
                    "feature_code": "PPL",
                    "population": None,
                },
                {
                    "name": "Yosemite Valley",
                    "latitude": 37.74075,
                    "longitude": -119.57788,
                    "admin1": "California",
                    "country": "United States",
                    "country_code": "US",
                    "feature_code": "PPL",
                    "population": 1035,
                },
            ]
        }

    def search_parks(_query):
        return [
            YOSEMITE_PARK,
            {
                "fullName": "Devils Postpile National Monument",
                "parkCode": "depo",
                "designation": "National Monument",
                "states": "CA",
                "latitude": 37.629,
                "longitude": -119.085,
            },
        ]

    place = resolve_place("Yosemite", settings=settings, get_json=get_json, search_parks=search_parks)
    assert place["source"] == "nps"
    assert place["name"] == "Yosemite National Park"
    assert place["state_code"] == "CA"
    assert place["state"] == "California"
    assert place["latitude"] == pytest.approx(37.84883288)
    assert place["longitude"] == pytest.approx(-119.5571873)


def test_geocode_ranks_hits_instead_of_taking_the_first(settings):
    def yosemite_hits(url, **_kwargs):
        assert "geocoding-api.open-meteo.com" in url
        return {
            "results": [
                {
                    "name": "Yosemite",
                    "latitude": 37.34674,
                    "longitude": -84.82467,
                    "admin1": "Kentucky",
                    "country_code": "US",
                    "country": "United States",
                    "feature_code": "PPL",
                    "population": 0,
                },
                {
                    "name": "Yosemite Lakes",
                    "latitude": 37.19106,
                    "longitude": -119.77265,
                    "admin1": "California",
                    "country_code": "US",
                    "country": "United States",
                    "feature_code": "PPL",
                    "population": 4952,
                },
            ]
        }

    populated = resolve_place("Yosemite", settings=settings, get_json=yosemite_hits, search_parks=lambda _query: [])
    assert populated["state_code"] == "CA"
    assert populated["longitude"] == pytest.approx(-119.77265)

    def gateway_hits(url, **_kwargs):
        assert "geocoding-api.open-meteo.com" in url
        return {
            "results": [
                {
                    "name": "El Portal",
                    "latitude": 37.35,
                    "longitude": -84.82,
                    "admin1": "Kentucky",
                    "country_code": "US",
                    "feature_code": "PPL",
                    "population": 200,
                },
                {
                    "name": "El Portal",
                    "latitude": 37.6747,
                    "longitude": -119.7843,
                    "admin1": "California",
                    "country_code": "US",
                    "feature_code": "PPL",
                    "population": 200,
                },
            ]
        }

    nearest = resolve_place(
        "El Portal",
        settings=settings,
        get_json=gateway_hits,
        search_parks=lambda _query: [YOSEMITE_PARK],
    )
    assert nearest["source"] == "open-meteo"
    assert nearest["state_code"] == "CA"
    assert nearest["longitude"] == pytest.approx(-119.7843)


def test_hosted_park_markdown_keeps_coordinates():
    parks = parks_from_find_text(
        "### Yosemite National Park\n"
        "**parkCode:** yose | **National Park** | **States:** CA\n"
        "A valley.\n"
        "**Coordinates:** 37.84883288, -119.5571873\n"
    )
    assert parks[0]["parkCode"] == "yose"
    assert parks[0]["latitude"] == pytest.approx(37.84883288)
    assert parks[0]["states"] == "CA"


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
    assert place["source"] == "error"
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


def test_near_me_punctuation_and_hike_wording_do_not_geocode(settings):
    def get_json(*_args, **_kwargs):
        raise AssertionError("near me must not geocode")

    for query in (
        "me!",
        "near me!",
        "hike near me",
        "day hike near me",
        "nearby",
        "around me",
        "close to me",
        "hike nearby",
    ):
        place = resolve_place(query, settings=settings, get_json=get_json)
        assert place["source"] == "needs-location", query


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
