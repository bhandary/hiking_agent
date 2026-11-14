"""Tests for the parks agent's tools. Network and MCP calls are faked."""

from __future__ import annotations

import requests

from hike_agent.agents.parks import find_hikes, parks_from_mcp_result
from hike_agent.config import Settings
from tests.helpers import mcp_down

PLACE = {
    "name": "Boulder",
    "latitude": 40.015,
    "longitude": -105.271,
    "state_code": "CO",
}


def test_mcp_markdown_fallback_parses_park_codes():
    text = """## 1 parks

### Rocky Mountain National Park
**parkCode:** romo | **National Park** | **States:** CO
High peaks above Estes Park.
"""
    parks = parks_from_mcp_result({"ok": True, "text": text, "structured": None, "error": None})
    assert parks[0]["name"] == "Rocky Mountain National Park"
    assert parks[0]["park_code"] == "romo"


def test_mcp_error_payload_is_not_a_park_list():
    result = {"ok": False, "text": "Error: upstream", "structured": {"error": {"message": "503"}}, "error": "503"}
    assert parks_from_mcp_result(result) == []


def test_mcp_success_attaches_trails_and_alerts(settings):
    def mcp_call(_target, tool, arguments, _timeout):
        if tool == "nps_find_parks":
            assert arguments["stateCode"] == "CO"
            return {
                "ok": True,
                "text": "",
                "error": None,
                "structured": {
                    "parks": [
                        {
                            "fullName": "Rocky Mountain National Park",
                            "parkCode": "romo",
                            "description": "Alpine tundra and lakes.",
                            "designation": "National Park",
                            "latitude": "40.34",
                            "longitude": "-105.68",
                            "url": "https://www.nps.gov/romo/",
                            "activities": ["Hiking"],
                        }
                    ]
                },
            }
        if tool == "nps_get_activities":
            assert arguments["parkCode"] == "romo"
            return {
                "ok": True,
                "text": "",
                "error": None,
                "structured": {"activities": [{"title": "Bear Lake Trail", "tags": ["hiking"]}]},
            }
        if tool == "nps_get_alerts":
            return {
                "ok": True,
                "text": "",
                "error": None,
                "structured": {"alerts": [{"title": "Trail Ridge Road closed", "category": "Closure"}]},
            }
        raise AssertionError(tool)

    report = find_hikes(PLACE, settings=settings, mcp_call=mcp_call, get_json=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("http")))
    assert report["source"] == "nps-mcp"
    assert report["parks"][0]["name"] == "Rocky Mountain National Park"
    assert report["parks"][0]["trails"] == ["Bear Lake Trail"]
    assert "Trail Ridge Road closed" in report["parks"][0]["alerts"][0]


def test_mcp_failure_falls_back_to_the_nps_api(settings: Settings):
    settings = Settings(**{**settings.__dict__, "nps_api_key": "secret-key"})

    def get_json(url, **kwargs):
        assert kwargs["params"]["api_key"] == "secret-key"
        if url.endswith("/parks"):
            return {
                "data": [
                    {
                        "fullName": "Rocky Mountain National Park",
                        "parkCode": "romo",
                        "description": "Mountains.",
                        "designation": "National Park",
                        "latitude": "40.34",
                        "longitude": "-105.68",
                        "url": "https://www.nps.gov/romo/",
                    },
                    {
                        "fullName": "Far Away Historic Site",
                        "parkCode": "farw",
                        "designation": "Historic Site",
                        "latitude": "37.0",
                        "longitude": "-109.0",
                    },
                ]
            }
        code = kwargs["params"].get("parkCode")
        if url.endswith("/thingstodo"):
            if code == "romo":
                return {"data": [{"title": "Emerald Lake Trail", "tags": ["hiking"]}, {"title": "Gift shop"}]}
            return {"data": []}
        if url.endswith("/alerts"):
            if code == "romo":
                return {"data": [{"title": "Bear activity", "category": "Caution"}]}
            return {"data": []}
        raise AssertionError(url)

    report = find_hikes(PLACE, settings=settings, mcp_call=mcp_down, get_json=get_json)
    assert report["source"] == "nps-api"
    assert report["parks"][0]["park_code"] == "romo"
    assert "Emerald Lake Trail" in report["parks"][0]["trails"]
    assert report["parks"][0]["alerts"] == ["Caution: Bear activity"]


def test_without_a_key_openstreetmap_is_the_fallback(settings):
    def get_json(_url, **kwargs):
        query = kwargs["params"]["q"]
        if query == "hiking":
            return [
                {
                    "name": "Bluebell Hiking Trail",
                    "display_name": "Bluebell Hiking Trail, Boulder",
                    "type": "path",
                    "lat": "39.995",
                    "lon": "-105.283",
                    "osm_type": "way",
                    "osm_id": 42,
                },
                {
                    "name": "National Park Village",
                    "display_name": "A housing area",
                    "type": "village",
                    "lat": "40.02",
                    "lon": "-105.27",
                    "osm_type": "node",
                    "osm_id": 7,
                },
            ]
        return []

    def overpass(_lat, _lon, *, timeout, user_agent):
        assert timeout == settings.http_timeout
        assert user_agent == settings.user_agent
        return [
            {
                "name": "Boulder Valley Ranch Trails",
                "description": "OpenStreetMap hiking",
                "designation": "hiking",
                "latitude": 40.087,
                "longitude": -105.259,
                "url": "https://www.openstreetmap.org/relation/1",
                "trails": ["Boulder Valley Ranch Trails"],
                "alerts": [],
                "park_code": None,
            }
        ]

    report = find_hikes(
        PLACE,
        settings=settings,
        mcp_call=mcp_down,
        get_json=get_json,
        overpass=overpass,
    )
    assert report["source"] == "openstreetmap"
    names = {park["name"] for park in report["parks"]}
    assert "Bluebell Hiking Trail" in names
    assert "Boulder Valley Ranch Trails" in names
    assert "National Park Village" not in names
    assert "not an official National Park Service" in report["note"]


def test_all_park_sources_down(settings):
    def get_json(*_args, **_kwargs):
        raise requests.ConnectionError("nominatim down")

    def overpass(*_args, **_kwargs):
        raise requests.ConnectionError("overpass down")

    report = find_hikes(PLACE, settings=settings, mcp_call=mcp_down, get_json=get_json, overpass=overpass)
    assert report["source"] == "none"
    assert report["parks"] == []
    assert "NPS_API_KEY" in report["note"]


def test_nps_http_error_does_not_include_the_api_key(settings: Settings):
    settings = Settings(**{**settings.__dict__, "nps_api_key": "secret-key"})
    response = requests.Response()
    response.status_code = 403
    error = requests.HTTPError("https://developer.nps.gov/api/v1/parks?api_key=secret-key")
    error.response = response

    def get_json(*_args, **_kwargs):
        raise error

    report = find_hikes(PLACE, settings=settings, mcp_call=mcp_down, get_json=get_json, overpass=lambda *_a, **_k: [])
    assert "secret-key" not in report["note"]
    assert "403" in report["note"]
