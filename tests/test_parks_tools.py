"""Tests for the parks agent's tools. Network and MCP calls are faked."""

from __future__ import annotations

import requests

from hike_agent.agents.parks import find_hikes, parks_from_mcp_result
from hike_agent.services.nps import rank_parks
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


def test_yosemite_mcp_text_keeps_coordinates_and_ranks_the_park_first(settings):
    text = """## 3 parks

### Devils Postpile National Monument
**parkCode:** depo | **National Monument** | **States:** CA
A basalt cliff south of the valley.
**Coordinates:** 37.62985, -119.0846

### Yosemite National Park
**parkCode:** yose | **National Park** | **States:** CA
Not just a great valley, but a shrine to human foresight.
**Coordinates:** 37.84883288, -119.5571873

### Mammoth Cave National Park
**parkCode:** maca | **National Park** | **States:** KY
A cave in Kentucky.
**Coordinates:** 37.186, -86.1
"""
    parsed = parks_from_mcp_result({"ok": True, "text": text, "structured": None, "error": None})
    yosemite = next(park for park in parsed if park["park_code"] == "yose")
    assert yosemite["latitude"] == 37.84883288
    assert yosemite["longitude"] == -119.5571873

    def mcp_call(_target, tool, _arguments, _timeout):
        if tool == "nps_find_parks":
            return {"ok": True, "text": text, "structured": None, "error": None}
        return {"ok": True, "text": "", "structured": None, "error": None}

    report = find_hikes(
        {
            "name": "Yosemite",
            "latitude": 37.84883288,
            "longitude": -119.5571873,
            "state_code": "CA",
        },
        settings=settings,
        mcp_call=mcp_call,
        get_json=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("OpenStreetMap should not run")),
    )
    assert report["source"] == "nps-mcp"
    assert report["parks"][0]["name"] == "Yosemite National Park"
    assert report["parks"][0]["park_code"] == "yose"
    assert "160 km" not in report["note"]
    assert all(park["park_code"] != "maca" for park in report["parks"])


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
    seen_targets = []

    def mcp_call(target, tool, _arguments, _timeout):
        assert tool == "nps_find_parks"
        seen_targets.append(target)
        return mcp_down()

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
        mcp_call=mcp_call,
        get_json=get_json,
        overpass=overpass,
    )
    assert seen_targets == [settings.nps_mcp_url]
    assert report["source"] == "openstreetmap"
    names = {park["name"] for park in report["parks"]}
    assert "Bluebell Hiking Trail" in names
    assert "Boulder Valley Ranch Trails" in names
    assert "National Park Village" not in names
    assert "not verified by NPS" in report["note"]
    assert report["nps_verified"] is False
    assert {park["source"] for park in report["parks"]} == {"openstreetmap"}
    assert all(park["nps_verified"] is False for park in report["parks"])


def test_all_park_sources_down(settings):
    def get_json(*_args, **_kwargs):
        raise requests.ConnectionError("nominatim down")

    def overpass(*_args, **_kwargs):
        raise requests.ConnectionError("overpass down")

    report = find_hikes(PLACE, settings=settings, mcp_call=mcp_down, get_json=get_json, overpass=overpass)
    assert report["source"] == "none"
    assert report["parks"] == []
    assert "NPS_API_KEY" in report["note"]


def _romo_payload():
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


def test_api_key_tries_local_mcp_before_hosted_then_the_api(settings: Settings):
    settings = Settings(**{**settings.__dict__, "nps_api_key": "secret-key"})
    seen = []

    def mcp_call(target, tool, _arguments, _timeout):
        assert tool == "nps_find_parks"
        seen.append(target)
        return {"ok": False, "text": "", "structured": None, "error": "down"}

    def get_json(url, **kwargs):
        assert "secret-key" == kwargs["params"]["api_key"]
        if url.endswith("/parks"):
            return {
                "data": [
                    {
                        "fullName": "Rocky Mountain National Park",
                        "parkCode": "romo",
                        "designation": "National Park",
                        "latitude": "40.34",
                        "longitude": "-105.68",
                    }
                ]
            }
        return {"data": []}

    report = find_hikes(PLACE, settings=settings, mcp_call=mcp_call, get_json=get_json)
    local, hosted = seen
    assert local["mcpServers"]["nps"]["command"] == "npx"
    assert "@cyanheads/national-parks-mcp-server@0.3.0" in local["mcpServers"]["nps"]["args"]
    assert local["mcpServers"]["nps"]["env"]["NPS_API_KEY"] == "secret-key"
    assert hosted == settings.nps_mcp_url
    assert report["source"] == "nps-api"
    assert report["nps_verified"] is True
    assert report["parks"][0]["source"] == "nps-api"
    assert report["parks"][0]["nps_verified"] is True
    assert "secret-key" not in report["note"]
    assert "not verified by NPS" not in report["note"]


def test_local_mcp_success_skips_hosted_mcp_and_the_api(settings: Settings):
    settings = Settings(**{**settings.__dict__, "nps_api_key": "secret-key"})
    targets = []

    def mcp_call(target, tool, _arguments, _timeout):
        targets.append(target)
        if tool == "nps_find_parks":
            return _romo_payload()
        if tool == "nps_get_activities":
            return {
                "ok": True,
                "text": "",
                "error": None,
                "structured": {"activities": [{"title": "Bear Lake Trail", "tags": ["hiking"]}]},
            }
        if tool == "nps_get_alerts":
            return {"ok": True, "text": "", "error": None, "structured": {"alerts": []}}
        raise AssertionError(tool)

    def get_json(*_args, **_kwargs):
        raise AssertionError("direct API should not run after local MCP succeeds")

    report = find_hikes(PLACE, settings=settings, mcp_call=mcp_call, get_json=get_json)
    assert report["source"] == "nps-mcp-local"
    assert report["nps_verified"] is True
    assert all(isinstance(target, dict) for target in targets)
    assert report["parks"][0]["source"] == "nps-mcp-local"
    assert report["parks"][0]["trails"] == ["Bear Lake Trail"]


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


def test_coordinates_without_a_place_name_skip_the_park_name_search(settings):
    def mcp_call(*_args):
        raise AssertionError("NPS must not be queried for a placeholder or an empty name")

    def get_json(_url, **kwargs):
        assert "Provided location" not in str(kwargs.get("params"))
        return []

    for name in (None, "Provided location"):
        report = find_hikes(
            {"name": name, "latitude": 40.0, "longitude": -105.2, "state_code": None},
            settings=settings,
            mcp_call=mcp_call,
            get_json=get_json,
            overpass=lambda *_a, **_k: [],
        )
        assert "park-name search was skipped" in report["note"]
        assert report["parks"] == []


def test_parks_without_coordinates_are_not_ranked_near_boulder():
    boulder = (40.015, -105.271)
    parks = [
        {
            "name": "Rocky Mountain National Park",
            "designation": "National Park",
            "latitude": 40.34,
            "longitude": -105.68,
        },
        {
            "name": "Colorado National Monument",
            "designation": "National Monument",
            "latitude": 39.07,
            "longitude": -108.69,
        },
        {
            "name": "Hovenweep National Monument",
            "designation": "National Monument",
            "latitude": None,
            "longitude": None,
        },
    ]
    ranked = rank_parks(parks, *boulder, limit=4)
    assert [park["name"] for park in ranked] == ["Rocky Mountain National Park"]


def test_only_coordinate_less_parks_are_reported_as_none_nearby(settings):
    def mcp_call(_target, tool, _arguments, _timeout):
        if tool != "nps_find_parks":
            return {"ok": True, "structured": {}, "error": None}
        return {
            "ok": True,
            "error": None,
            "structured": {
                "parks": [
                    {"fullName": "Hovenweep National Monument", "parkCode": "hove", "designation": "National Monument"},
                    {
                        "fullName": "Colorado National Monument",
                        "parkCode": "colm",
                        "designation": "National Monument",
                        "latitude": "39.07",
                        "longitude": "-108.69",
                    },
                ]
            },
        }

    report = find_hikes(
        PLACE,
        settings=settings,
        mcp_call=mcp_call,
        get_json=lambda *_a, **_k: [],
        overpass=lambda *_a, **_k: [],
    )
    assert report["parks"] == []
    assert "within 160 km" in report["note"]
