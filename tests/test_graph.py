"""Integration-style test of the LangGraph supervisor.

Tool functions and the planner model are fakes. The test checks that the graph
resolves a place, runs parks and weather on those coordinates, and that the
planner sees both reports.
"""

from __future__ import annotations

import pytest

from hike_agent.graph import build_graph


PLACE = {
    "query": "Boulder, Colorado",
    "name": "Boulder",
    "label": "Boulder, Colorado, United States",
    "latitude": 40.015,
    "longitude": -105.271,
    "state": "Colorado",
    "state_code": "CO",
    "country": "United States",
    "source": "open-meteo",
    "note": "",
}

PARKS = {
    "source": "nps-api",
    "note": "",
    "parks": [
        {
            "name": "Rocky Mountain National Park",
            "park_code": "romo",
            "description": "Alpine lakes.",
            "designation": "National Park",
            "trails": ["Bear Lake Trail"],
            "alerts": ["Closure: Trail Ridge Road closed"],
        }
    ],
}

CONDITIONS = {
    "source": "open-meteo+nws",
    "summary": "Today's forecast: Partly cloudy, with an average temperature of 10°C and a maximum precipitation probability of 15%.",
    "suitable": True,
    "risk": "ok",
    "safety_notes": ["No active weather alerts were returned for this point."],
    "alerts": [],
}


def test_graph_joins_location_parks_weather_and_planner():
    seen = {}

    def resolve_place(query: str):
        seen["query"] = query
        return PLACE

    def find_hikes(place):
        seen["parks_place"] = place
        return PARKS

    def hiking_conditions(place):
        seen["weather_place"] = place
        return CONDITIONS

    def complete(brief: str) -> str:
        seen["brief"] = brief
        return "Hike Bear Lake Trail in Rocky Mountain National Park for the lake views."

    graph = build_graph(resolve_place, find_hikes, hiking_conditions, complete)
    state = graph.invoke({"query": "Boulder, Colorado", "errors": [], "trace": []})

    assert seen["query"] == "Boulder, Colorado"
    assert seen["parks_place"]["state_code"] == "CO"
    assert seen["weather_place"]["latitude"] == pytest.approx(40.015)
    assert "Rocky Mountain National Park" in seen["brief"]
    assert "Bear Lake Trail" in seen["brief"]
    assert "Partly cloudy" in seen["brief"]
    assert "Trail Ridge Road closed" in seen["brief"]
    assert state["planner"] == "model"
    assert "Bear Lake Trail" in state["recommendation"]
    assert "Safety notes" in state["recommendation"]
    assert "Trail Ridge Road closed" in state["recommendation"]
    assert state["trace"][0].startswith("location:")
    assert any(line.startswith("parks:") for line in state["trace"])
    assert any(line.startswith("weather:") for line in state["trace"])
    assert state["trace"][-1] == "planner: model"


def test_graph_uses_the_template_when_the_model_fails(caplog):
    def complete(_brief: str) -> str:
        raise ConnectionError("the model request failed")

    graph = build_graph(
        lambda _query: PLACE,
        lambda _place: PARKS,
        lambda _place: CONDITIONS,
        complete,
    )
    with caplog.at_level("WARNING"):
        state = graph.invoke({"query": "Boulder, Colorado", "errors": [], "trace": []})
    assert state["planner"] == "template"
    assert "Rocky Mountain National Park" in state["recommendation"]
    assert "poor day" not in state["recommendation"].lower()
    assert any("Planner model call failed" in error for error in state["errors"])
    assert "ConnectionError" in state["errors"][0]
    assert "factual brief" in caplog.text
    assert "the model request failed" in caplog.text


def test_named_date_reaches_weather_and_the_brief():
    seen = {}

    def hiking_conditions(place, today=None):
        seen["today"] = today
        return CONDITIONS

    def complete(brief: str) -> str:
        seen["brief"] = brief
        return "Hike later."

    graph = build_graph(lambda _query: PLACE, lambda _place: PARKS, hiking_conditions, complete)
    graph.invoke(
        {
            "query": "Boulder",
            "when": {"label": "November 20 (2099-11-20)", "start": "2099-11-20", "end": "2099-11-20"},
            "errors": [],
            "trace": [],
        }
    )
    assert seen["today"].isoformat() == "2099-11-20"
    assert "When: November 20 (2099-11-20)" in seen["brief"]


def test_weather_fake_without_a_date_argument_still_runs():
    def hiking_conditions(place):
        return CONDITIONS

    graph = build_graph(lambda _query: PLACE, lambda _place: PARKS, hiking_conditions, lambda _brief: "ok")
    state = graph.invoke(
        {
            "query": "Boulder",
            "when": {"label": "this weekend", "start": "2099-11-20", "end": "2099-11-21"},
            "errors": [],
            "trace": [],
        }
    )
    assert state["conditions"]["source"] == "open-meteo+nws"


def test_graph_stops_when_the_place_cannot_be_resolved():
    def fail(_place):
        raise AssertionError("parks and weather should not run without coordinates")

    def complete(_brief: str) -> str:
        raise AssertionError("the planner should not run when the place was not found")

    graph = build_graph(
        lambda _query: {"latitude": None, "longitude": None, "label": "unknown", "source": "none", "note": "No coordinates found."},
        fail,
        fail,
        complete,
    )
    state = graph.invoke({"query": "not a place", "errors": [], "trace": []})
    assert state["recommendation"] == "I couldn't find that place, try a city/park name."
    assert state["planner"] == "stopped"
    assert state["trace"] == ["location: stopped via none"]


def test_near_me_without_coordinates_asks_instead_of_planning():
    def fail(_place):
        raise AssertionError("parks and weather should not run")

    graph = build_graph(
        lambda _query: {"latitude": None, "longitude": None, "label": "me", "source": "needs-location", "note": "need a place"},
        fail,
        fail,
        lambda _brief: "should not be called",
    )
    state = graph.invoke({"query": "me", "errors": [], "trace": []})
    assert "city or park name" in state["recommendation"]
    assert state["planner"] == "stopped"


def test_forecast_is_kept_when_the_model_omits_it():
    def complete(_brief: str) -> str:
        return "Hike Bear Lake Trail in Rocky Mountain National Park."

    graph = build_graph(lambda _query: PLACE, lambda _place: PARKS, lambda _place: CONDITIONS, complete)
    state = graph.invoke({"query": "Boulder", "errors": [], "trace": []})
    assert "Partly cloudy" in state["recommendation"]
    assert "Bear Lake Trail" in state["recommendation"]
    assert state["planner"] == "model"


def test_invented_parks_are_replaced_with_tool_results():
    calls = {"n": 0}

    def complete(_brief: str) -> str:
        calls["n"] += 1
        return "Sunrise is best from Imaginary Peak Trail."

    graph = build_graph(lambda _query: PLACE, lambda _place: PARKS, lambda _place: CONDITIONS, complete)
    state = graph.invoke({"query": "Boulder", "errors": [], "trace": []})
    assert calls["n"] == 2
    assert "Imaginary Peak Trail" not in state["recommendation"]
    assert "Bear Lake Trail" in state["recommendation"]
    assert "Rocky Mountain National Park" in state["recommendation"]
    assert "Partly cloudy" in state["recommendation"]
    assert state["planner"] == "grounded"


def test_geocoding_failure_is_not_reported_as_an_unknown_place():
    def fail(_place):
        raise AssertionError("parks and weather should not run when geocoding failed")

    graph = build_graph(
        lambda _query: {
            "latitude": None,
            "longitude": None,
            "label": "Boulder",
            "source": "error",
            "note": "Geocoding failed (TimeoutError).",
        },
        fail,
        fail,
        lambda _brief: "should not be called",
    )
    state = graph.invoke({"query": "Boulder", "errors": [], "trace": []})
    assert state["recommendation"] == "Geocoding failed (TimeoutError)."
    assert state["planner"] == "stopped"
