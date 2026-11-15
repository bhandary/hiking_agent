"""Grounding checks: invented places, weather, dates, distances, and follow-ups."""

import copy
from datetime import date

from hike_agent.agents.planner import (
    PLANNER_SYSTEM,
    _verbatim_in_tools,
    ground_recommendation,
    invented_place_names,
    render_brief,
)
from hike_agent.cli import answer_follow_up, plan_hike
from hike_agent.coordinates import parse_coordinates
from hike_agent.extract import INVALID_DATE, extract_hike_request
from hike_agent.graph import build_graph

SUMMARY = "Partly cloudy, 12 C, 20% chance of rain."
STATE = {
    "when": {"label": "this Saturday (2026-11-14)", "start": "2026-11-14", "end": "2026-11-14"},
    "conditions": {"summary": SUMMARY, "risk": "ok", "safety_notes": [], "alerts": []},
    "parks": {
        "source": "nps-api",
        "parks": [
            {
                "name": "Rocky Mountain National Park",
                "description": "Alpine lakes.",
                "trails": ["Bear Lake Trail"],
                "alerts": [],
            }
        ],
    },
}


def _reject(text: str):
    calls = {"n": 0}

    def complete(_prompt: str) -> str:
        calls["n"] += 1
        return text

    grounded, how = ground_recommendation(text, STATE, complete)
    return grounded, how, calls["n"]


def test_lowercase_and_unsuffixed_places_are_not_published():
    lowered, how, calls = _reject("hike the imaginary peak trail at dawn. it will be hot.")
    assert calls == 1
    assert how == "grounded"
    assert "imaginary peak trail" not in lowered.lower()
    assert "Bear Lake Trail" in lowered

    pond, pond_how, _pond_calls = _reject("Hike Sky Pond.")
    assert pond_how == "grounded"
    assert "Sky Pond" not in pond
    assert invented_place_names("Hike Sky Pond.", STATE) == ["Sky Pond"]


def test_a_longer_phrase_around_a_real_trail_name_is_rejected():
    assert invented_place_names("Hike the Imaginary Bear Lake Trail.", STATE) == ["Imaginary Bear Lake Trail"]
    assert invented_place_names("Bear Lake Trail!", STATE) == []
    assert invented_place_names("bear-lake trail", STATE) == []
    grounded, how, calls = _reject(
        "Hike the Imaginary Bear Lake Trail at dawn. Partly cloudy, 12 C, 20% chance of rain."
    )
    assert calls == 1
    assert how == "grounded"
    assert "Imaginary Bear Lake Trail" not in grounded
    assert "Bear Lake Trail" in grounded


def test_a_conflicting_risk_sentence_is_replaced_from_conditions():
    grounded, how, calls = _reject(
        "Hike Bear Lake Trail. Risk: unsafe. Partly cloudy, 12 C, 20% chance of rain."
    )
    assert calls == 1
    assert how == "grounded"
    assert "Risk: unsafe" not in grounded
    assert "Risk: ok." in grounded
    assert "Bear Lake Trail" in grounded


def test_a_slice_of_a_longer_tool_number_is_not_grounded():
    state = copy.deepcopy(STATE)
    state["conditions"]["summary"] = "49 F / November 21 / 12 miles"
    state["conditions"]["safety_notes"] = []
    false_text = "It will be 9 F on November 2. The trail is 2 miles. Hike Bear Lake Trail."
    grounded, how = ground_recommendation(false_text, state, lambda _prompt: false_text)
    assert how == "grounded"
    assert "It will be 9 F on November 2." not in grounded
    assert "The trail is 2 miles." not in grounded

    true_text = "It will be 49 F on November 21. The trail is 12 miles. Hike Bear Lake Trail."
    kept, how = ground_recommendation(true_text, state, lambda _prompt: true_text)
    assert how == "model"
    assert "49 F" in kept
    assert "November 21" in kept
    assert "12 miles" in kept

    decimal = copy.deepcopy(STATE)
    decimal["conditions"]["summary"] = "4.9 F"
    rejected, how = ground_recommendation(
        "It will be 49 F. It will be 9 F. Hike Bear Lake Trail.",
        decimal,
        lambda _prompt: "It will be 49 F. It will be 9 F. Hike Bear Lake Trail.",
    )
    assert how == "grounded"
    assert "It will be 49 F." not in rejected
    assert "It will be 9 F." not in rejected
    accepted, how = ground_recommendation(
        "It will be 4.9 F. Hike Bear Lake Trail.",
        decimal,
        lambda _prompt: "It will be 4.9 F. Hike Bear Lake Trail.",
    )
    assert how == "model"
    assert "4.9 F" in accepted


def test_a_signed_temperature_is_not_the_unsigned_number():
    minus = "\u2212"
    assert _verbatim_in_tools("5 F", "-5 F") is False
    assert _verbatim_in_tools("5.5 F", "-5.5 F") is False
    assert _verbatim_in_tools("5 F", "+5 F") is False
    assert _verbatim_in_tools("5 F", f"{minus}5 F") is False
    assert _verbatim_in_tools("5.5 F", f"{minus}5.5 F") is False
    assert _verbatim_in_tools("-5 F", "5 F") is False
    assert _verbatim_in_tools("-5.5 F", "5.5 F") is False
    assert _verbatim_in_tools("-5 F", "-5 F") is True
    assert _verbatim_in_tools("-5.5 F", "-5.5 F") is True
    assert _verbatim_in_tools("45 F", "40-45 F") is True
    assert _verbatim_in_tools("45 F", f"40{minus}45 F") is True
    assert _verbatim_in_tools("5 F", "40-45 F") is False

    def check(summary: str, text: str) -> tuple[str, str]:
        state = copy.deepcopy(STATE)
        state["conditions"]["summary"] = summary
        state["conditions"]["safety_notes"] = []
        return ground_recommendation(text, state, lambda _prompt: text)

    dropped, how = check("-5 F", "It will be 5 F. Hike Bear Lake Trail.")
    assert how == "grounded"
    assert "It will be 5 F." not in dropped

    dropped, how = check("-5.5 F", "It will be 5.5 F. Hike Bear Lake Trail.")
    assert how == "grounded"
    assert "It will be 5.5 F." not in dropped

    dropped, how = check("5 F", "It will be -5 F. Hike Bear Lake Trail.")
    assert how == "grounded"
    assert "It will be -5 F." not in dropped

    kept, how = check("-5 F", "It will be -5 F. Hike Bear Lake Trail.")
    assert how == "model"
    assert "-5 F" in kept

    kept, how = check("-5.5 F", "It will be -5.5 F. Hike Bear Lake Trail.")
    assert how == "model"
    assert "-5.5 F" in kept

    kept, how = check("40-45 F", "It will be 45 F. Hike Bear Lake Trail.")
    assert how == "model"
    assert "45 F" in kept


def test_numbers_and_dates_copied_from_tool_results_are_kept():
    state = copy.deepcopy(STATE)
    state["parks"]["parks"][0]["description"] = "The range runs 300 miles. Established September 14."
    text = (
        "The range runs 300 miles. Established September 14. "
        "Hike Bear Lake Trail. Partly cloudy, 12 C, 20% chance of rain."
    )
    grounded, how = ground_recommendation(text, state, lambda _prompt: text)
    assert how == "model"
    assert "300 miles" in grounded
    assert "September 14" in grounded


def test_a_future_unsafe_day_is_not_called_today():
    state = copy.deepcopy(STATE)
    state["when"] = {"label": "Saturday (2027-04-10)", "start": "2027-04-10", "end": "2027-04-10"}
    state["conditions"] = {
        "summary": "Thunderstorms, high 60 F.",
        "risk": "unsafe",
        "safety_notes": [],
        "alerts": [],
    }
    brief = render_brief(state)
    assert "today is a poor day" not in brief.lower()
    assert "hike today" not in brief.lower()
    assert "2027-04-10" in brief
    assert "poor day to hike" in brief.lower()
    assert "not to hike on the requested date" in PLANNER_SYSTEM
    assert "Do not say to hike today" in PLANNER_SYSTEM

    grounded, how = ground_recommendation(
        "It is not recommended to hike today. Hike Bear Lake Trail.",
        state,
        lambda _prompt: "It is not recommended to hike today. Hike Bear Lake Trail.",
    )
    assert how == "grounded"
    assert "hike today" not in grounded.lower()
    assert "today" not in grounded.lower()
    assert "2027-04-10" in grounded


def test_a_shorter_trail_name_is_not_the_tool_trail():
    text = "Hike the Lake Trail at dawn. Partly cloudy, 12 C, 20% chance of rain."
    grounded, how, calls = _reject(text)
    assert calls == 1
    assert how == "grounded"
    assert "at dawn" not in grounded
    assert invented_place_names(grounded, STATE) == []
    assert "Bear Lake Trail" in grounded


def test_forecast_date_distance_and_all_clear_contradictions_are_removed():
    drafts = [
        "It will be 95 F and sunny. Partly cloudy, 12 C, 20% chance of rain. Hike Bear Lake Trail.",
        "Go on Sunday, November 22. Hike Bear Lake Trail. Partly cloudy, 12 C, 20% chance of rain.",
        "Bear Lake Trail is 14 miles with 3000 ft of gain. Partly cloudy, 12 C, 20% chance of rain.",
        "No alerts are in effect, so it is safe. Hike Bear Lake Trail. Partly cloudy, 12 C, 20% chance of rain.",
    ]
    for draft in drafts:
        grounded, how, calls = _reject(draft)
        assert calls == 1
        assert how == "grounded"
        assert "95 F" not in grounded
        assert "November 22" not in grounded
        assert "14 miles" not in grounded
        assert "3000" not in grounded
        assert "it is safe" not in grounded.lower()
        assert "Bear Lake Trail" in grounded
        assert SUMMARY in grounded


def test_a_real_forecast_and_tool_names_stay_a_model_plan():
    text = f"Hike Bear Lake Trail in Rocky Mountain National Park. {SUMMARY}"
    grounded, how = ground_recommendation(text, STATE, lambda _prompt: "should not be called")
    assert how == "model"
    assert "Bear Lake Trail" in grounded
    assert SUMMARY in grounded


def test_follow_up_is_grounded():
    calls = {"n": 0}

    def complete(_prompt: str) -> str:
        calls["n"] += 1
        return "It will be 95 F and sunny. Hike Imaginary Peak Trail."

    answer = answer_follow_up(STATE, "What will the weather be?", complete)
    assert calls["n"] == 2
    assert "95 F" not in answer
    assert "Imaginary Peak Trail" not in answer
    assert SUMMARY in answer


def test_invalid_date_stops_before_planning():
    found = extract_hike_request("hike on 2026-02-30", today=date(2026, 11, 15))
    state = plan_hike(found["place"], when=found["when"], complete=lambda _brief: "should not run")
    assert state["planner"] == "stopped"
    assert state["recommendation"] == INVALID_DATE
    assert state["trace"] == ["extract: invalid date"]


def test_hike_tomorrow_does_not_resolve_the_word_hike():
    found = extract_hike_request("hike tomorrow", today=date(2026, 11, 15))
    assert found["place"] == ""

    def fail(_query, coordinates=None):
        raise AssertionError(f"location should be the empty place, not {_query!r}")

    graph = build_graph(
        lambda query, coordinates=None: {"latitude": None, "longitude": None, "source": "needs-location", "label": query},
        fail,
        fail,
        fail,
    )
    state = graph.invoke({"query": found["place"], "when": found["when"], "errors": [], "trace": []})
    assert state["planner"] == "stopped"
    assert "city or park name" in state["recommendation"]


def test_three_digit_longitude_is_not_sliced_into_another_point():
    assert parse_coordinates("-105.271, 40.015") is None
    assert parse_coordinates("lost -105.271, 40.015 today") is None
    assert parse_coordinates("40.015, -105.271") == (40.015, -105.271)
