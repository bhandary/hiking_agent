"""The planner must not keep facts the tools did not return."""

from hike_agent.agents.planner import (
    contradicted_risks,
    ground_recommendation,
    invented_place_names,
    invented_temperatures,
)

SUMMARY = (
    "Today's forecast: Partly cloudy, with an average temperature of 10°C "
    "and a maximum precipitation probability of 15%."
)


def _state():
    return {
        "conditions": {"summary": SUMMARY, "risk": "ok", "safety_notes": []},
        "parks": {
            "parks": [
                {"name": "Rocky Mountain National Park", "trails": ["Bear Lake Trail"]},
            ]
        },
        "when": {"trip_length": "day hike"},
    }


def test_a_real_trail_and_a_shorter_form_are_allowed():
    state = _state()
    assert invented_place_names("Hike Bear Lake Trail in Rocky Mountain National Park.", state) == []
    assert invented_place_names("Bear Lake is the shorter walk.", state) == []


def test_an_invented_prefix_on_a_real_trail_is_rejected():
    found = invented_place_names("Sunrise is best from Imaginary Bear Lake Trail.", _state())
    assert found == ["Imaginary Bear Lake", "Imaginary Bear Lake Trail"]


def test_a_lowercase_invented_trail_is_rejected():
    found = invented_place_names("Try imaginary peak trail at dawn.", _state())
    assert [name.lower() for name in found] == ["imaginary peak", "imaginary peak trail"]


def test_a_generic_national_park_phrase_is_not_treated_as_invented():
    assert invented_place_names("Visit National Park this weekend.", _state()) == []


def test_a_temperature_missing_from_the_forecast_is_rejected():
    assert invented_temperatures("It will be 95°F by noon.", _state()) == ["95°F"]
    assert invented_temperatures("The forecast average is 10°C.", _state()) == []


def test_a_contradicted_risk_line_is_rejected():
    assert contradicted_risks("Risk: unsafe. Stay home.", _state()) == ["Risk: unsafe"]
    assert contradicted_risks("Risk: ok.", _state()) == []


def test_grounding_drops_invented_facts_and_keeps_the_forecast():
    draft = "It will be 95°F. Hike Imaginary Peak Trail. Risk: unsafe."

    def complete(_brief: str) -> str:
        return draft

    body, how = ground_recommendation(draft, _state(), complete)
    assert how == "grounded"
    assert "95°F" not in body
    assert "Imaginary Peak Trail" not in body
    assert "Risk: unsafe" not in body
    assert "10°C" in body
    assert "Risk: ok" in body
    assert "Bear Lake Trail" in body
    assert "day hike" in body.lower()
