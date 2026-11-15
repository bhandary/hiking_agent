"""The planner must not keep place names the tools did not return."""

from hike_agent.agents.planner import invented_place_names

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
