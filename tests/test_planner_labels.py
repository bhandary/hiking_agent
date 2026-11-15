"""The plan must say when a place did not come from the National Park Service."""

from hike_agent.agents.planner import render_brief, safety_footer

CONDITIONS = {
    "summary": "Clear, 12C.",
    "risk": "ok",
    "safety_notes": ["No active weather alerts."],
}


def test_openstreetmap_places_are_labeled_not_verified_by_nps():
    state = {
        "query": "Boulder",
        "place": {"label": "Boulder, Colorado"},
        "conditions": CONDITIONS,
        "parks": {
            "source": "openstreetmap",
            "nps_verified": False,
            "note": "Results are from OpenStreetMap and are not verified by NPS.",
            "parks": [
                {
                    "name": "Bluebell Hiking Trail",
                    "trails": ["Bluebell Hiking Trail"],
                    "description": "A community-mapped path.",
                    "source": "openstreetmap",
                    "nps_verified": False,
                    "alerts": [],
                }
            ],
        },
    }
    brief = render_brief(state)
    assert "Bluebell Hiking Trail (not verified by NPS)" in brief
    assert "Source: openstreetmap" in brief
    assert "not verified by NPS" in safety_footer(state)


def test_nps_places_are_not_marked_unverified():
    state = {
        "query": "Estes Park",
        "place": {"label": "Estes Park, Colorado"},
        "conditions": CONDITIONS,
        "parks": {
            "source": "nps-mcp-local",
            "nps_verified": True,
            "note": "",
            "parks": [
                {
                    "name": "Rocky Mountain National Park",
                    "trails": ["Bear Lake Trail"],
                    "description": "Alpine lakes.",
                    "source": "nps-mcp-local",
                    "nps_verified": True,
                    "alerts": ["Closure: Trail Ridge Road closed"],
                }
            ],
        },
    }
    brief = render_brief(state)
    footer = safety_footer(state)
    assert "Rocky Mountain National Park" in brief
    assert "Source: nps-mcp-local" in brief
    assert "not verified by NPS" not in brief
    assert "not verified by NPS" not in footer
    assert "Trail Ridge Road closed" in brief
