"""LangGraph that supervises the hike agents.

The workflow is fixed: resolve the place, then collect parks and weather
together, then write the recommendation. That order is not chosen by a model.
A supervisor language model that picks the next agent could skip weather or
parks. This graph does not. The planner is the only model call, and it sees
both reports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from hike_agent.agents.location import resolve_place as default_resolve_place
from hike_agent.agents.parks import find_hikes as default_find_hikes
from hike_agent.agents.planner import render_brief, safety_footer
from hike_agent.agents.weather import hiking_conditions as default_hiking_conditions
from hike_agent.state import HikeState

Complete = Callable[[str], str]


def _has_coordinates(state: HikeState) -> bool:
    place = state.get("place") or {}
    return place.get("latitude") is not None and place.get("longitude") is not None


def build_graph(
    resolve_place: Callable[[str], dict[str, Any]] = default_resolve_place,
    find_hikes: Callable[[dict[str, Any]], dict[str, Any]] = default_find_hikes,
    hiking_conditions: Callable[[dict[str, Any]], dict[str, Any]] = default_hiking_conditions,
    complete: Complete | None = None,
):
    """Compile the hike graph. Pass fakes for ``complete`` and the tools in tests."""

    def location_node(state: HikeState) -> dict[str, Any]:
        place = resolve_place(state.get("query") or "")
        label = place.get("label") or state.get("query") or "unknown place"
        return {
            "place": place,
            "trace": [f"location: {label} via {place.get('source')}"],
        }

    def parks_node(state: HikeState) -> dict[str, Any]:
        report = find_hikes(state.get("place") or {})
        count = len(report.get("parks") or [])
        return {
            "parks": report,
            "trace": [f"parks: {count} places via {report.get('source')}"],
        }

    def weather_node(state: HikeState) -> dict[str, Any]:
        report = hiking_conditions(state.get("place") or {})
        return {
            "conditions": report,
            "trace": [f"weather: {report.get('risk', 'unknown')} risk via {report.get('source')}"],
        }

    def planner_node(state: HikeState) -> dict[str, Any]:
        brief = render_brief(state)
        if complete is None:
            return {"recommendation": brief, "planner": "template", "trace": ["planner: template"]}
        try:
            text = complete(brief).strip()
            if not text:
                raise RuntimeError("the model returned an empty recommendation")
        except Exception as exc:  # noqa: BLE001 - template output is the safe fallback
            return {
                "recommendation": brief,
                "planner": "template",
                "errors": [f"planner model failed: {type(exc).__name__}: {exc}"],
                "trace": ["planner: template after model error"],
            }
        footer = safety_footer(state)
        recommendation = f"{text}\n\n{footer}" if footer else text
        return {"recommendation": recommendation, "planner": "model", "trace": ["planner: model"]}

    def route_after_location(state: HikeState) -> str | list[str]:
        if not _has_coordinates(state):
            return "planner"
        return ["parks", "weather"]

    graph = StateGraph(HikeState)
    graph.add_node("location", location_node)
    graph.add_node("parks", parks_node)
    graph.add_node("weather", weather_node)
    graph.add_node("planner", planner_node)
    graph.add_edge(START, "location")
    graph.add_conditional_edges("location", route_after_location)
    graph.add_edge("parks", "planner")
    graph.add_edge("weather", "planner")
    graph.add_edge("planner", END)
    return graph.compile()
