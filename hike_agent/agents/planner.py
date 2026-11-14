"""Planner agent: write the hike recommendation from the other agents' facts.

The graph calls the configured chat model with this brief. When the model is
unreachable, the brief itself is the recommendation, so safety notes still appear.
"""

from __future__ import annotations

from typing import Any

AGENT_NAME = "planner"
AGENT_BRIEF = "Combine location, parks, and conditions into a short hike recommendation with safety notes."

PLANNER_SYSTEM = """You are a hiking guide. Rewrite the factual brief into a short recommendation.
Rules:
- Recommend at most three options that appear in the brief. Do not invent parks, trails, temperatures, or alerts.
- If the risk is unsafe, say not to hike today and offer the options only as later ideas.
- Keep every safety note. A missing alert is not an all-clear.
- When the parks source is OpenStreetMap, say these are community-mapped places, not an official National Park Service listing.
- Stay under 250 words.
"""


def render_brief(state: dict[str, Any]) -> str:
    """Factual brief shared by the model and the template fallback."""

    place = state.get("place") or {}
    parks_report = state.get("parks") or {}
    conditions = state.get("conditions") or {}
    label = place.get("label") or state.get("query") or "the requested place"
    lines = [f"Hike plan for {label}", ""]
    lines.append("Conditions")
    lines.append(conditions.get("summary") or "Weather was not available.")
    lines.append(f"Risk: {conditions.get('risk') or 'unknown'}.")
    for note in conditions.get("safety_notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    if conditions.get("risk") == "unsafe":
        lines.append("Today is a poor day to hike. Treat the options below as later-trip ideas, not a green light to go out.")
        lines.append("")
    lines.append("Options")
    parks = parks_report.get("parks") or []
    if not parks:
        lines.append("No parks or trails were found.")
    for index, park in enumerate(parks, start=1):
        trails = ", ".join(park.get("trails") or []) or "no named trails in the data"
        lines.append(f"{index}. {park.get('name')} — {trails}")
        if park.get("description"):
            lines.append(f"   {park['description']}")
        for alert in park.get("alerts") or []:
            lines.append(f"   Safety: {alert}")
    if parks_report.get("note"):
        lines.append("")
        lines.append(parks_report["note"])
    return "\n".join(lines).strip()


def safety_footer(state: dict[str, Any]) -> str:
    """Repeat structured safety notes under a model-written recommendation."""

    conditions = state.get("conditions") or {}
    notes = list(conditions.get("safety_notes") or [])
    for park in (state.get("parks") or {}).get("parks") or []:
        for alert in park.get("alerts") or []:
            notes.append(f"{park.get('name')}: {alert}")
    unique: list[str] = []
    for note in notes:
        if note not in unique:
            unique.append(note)
    if not unique:
        return ""
    return "Safety notes\n" + "\n".join(f"- {note}" for note in unique)
