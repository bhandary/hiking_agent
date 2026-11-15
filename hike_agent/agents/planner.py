"""Planner agent: write the hike recommendation from the other agents' facts.

The graph calls the configured chat model with this brief. When the model is
unreachable, the brief itself is the recommendation, so safety notes still appear.
"""

from __future__ import annotations

from typing import Any

from hike_agent.agents.parks import NOT_VERIFIED, NPS_SOURCES

AGENT_NAME = "planner"
AGENT_BRIEF = "Combine location, parks, and conditions into a short hike recommendation with safety notes."

PLANNER_SYSTEM = """You are a hiking guide. Rewrite the factual brief into a short recommendation.
Rules:
- Recommend at most three options that appear in the brief. Do not invent parks, trails, temperatures, or alerts.
- If the risk is unsafe, say not to hike today and offer the options only as later ideas.
- Keep every safety note. A missing alert is not an all-clear.
- When a place is marked "not verified by NPS", repeat that phrase. Do not present those names as National Park Service data.
- Stay under 250 words.
"""


def _nps_verified(park: dict[str, Any], report: dict[str, Any]) -> bool:
    if "nps_verified" in park:
        return bool(park["nps_verified"])
    source = park.get("source") or report.get("source")
    return source in NPS_SOURCES


def render_brief(state: dict[str, Any]) -> str:
    """Factual brief shared by the model and the template fallback."""

    place = state.get("place") or {}
    parks_report = state.get("parks") or {}
    conditions = state.get("conditions") or {}
    label = place.get("label") or state.get("query") or "the requested place"
    when = state.get("when") or {}
    lines = [f"Hike plan for {label}"]
    if isinstance(when, dict) and when.get("label"):
        lines.append(f"When: {when['label']}")
    lines.append("")
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
        label = f"{index}. {park.get('name')} — {trails}"
        if not _nps_verified(park, parks_report):
            label = f"{label} ({NOT_VERIFIED})"
        lines.append(label)
        if park.get("source"):
            lines.append(f"   Source: {park['source']}")
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
    parks_report = state.get("parks") or {}
    notes = list(conditions.get("safety_notes") or [])
    if parks_report.get("parks") and not parks_report.get("nps_verified", parks_report.get("source") in NPS_SOURCES):
        notes.append(f"Park and trail names in this plan are {NOT_VERIFIED}.")
    for park in parks_report.get("parks") or []:
        for alert in park.get("alerts") or []:
            notes.append(f"{park.get('name')}: {alert}")
    unique: list[str] = []
    for note in notes:
        if note not in unique:
            unique.append(note)
    if not unique:
        return ""
    return "Safety notes\n" + "\n".join(f"- {note}" for note in unique)
