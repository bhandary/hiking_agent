"""Planner agent: write the hike recommendation from the other agents' facts.

The graph calls the configured chat model with this brief. When the model is
unreachable, the brief itself is the recommendation, so safety notes still appear.
"""

from __future__ import annotations

import re
from typing import Any, Callable

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
- Include the forecast summary from the brief word for word.
"""

_SUFFIX_WORD = re.compile(
    r"\b(Trail|Park|Path|Loop|Canyon|Peak|Lake|Ridge|Wilderness|Monument|Creek|Mountain)\b",
    re.IGNORECASE,
)
_PREV_WORD = re.compile(r"([A-Za-z][\w'’.-]*)\s+$")
_TEMPERATURE = re.compile(r"-?\d+(?:\.\d+)?\s*°\s*[cCfF]")
_RISK_CLAIM = re.compile(r"\brisk\s*:\s*(ok|caution|unsafe)\b", re.IGNORECASE)
# Words that can sit inside a place name. "and" starts a second name, so it stops the walk.
_CONNECTORS = frozenset({"of", "the", "de", "la", "del", "du"})
_WALK_STOP = frozenset({
    "a",
    "an",
    "and",
    "are",
    "around",
    "at",
    "be",
    "best",
    "by",
    "for",
    "from",
    "hike",
    "hikes",
    "hiking",
    "in",
    "is",
    "near",
    "on",
    "or",
    "see",
    "take",
    "then",
    "to",
    "try",
    "via",
    "visit",
    "walk",
    "was",
    "with",
})
_GENERIC = frozenset({
    "area",
    "canyon",
    "creek",
    "forest",
    "lake",
    "loop",
    "monument",
    "mount",
    "mountain",
    "mt",
    "national",
    "park",
    "parks",
    "path",
    "paths",
    "peak",
    "point",
    "ridge",
    "state",
    "trail",
    "trails",
    "wilderness",
})


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
    if isinstance(when, dict) and when.get("trip_length"):
        lines.append(f"Trip length: {when['trip_length']}")
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


def forecast_section(state: dict[str, Any]) -> str:
    """The looked-up forecast, copied so a model rewrite cannot drop it."""

    conditions = state.get("conditions") or {}
    summary = (conditions.get("summary") or "Weather was not available.").strip()
    risk = conditions.get("risk") or "unknown"
    return f"Forecast\n{summary}\nRisk: {risk}."


def allowed_place_names(state: dict[str, Any]) -> list[str]:
    """Park and trail names the tools actually returned."""

    names: list[str] = []
    for park in (state.get("parks") or {}).get("parks") or []:
        if park.get("name"):
            names.append(str(park["name"]))
        names.extend(str(trail) for trail in (park.get("trails") or []) if trail)
    return names


def _tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def _distinctive(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in _GENERIC and token not in _CONNECTORS]


def _place_candidates(text: str) -> list[str]:
    """Place-like phrases, without the sentence words that sit in front of them.

    The walk stops at a verb or preposition, so ``Hike Bear Lake Trail`` yields
    ``Bear Lake Trail`` and ``Imaginary Bear Lake Trail`` keeps the extra word.
    Lowercase names are included; a title-case requirement missed those.
    """

    names: list[str] = []
    for match in _SUFFIX_WORD.finditer(text or ""):
        words = [match.group(1)]
        cursor = match.start()
        for _ in range(6):
            previous = _PREV_WORD.search(text[:cursor])
            if not previous:
                break
            word = previous.group(1)
            lowered = word.lower()
            if lowered in _CONNECTORS:
                words.append(word)
                cursor = previous.start()
                continue
            if lowered in _WALK_STOP:
                break
            words.append(word)
            cursor = previous.start()
        words.reverse()
        while len(words) > 1 and words[0].lower() in _CONNECTORS:
            words.pop(0)
        name = " ".join(words).strip()
        if name and name not in names:
            names.append(name)
    return names


def invented_place_names(text: str, state: dict[str, Any]) -> list[str]:
    """Park or trail names in ``text`` that no tool returned."""

    allowed = allowed_place_names(state)
    invented: list[str] = []
    for name in _place_candidates(text):
        if not _distinctive(_tokens(name)):
            continue
        if name not in invented and not _name_allowed(name, allowed):
            invented.append(name)
    return invented


def invented_temperatures(text: str, state: dict[str, Any]) -> list[str]:
    """Degree temperatures in ``text`` that are not in the looked-up forecast."""

    compact = re.sub(r"\s+", "", _forecast_summary(state).lower())
    found: list[str] = []
    for match in _TEMPERATURE.finditer(text or ""):
        token = re.sub(r"\s+", "", match.group(0).lower())
        shown = match.group(0).strip()
        if token not in compact and shown not in found:
            found.append(shown)
    return found


def contradicted_risks(text: str, state: dict[str, Any]) -> list[str]:
    """``Risk:`` lines that disagree with the weather report."""

    actual = ((state.get("conditions") or {}).get("risk") or "").strip().lower()
    if actual not in {"ok", "caution", "unsafe"}:
        return []
    found: list[str] = []
    for match in _RISK_CLAIM.finditer(text or ""):
        shown = match.group(0).strip()
        if match.group(1).lower() != actual and shown not in found:
            found.append(shown)
    return found


def _ungrounded(text: str, state: dict[str, Any]) -> list[str]:
    return invented_place_names(text, state) + invented_temperatures(text, state) + contradicted_risks(text, state)


def ground_recommendation(
    text: str,
    state: dict[str, Any],
    complete: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Keep the model text only when it stays on the tool results.

    A missing forecast is copied from the weather report. Invented park or
    trail names get one correction prompt, then those sentences are dropped
    and the options are rendered from the tool results.
    """

    summary = _forecast_summary(state)
    problems = _ungrounded(text, state)
    if not problems:
        return _with_forecast_and_length(text, state, summary), "model"

    corrected = ""
    if complete is not None:
        try:
            corrected = complete(_correction(state, text, problems)).strip()
        except Exception:  # noqa: BLE001 - the tool render is the fallback
            corrected = ""
    if corrected and not _ungrounded(corrected, state):
        return _with_forecast_and_length(corrected, state, summary), "model"

    cleaned = _without_invented_sentences(corrected or text, _ungrounded(corrected or text, state) or problems)
    parts = [part for part in (cleaned, forecast_section(state), _options_block(state)) if part]
    length = (state.get("when") or {}).get("trip_length")
    body = "\n\n".join(parts)
    if length and length.lower() not in body.lower():
        body = f"Trip length: {length}\n\n{body}"
    return body, "grounded"


def _with_forecast_and_length(text: str, state: dict[str, Any], summary: str) -> str:
    body = (text or "").rstrip()
    if summary and summary.lower() not in body.lower():
        body = f"{body}\n\n{forecast_section(state)}".strip()
    length = (state.get("when") or {}).get("trip_length")
    if length and length.lower() not in body.lower():
        body = f"Trip length: {length}\n\n{body}".strip()
    return body


def _forecast_summary(state: dict[str, Any]) -> str:
    return ((state.get("conditions") or {}).get("summary") or "").strip()


def _name_allowed(candidate: str, allowed: list[str]) -> bool:
    """True when every token of ``candidate`` belongs to one looked-up name.

    A longer invented name that merely contains a real trail, such as
    ``Imaginary Bear Lake Trail``, is not allowed. A shorter real form, such as
    ``Bear Lake`` for ``Bear Lake Trail``, is.
    """

    tokens = _tokens(candidate)
    distinctive = _distinctive(tokens)
    if not tokens or not distinctive:
        return True
    for name in allowed:
        other = _tokens(name)
        if not other:
            continue
        if tokens == other:
            return True
        if all(token in other for token in tokens) and any(token in other for token in distinctive):
            return True
    return False


def _without_invented_sentences(text: str, invented: list[str]) -> str:
    if not text or not invented:
        return (text or "").strip()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(name.lower() in lowered for name in invented):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _options_block(state: dict[str, Any]) -> str:
    brief = render_brief(state)
    marker = "\nOptions\n"
    if marker not in f"\n{brief}":
        return ""
    return brief.split("\nOptions\n", 1)[1].strip()


def _correction(state: dict[str, Any], previous: str, problems: list[str]) -> str:
    names = ", ".join(allowed_place_names(state)) or "none"
    summary = _forecast_summary(state) or "Weather was not available."
    risk = (state.get("conditions") or {}).get("risk") or "unknown"
    return (
        f"{render_brief(state)}\n\n"
        "Rewrite the recommendation. The previous draft was rejected because it added "
        f"facts the tools did not return: {', '.join(problems)}.\n"
        f"Use only these parks and trails: {names}.\n"
        f"Include this forecast verbatim: {summary}\n"
        f"The risk is {risk}. Do not state a different risk or a temperature that is not in the forecast.\n\n"
        f"Rejected draft:\n{previous}"
    )
