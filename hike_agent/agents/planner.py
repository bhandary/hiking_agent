"""Planner agent: write the hike recommendation from the other agents' facts.

The graph calls the configured chat model with this brief. When the model is
unreachable, the brief itself is the recommendation, so safety notes still appear.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Callable

from hike_agent.agents.parks import NOT_VERIFIED, NPS_SOURCES
from hike_agent.clock import today_in

AGENT_NAME = "planner"
AGENT_BRIEF = "Combine location, parks, and conditions into a short hike recommendation with safety notes."

PLANNER_SYSTEM = """You are a hiking guide. Rewrite the factual brief into a short recommendation.
Rules:
- Recommend at most three options that appear in the brief. Do not invent parks, trails, temperatures, or alerts.
- If the risk is unsafe, say not to hike on the requested date and offer the options only as later ideas. When that date is not today, name the date. Do not say to hike today.
- Keep every safety note. A missing alert is not an all-clear.
- When a place is marked "not verified by NPS", repeat that phrase. Do not present those names as National Park Service data.
- Stay under 250 words.
- Include the forecast summary from the brief word for word.
"""

_PLACE_SUFFIXES = (
    "national park",
    "state park",
    "national forest",
    "national monument",
    "wilderness area",
    "waterfall",
    "wilderness",
    "monument",
    "mountain",
    "mountains",
    "canyon",
    "summit",
    "meadow",
    "overlook",
    "glacier",
    "valley",
    "basin",
    "ridge",
    "creek",
    "falls",
    "trail",
    "trails",
    "route",
    "forest",
    "butte",
    "gulch",
    "point",
    "mesa",
    "peak",
    "lake",
    "pond",
    "loop",
    "path",
    "pass",
    "river",
)
_PLACE_SUFFIX = re.compile(
    r"\b(?:" + "|".join(re.escape(suffix) for suffix in _PLACE_SUFFIXES) + r")\b",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "around",
        "at",
        "be",
        "best",
        "but",
        "cold",
        "dawn",
        "day",
        "for",
        "from",
        "go",
        "hike",
        "hikes",
        "hiking",
        "hot",
        "in",
        "is",
        "it",
        "near",
        "of",
        "on",
        "or",
        "see",
        "sunrise",
        "sunset",
        "that",
        "the",
        "this",
        "to",
        "today",
        "tomorrow",
        "try",
        "via",
        "visit",
        "was",
        "will",
        "with",
    }
)
_TEMPERATURE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([FC])\b", re.IGNORECASE)
_MEASURE = re.compile(
    r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(miles?|mi|kilometers?|km|feet|foot|ft|meters?|m)\b",
    re.IGNORECASE,
)
_MONTH_DAY = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_DAY_MONTH = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>january|february|march|april|may|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
_ALL_CLEAR = re.compile(
    r"\b(?:no alerts?(?:\s+are\s+in\s+effect)?|all[- ]clear|it is safe|safe to hike|no closures|no hazards)\b",
    re.IGNORECASE,
)
_RISK_CLAIM = re.compile(r"\bRisk:\s*(ok|caution|unsafe|unknown)\b", re.IGNORECASE)
_TODAY_HIKE = re.compile(
    r"\b(?:not recommended to hike today|do not hike today|don't hike today|not to hike today|"
    r"today is a poor day to hike|hike today|hiking today)\b",
    re.IGNORECASE,
)
_MONTH_INDEX = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_UNIT_FAMILY = {
    "mile": "mi", "miles": "mi", "mi": "mi",
    "kilometer": "km", "kilometers": "km", "km": "km",
    "foot": "ft", "feet": "ft", "ft": "ft",
    "meter": "m", "meters": "m", "m": "m",
}


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
    advice = _unsafe_day_sentence(state)
    if advice:
        lines.append(advice)
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
    for candidate in _place_candidates(_tool_fact_text(state)):
        if candidate not in names:
            names.append(candidate)
    return names


def invented_place_names(text: str, state: dict[str, Any]) -> list[str]:
    """Park or trail names in ``text`` that no tool returned.

    Matching is case-insensitive and is not limited to Title Case. A name is
    allowed only when it is a tool name, not when one string merely contains
    the other. ``Lake Trail`` is not ``Bear Lake Trail``.
    """

    allowed = allowed_place_names(state)
    invented: list[str] = []
    for name in _place_candidates(text or ""):
        if name not in invented and not _name_allowed(name, allowed):
            invented.append(name)
    return invented


def ungrounded_claims(text: str, state: dict[str, Any]) -> list[str]:
    """Place names, weather, dates, distances, and all-clears the tools do not support."""

    claims = [f"place not in tool results: {name}" for name in invented_place_names(text, state)]
    claims.extend(_forecast_conflicts(text, state))
    return claims


def ground_recommendation(
    text: str,
    state: dict[str, Any],
    complete: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Keep the model text only when it stays on the tool results.

    A missing forecast is copied from the weather report. Invented park or
    trail names, and temperatures, dates, distances, or all-clears the tools
    did not return, get one correction prompt. If that draft still conflicts,
    those sentences are dropped and the options are rendered from the tools.
    """

    summary = ((state.get("conditions") or {}).get("summary") or "").strip()
    problems = ungrounded_claims(text, state)
    if not problems:
        return _with_forecast_and_length(text, state, summary), "model"

    corrected = ""
    if complete is not None:
        try:
            corrected = complete(_correction(state, text, problems)).strip()
        except Exception:  # noqa: BLE001 - the tool render is the fallback
            corrected = ""
    if corrected and not ungrounded_claims(corrected, state):
        return _with_forecast_and_length(corrected, state, summary), "model"

    cleaned = _without_problem_sentences(corrected or text, state)
    parts = [part for part in (cleaned, forecast_section(state), _options_block(state)) if part]
    length = (state.get("when") or {}).get("trip_length")
    body = "\n\n".join(parts)
    if length and length.lower() not in body.lower():
        body = f"Trip length: {length}\n\n{body}"
    body = _ensure_date_risk_language(body, state)
    return body, "grounded"


def _with_forecast_and_length(text: str, state: dict[str, Any], summary: str) -> str:
    body = (text or "").rstrip()
    if summary and summary.lower() not in body.lower():
        body = f"{body}\n\n{forecast_section(state)}".strip()
    length = (state.get("when") or {}).get("trip_length")
    if length and length.lower() not in body.lower():
        body = f"Trip length: {length}\n\n{body}".strip()
    return body


def _name_allowed(candidate: str, allowed: list[str]) -> bool:
    """True only when the normalized names are the same string.

    Case, punctuation, and extra whitespace do not count. A longer phrase that
    merely contains a real name is a different place: ``Imaginary Bear Lake
    Trail`` is not ``Bear Lake Trail``.
    """

    cand = _norm_name(candidate)
    if not cand:
        return False
    return any(_norm_name(name) == cand for name in allowed)


def _without_problem_sentences(text: str, state: dict[str, Any]) -> str:
    if not text:
        return ""
    actual = str((state.get("conditions") or {}).get("risk") or "").lower()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for sentence in sentences:
        if not sentence:
            continue
        claimed = _RISK_CLAIM.search(sentence)
        if claimed and claimed.group(1).lower() != actual:
            continue
        if _today_hike_conflicts(sentence, state):
            continue
        if ungrounded_claims(sentence, state):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _place_candidates(text: str) -> list[str]:
    """Place-like phrases, including lowercase ones, longest span at each suffix."""

    found: list[str] = []
    for match in _PLACE_SUFFIX.finditer(text or ""):
        name = _grow_place(text, match.start(), match.group(0))
        if name and name not in found:
            found.append(name)
    longest: list[str] = []
    for name in found:
        if any(name != other and _phrase_in(name, other) for other in found):
            continue
        longest.append(name)
    return longest


def _grow_place(text: str, suffix_start: int, suffix: str) -> str:
    words: list[str] = []
    index = suffix_start
    while len(words) < 6:
        cursor = index
        while cursor > 0 and text[cursor - 1] in " \t":
            cursor -= 1
        if cursor == index or cursor == 0:
            break
        word_start = cursor
        while word_start > 0 and re.match(r"[A-Za-z0-9'’.-]", text[word_start - 1]):
            word_start -= 1
        if word_start == cursor:
            break
        word = text[word_start:cursor]
        if not re.match(r"[A-Za-z]", word) or word.lower() in _STOP_WORDS:
            break
        words.append(word)
        index = word_start
    if not words:
        return ""
    return " ".join(reversed(words)) + " " + suffix


def _forecast_conflicts(text: str, state: dict[str, Any]) -> list[str]:
    facts = _tool_fact_text(state)
    summary = ((state.get("conditions") or {}).get("summary") or "").strip()
    conflicts: list[str] = []
    for claim in _temperature_conflicts(text, summary, facts):
        conflicts.append(claim)
    for claim in _date_conflicts(text, state, facts):
        conflicts.append(claim)
    for claim in _measure_conflicts(text, facts):
        conflicts.append(claim)
    for claim in _all_clear_conflicts(text, facts):
        conflicts.append(claim)
    for claim in _risk_conflicts(text, state):
        conflicts.append(claim)
    for claim in _today_hike_conflicts(text, state):
        conflicts.append(claim)
    return conflicts


def _tool_fact_text(state: dict[str, Any]) -> str:
    conditions = state.get("conditions") or {}
    parts = [conditions.get("summary") or "", *(conditions.get("safety_notes") or []), *(conditions.get("alerts") or [])]
    for park in (state.get("parks") or {}).get("parks") or []:
        parts.extend(
            [
                park.get("name") or "",
                park.get("description") or "",
                *(park.get("trails") or []),
                *(park.get("alerts") or []),
            ]
        )
    when = state.get("when") or {}
    if isinstance(when, dict) and when.get("label"):
        parts.append(str(when["label"]))
    return "\n".join(str(part) for part in parts if part)


def _temperature_conflicts(text: str, summary: str, facts: str) -> list[str]:
    claimed = _temperatures_c(text)
    if not claimed:
        return []
    allowed = [celsius for _original, celsius in _temperatures_c(summary)]
    conflicts = []
    for original, celsius in claimed:
        if _verbatim_in_tools(original, facts):
            continue
        if any(abs(celsius - known) <= 1.5 for known in allowed):
            continue
        conflicts.append(f"temperature not in the forecast: {original}")
    return conflicts


def _temperatures_c(text: str) -> list[tuple[str, float]]:
    found = []
    for match in _TEMPERATURE.finditer(text or ""):
        value = float(match.group(1))
        unit = match.group(2).upper()
        celsius = value if unit == "C" else (value - 32) * 5 / 9
        found.append((match.group(0), celsius))
    return found


def _date_conflicts(text: str, state: dict[str, Any], facts: str) -> list[str]:
    window = _requested_window(state)
    if window is None:
        return []
    start, end = window
    allowed_days = set()
    cursor = start
    while cursor <= end:
        allowed_days.add(cursor.weekday())
        cursor += timedelta(days=1)
    conflicts = []
    for spoken, found in _mentioned_dates(text, start.year):
        if _verbatim_in_tools(spoken, facts):
            continue
        if found < start or found > end:
            conflicts.append(f"date outside the requested trip: {spoken}")
    for match in _WEEKDAY.finditer(text or ""):
        if _verbatim_in_tools(match.group(1), facts):
            continue
        if _WEEKDAY_INDEX[match.group(1).lower()] not in allowed_days:
            conflicts.append(f"weekday outside the requested trip: {match.group(1)}")
    return conflicts


def _requested_window(state: dict[str, Any]) -> tuple[date, date] | None:
    when = state.get("when") or {}
    if not isinstance(when, dict) or not when.get("start"):
        return None
    try:
        start = date.fromisoformat(str(when["start"]))
        end = date.fromisoformat(str(when.get("end") or when["start"]))
    except ValueError:
        return None
    if end < start:
        end = start
    return start, end


def _mentioned_dates(text: str, default_year: int) -> list[tuple[str, date]]:
    found: list[tuple[str, date]] = []
    for match in _ISO_DATE.finditer(text or ""):
        try:
            found.append((match.group(1), date.fromisoformat(match.group(1))))
        except ValueError:
            continue
    for pattern in (_MONTH_DAY, _DAY_MONTH):
        for match in pattern.finditer(text or ""):
            month = _MONTH_INDEX[match.group("month").lower()]
            day_number = int(match.group("day"))
            year = int(match.group("year")) if match.group("year") else default_year
            try:
                found.append((match.group(0), date(year, month, day_number)))
            except ValueError:
                continue
    return found


def _measure_conflicts(text: str, facts: str) -> list[str]:
    allowed = set(_measures(facts))
    conflicts = []
    for spoken, measure in _measures(text):
        if _verbatim_in_tools(spoken, facts):
            continue
        if measure not in allowed:
            conflicts.append(f"distance or elevation not in the tool results: {spoken}")
    return conflicts


def _verbatim_in_tools(spoken: str, facts: str) -> bool:
    """True when the number or date is copied from tool text, not sliced from a longer one.

    ``9 F`` is not ``49 F``, ``November 2`` is not ``November 21``, and ``2 miles``
    is not ``12 miles``. A decimal point is part of the number, so ``4.9`` is not
    ``49`` and not the trailing ``9``.
    """

    needle = " ".join(spoken.lower().split())
    haystack = " ".join((facts or "").lower().split())
    if not needle:
        return False
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return False
        if _number_is_bounded(haystack, index, index + len(needle)):
            return True
        start = index + 1


def _number_is_bounded(text: str, start: int, end: int) -> bool:
    """No digit, or a decimal that continues a number, touches this match."""

    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before.isdigit() or after.isdigit():
        return False
    starts_digit = start < len(text) and text[start].isdigit()
    ends_digit = end > 0 and text[end - 1].isdigit()
    if before == "." and starts_digit:
        return False
    if before == "," and starts_digit and start > 1 and text[start - 2].isdigit():
        return False
    if after == "." and ends_digit and end + 1 < len(text) and text[end + 1].isdigit():
        return False
    if after == "," and ends_digit and end + 1 < len(text) and text[end + 1].isdigit():
        return False
    return True


def _risk_conflicts(text: str, state: dict[str, Any]) -> list[str]:
    actual = str((state.get("conditions") or {}).get("risk") or "").lower()
    conflicts = []
    for match in _RISK_CLAIM.finditer(text or ""):
        if match.group(1).lower() != actual:
            conflicts.append(f"risk does not match conditions: {match.group(0)}")
    return conflicts


def _requested_day_phrase(state: dict[str, Any]) -> str:
    when = state.get("when") or {}
    if not isinstance(when, dict) or not when.get("start"):
        return "today"
    try:
        requested = date.fromisoformat(str(when["start"]))
    except ValueError:
        return "today"
    if requested == today_in(state.get("timezone")):
        return "today"
    return str(when.get("label") or requested.isoformat())


def _unsafe_day_sentence(state: dict[str, Any]) -> str:
    if (state.get("conditions") or {}).get("risk") != "unsafe":
        return ""
    phrase = _requested_day_phrase(state)
    if phrase == "today":
        return "Today is a poor day to hike. Treat the options below as later-trip ideas, not a green light to go out."
    return f"{phrase} is a poor day to hike. Treat the options below as later-trip ideas, not a green light to go out."


def _today_hike_conflicts(text: str, state: dict[str, Any]) -> list[str]:
    if _requested_day_phrase(state) == "today":
        return []
    if _TODAY_HIKE.search(text or ""):
        return ["risk language says today for a different date"]
    return []


def _ensure_date_risk_language(body: str, state: dict[str, Any]) -> str:
    sentence = _unsafe_day_sentence(state)
    if not sentence:
        return body
    if _requested_day_phrase(state) != "today" and _TODAY_HIKE.search(body):
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        body = " ".join(part for part in sentences if part and not _TODAY_HIKE.search(part)).strip()
    if "poor day to hike" in body.lower():
        return body
    return f"{body}\n\n{sentence}".strip()


def _measures(text: str) -> list[tuple[str, tuple[float, str]]]:
    found = []
    for match in _MEASURE.finditer(text or ""):
        number = float(match.group(1).replace(",", ""))
        family = _UNIT_FAMILY[match.group(2).lower()]
        found.append((match.group(0), (number, family)))
    return found


def _all_clear_conflicts(text: str, facts: str) -> list[str]:
    conflicts = []
    fact_text = facts.lower()
    for match in _ALL_CLEAR.finditer(text or ""):
        phrase = match.group(0).lower()
        if phrase not in fact_text:
            conflicts.append(f"all-clear the tools did not state: {match.group(0)}")
    return conflicts


def _norm_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (name or "").lower())
    return " ".join(cleaned.split())


def _phrase_in(short: str, long: str) -> bool:
    if not short or not long:
        return False
    return re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long) is not None


def _options_block(state: dict[str, Any]) -> str:
    brief = render_brief(state)
    marker = "\nOptions\n"
    if marker not in f"\n{brief}":
        return ""
    return brief.split("\nOptions\n", 1)[1].strip()


def _correction(state: dict[str, Any], previous: str, problems: list[str]) -> str:
    names = ", ".join(allowed_place_names(state)) or "none"
    summary = ((state.get("conditions") or {}).get("summary") or "Weather was not available.").strip()
    reasons = "\n".join(f"- {problem}" for problem in problems)
    return (
        f"{render_brief(state)}\n\n"
        "Rewrite the recommendation. The previous draft was rejected:\n"
        f"{reasons}\n"
        f"Use only these parks and trails: {names}.\n"
        "Do not add a temperature, distance, date, or all-clear that the brief does not state. "
        "A missing alert is not an all-clear.\n"
        f"Include this forecast verbatim: {summary}\n\n"
        f"Rejected draft:\n{previous}"
    )
