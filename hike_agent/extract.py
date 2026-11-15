"""Pull a place, a date, and a trip length out of a chat message.

The CLI still accepts a bare place such as ``Boulder, Colorado``. A sentence
such as ``day hike near Boulder, CO this Saturday`` becomes the place
``Boulder, CO``, that Saturday's date, and a day-hike length.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from hike_agent.coordinates import parse_coordinates

_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_LEAD = re.compile(
    r"^(?:please\s+)?(?:(?:find|suggest|recommend|plan)\s+)?(?:a\s+)?(?:hike|hikes|hiking)\s+",
    re.IGNORECASE,
)
_TRIP = re.compile(
    r"\b(?:an?\s+)?(?P<kind>day\s*hikes?|dayhike|day\s+trips?|overnighter|overnight(?:\s+hikes?)?)\b",
    re.IGNORECASE,
)
_WEEKDAY_PHRASE = re.compile(
    r"\b(?P<which>this|next)\s+(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_WEEKEND = re.compile(r"\b(?P<which>this|next)\s+weekend\b", re.IGNORECASE)
_RELATIVE_DAY = re.compile(r"\b(?:tomorrow|today)\b", re.IGNORECASE)
_NAMED_DATE = re.compile(
    r"\b(?:on\s+)?(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b"
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_ON_NAMED = re.compile(
    r"\bon\s+[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    re.IGNORECASE,
)


def extract_hike_request(text: str, *, today: date | None = None) -> dict:
    """Return ``place``, ``when``, and any client ``coordinates`` in the text."""

    today = today or date.today()
    raw = " ".join((text or "").split())
    when = _when(raw, today)
    length = _trip_length(raw)
    if length:
        when = {**when, "trip_length": length}
    coords = parse_coordinates(raw)
    return {
        "place": _place(raw),
        "when": when,
        "coordinates": {"latitude": coords[0], "longitude": coords[1]} if coords else None,
    }


def _trip_length(raw: str) -> str:
    match = _TRIP.search(raw)
    if not match:
        return ""
    kind = re.sub(r"\s+", " ", match.group("kind").lower())
    if kind.startswith("day"):
        return "day hike"
    return "overnight"


def _clean_place(value: str) -> str:
    return value.strip(" ?!.,;:")


def _place(raw: str) -> str:
    trimmed = _strip_scheduling(raw)
    trimmed = _LEAD.sub("", trimmed).strip()
    trimmed = _clean_place(trimmed)
    for prep in ("near", "around"):
        match = re.search(rf"\b{prep}\s+(.+)$", trimmed, re.IGNORECASE)
        if match:
            return _clean_place(match.group(1))
    match = re.search(r"\bin\s+(.+)$", trimmed, re.IGNORECASE)
    if match:
        return _clean_place(match.group(1))
    return trimmed or _clean_place(raw)


def _strip_scheduling(raw: str) -> str:
    text = _TRIP.sub(" ", raw)
    text = _WEEKDAY_PHRASE.sub(" ", text)
    text = _WEEKEND.sub(" ", text)
    text = _RELATIVE_DAY.sub(" ", text)
    text = _ON_NAMED.sub(" ", text)
    text = _ISO_DATE.sub(" ", text)
    text = re.sub(r"[+-]?\d{1,2}\.\d+\s*,\s*[+-]?\d{1,3}\.\d+", " ", text)
    return " ".join(text.split())


def _when(raw: str, today: date) -> dict:
    lower = raw.lower()
    if "next weekend" in lower:
        start, end = _weekend(today, upcoming=True)
        return _span("next weekend", start, end)
    if "this weekend" in lower:
        start, end = _weekend(today, upcoming=False)
        return _span("this weekend", start, end)
    weekday = _WEEKDAY_PHRASE.search(raw)
    if weekday and weekday.group("day").lower() in _WEEKDAYS:
        day = _weekday_date(
            today,
            _WEEKDAYS[weekday.group("day").lower()],
            upcoming=weekday.group("which").lower() == "next",
        )
        label = f"{weekday.group('which').lower()} {weekday.group('day').title()}"
        return _span(label, day, day)
    if re.search(r"\btomorrow\b", lower):
        day = today + timedelta(days=1)
        return _span("tomorrow", day, day)
    if re.search(r"\btoday\b", lower):
        return _span("today", today, today)
    named = _NAMED_DATE.search(raw)
    if named and named.group("month").lower() in _MONTHS:
        month = _MONTHS[named.group("month").lower()]
        day_number = int(named.group("day"))
        year = int(named.group("year")) if named.group("year") else today.year
        try:
            found = date(year, month, day_number)
        except ValueError:
            return {}
        if named.group("year") is None and found < today:
            try:
                found = date(year + 1, month, day_number)
            except ValueError:
                return {}
        return _span(found.strftime("%B") + f" {found.day}", found, found)
    iso = _ISO_DATE.search(raw)
    if iso:
        found = date.fromisoformat(iso.group(1))
        return _span(found.isoformat(), found, found)
    return {}


def _weekday_date(today: date, weekday_index: int, *, upcoming: bool) -> date:
    """``this <weekday>`` is the next one, including today. ``next`` is seven days after that."""

    delta = (weekday_index - today.weekday()) % 7
    if upcoming:
        delta = 7 if delta == 0 else delta + 7
    return today + timedelta(days=delta)


def _weekend(today: date, *, upcoming: bool) -> tuple[date, date]:
    """Saturday-Sunday for this weekend, or the one after it."""

    if today.weekday() == 6:
        saturday = today - timedelta(days=1)
    else:
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    if upcoming:
        saturday = saturday + timedelta(days=7)
    return saturday, saturday + timedelta(days=1)


def _span(name: str, start: date, end: date) -> dict:
    if start == end:
        label = f"{name} ({start.isoformat()})"
    else:
        label = f"{name} ({start.isoformat()} to {end.isoformat()})"
    return {"label": label, "start": start.isoformat(), "end": end.isoformat()}
