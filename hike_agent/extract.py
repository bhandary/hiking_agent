"""Pull a place, a date, and a trip length out of a chat message.

The CLI still accepts a bare place such as ``Boulder, Colorado``. A sentence
such as ``day hike near Boulder, CO this Saturday`` becomes the place
``Boulder, CO``, that Saturday's date, and a day-hike length. A sentence that
is only a verb and a date (``hike tomorrow``) has no place: the leftover verb
is not geocoded.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from hike_agent.coordinates import parse_coordinates, strip_coordinates

INVALID_DATE = "That date isn't a real day. Try a date that exists."

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
_WEEKDAY_WORDS = "|".join(_WEEKDAYS)
_WEEKDAY_SHORT = {
    "mon": 0,
    "tue": 1,
    "tues": 1,
    "wed": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
_WEEKDAY_ANY = "|".join([*_WEEKDAYS, *_WEEKDAY_SHORT])

# Words that are never a place by themselves. Geocoding ``hike`` or ``Saturday``
# returns a town in another country.
NOT_A_PLACE = frozenset(
    {
        "a",
        "an",
        "at",
        "around",
        "day",
        "days",
        "dayhike",
        "find",
        "for",
        "friday",
        "go",
        "hike",
        "hikes",
        "hiking",
        "in",
        "monday",
        "near",
        "on",
        "overnight",
        "overnighter",
        "path",
        "paths",
        "plan",
        "please",
        "recommend",
        "route",
        "routes",
        "saturday",
        "suggest",
        "sunday",
        "the",
        "thursday",
        "to",
        "trail",
        "trails",
        "trip",
        "trips",
        "tuesday",
        "walk",
        "walking",
        "walks",
        "wednesday",
        "weekend",
    }
)

_LEAD = re.compile(
    r"^(?:please\s+)?(?:(?:find|suggest|recommend|plan)\s+)?(?:a\s+)?(?:hike|hikes|hiking)\s+",
    re.IGNORECASE,
)
_TRIP = re.compile(
    r"\b(?:an?\s+)?(?P<kind>day\s*hikes?|dayhike|day\s+trips?|overnighter|overnight(?:\s+hikes?)?)\b",
    re.IGNORECASE,
)
_WEEKDAY_PHRASE = re.compile(
    rf"\b(?P<which>this|next)\s+(?P<day>{_WEEKDAY_ANY})\b",
    re.IGNORECASE,
)
_BARE_WEEKDAY = re.compile(rf"\b(?P<day>{_WEEKDAY_WORDS})\b", re.IGNORECASE)
_WEEKEND = re.compile(r"\b(?P<which>this|next)\s+weekend\b", re.IGNORECASE)
_RELATIVE_DAY = re.compile(r"\b(?:tomorrow|today)\b", re.IGNORECASE)
_NAMED_DATE = re.compile(
    r"\b(?:on\s+)?(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b"
)
_DAY_MONTH = re.compile(
    r"\b(?:on\s+)?(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>[A-Za-z]{3,9})(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_NUMERIC_DATE = re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\b")
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def is_activity_phrase(text: str) -> bool:
    """True when ``text`` is only hike words, filler, or a weekday name."""

    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    return bool(words) and all(word in NOT_A_PLACE for word in words)


def extract_hike_request(text: str, *, today: date | None = None) -> dict:
    """Return ``place``, ``when``, and any client ``coordinates`` in the text."""

    today = today or date.today()
    raw = " ".join((text or "").split())
    when = _when(raw, today)
    length = _trip_length(raw)
    if length and not when.get("invalid"):
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


def _place(raw: str) -> str:
    trimmed = _strip_scheduling(raw)
    trimmed = _LEAD.sub("", trimmed).strip()
    trimmed = trimmed.strip(" ?.,")
    place = ""
    for prep in ("near", "around"):
        match = re.search(rf"\b{prep}\s+(.+)$", trimmed, re.IGNORECASE)
        if match:
            place = match.group(1).strip(" ?.,")
            break
    if not place:
        match = re.search(r"\bin\s+(.+)$", trimmed, re.IGNORECASE)
        if match:
            place = match.group(1).strip(" ?.,")
    if not place:
        place = trimmed
    if is_activity_phrase(place) or not place.strip(" ?.,"):
        return ""
    return place


def _strip_scheduling(raw: str) -> str:
    text = _TRIP.sub(" ", raw)
    text = _WEEKDAY_PHRASE.sub(" ", text)
    text = _WEEKEND.sub(" ", text)
    text = _RELATIVE_DAY.sub(" ", text)
    text = _NAMED_DATE.sub(" ", text)
    text = _DAY_MONTH.sub(" ", text)
    text = _NUMERIC_DATE.sub(" ", text)
    text = _ISO_DATE.sub(" ", text)
    text = _BARE_WEEKDAY.sub(" ", text)
    text = strip_coordinates(text)
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
    if weekday:
        index = _weekday_index(weekday.group("day"))
        if index is not None:
            day = _weekday_date(today, index, upcoming=weekday.group("which").lower() == "next")
            label = f"{weekday.group('which').lower()} {_weekday_name(index)}"
            return _span(label, day, day)
    if re.search(r"\btomorrow\b", lower):
        day = today + timedelta(days=1)
        return _span("tomorrow", day, day)
    if re.search(r"\btoday\b", lower):
        return _span("today", today, today)

    explicit = _explicit_date(raw, today)
    if explicit is not None:
        return explicit

    bare = _BARE_WEEKDAY.search(raw)
    if bare:
        index = _WEEKDAYS[bare.group("day").lower()]
        day = _weekday_date(today, index, upcoming=False)
        return _span(_weekday_name(index), day, day)
    return {}


def _explicit_date(raw: str, today: date) -> dict | None:
    """A calendar date written in the message, or an invalid-date marker."""

    found: list[tuple[int, str, re.Match[str]]] = []
    iso = _ISO_DATE.search(raw)
    if iso:
        found.append((iso.start(), "iso", iso))
    numeric = _NUMERIC_DATE.search(raw)
    if numeric:
        found.append((numeric.start(), "numeric", numeric))
    named = _NAMED_DATE.search(raw)
    if named and named.group("month").lower() in _MONTHS:
        found.append((named.start(), "named", named))
    day_month = _DAY_MONTH.search(raw)
    if day_month and day_month.group("month").lower() in _MONTHS:
        found.append((day_month.start(), "day-month", day_month))
    if not found:
        return None
    _start, kind, match = min(found, key=lambda item: item[0])
    if kind == "iso":
        try:
            parsed = date.fromisoformat(match.group(1))
        except ValueError:
            return {"invalid": match.group(1)}
        return _span(parsed.isoformat(), parsed, parsed)
    if kind == "numeric":
        return _calendar_date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            match.group(0),
            today,
            year_was_written=True,
        )
    year_text = match.group("year")
    year = int(year_text) if year_text else today.year
    return _calendar_date(
        year,
        _MONTHS[match.group("month").lower()],
        int(match.group("day")),
        match.group(0),
        today,
        year_was_written=year_text is not None,
    )


def _calendar_date(
    year: int,
    month: int,
    day_number: int,
    phrase: str,
    today: date,
    *,
    year_was_written: bool,
) -> dict:
    try:
        found = date(year, month, day_number)
    except ValueError:
        return {"invalid": phrase.strip()}
    if not year_was_written and found < today:
        try:
            found = date(year + 1, month, day_number)
        except ValueError:
            return {"invalid": phrase.strip()}
    return _span(found.strftime("%B") + f" {found.day}", found, found)


def _weekday_index(word: str) -> int | None:
    lowered = word.lower()
    if lowered in _WEEKDAYS:
        return _WEEKDAYS[lowered]
    return _WEEKDAY_SHORT.get(lowered)


def _weekday_name(index: int) -> str:
    return next(name.title() for name, value in _WEEKDAYS.items() if value == index)


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
