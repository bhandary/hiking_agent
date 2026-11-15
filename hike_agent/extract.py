"""Pull a place and a date out of a chat message.

The CLI still accepts a bare place such as ``Boulder, Colorado``. A sentence
such as ``hikes near Boulder this weekend`` is split so the location agent
sees the place and the planner sees the date.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})

_LEAD = re.compile(
    r"^(?:please\s+)?(?:(?:find|suggest|recommend|plan)\s+)?(?:a\s+)?(?:hike|hikes|hiking)\s+",
    re.IGNORECASE,
)
_LEADING_WHEN = re.compile(
    r"^(?:tomorrow|today|(?:this|next)\s+weekend)\b[:,]?\s*",
    re.IGNORECASE,
)
_TRAILING_WHEN = re.compile(
    r"(?:\s+(?:this|next)\s+weekend|\s+tomorrow|\s+today|\s+on\s+\w[\w\s,.-]*|\s+for\s+(?:this|next)\s+weekend)$",
    re.IGNORECASE,
)
_NAMED_DATE = re.compile(
    r"\b(?:on\s+)?(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b"
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_hike_request(text: str, *, today: date | None = None) -> dict:
    """Return ``place`` and an optional ``when`` dict with label, start, and end."""

    today = today or date.today()
    raw = " ".join((text or "").split())
    return {
        "place": _place(raw),
        "when": _when(raw, today),
    }


def _place(raw: str) -> str:
    trimmed = _LEAD.sub("", raw).strip()
    trimmed = _LEADING_WHEN.sub("", trimmed).strip()
    trimmed = _TRAILING_WHEN.sub("", trimmed).strip(" ?.,")
    for prep in ("near", "around"):
        match = re.search(rf"\b{prep}\s+(.+)$", trimmed, re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?.,")
    match = re.search(r"\bin\s+(.+)$", trimmed, re.IGNORECASE)
    if match:
        return match.group(1).strip(" ?.,")
    return trimmed or raw.strip(" ?.")


def _when(raw: str, today: date) -> dict:
    lower = raw.lower()
    if "next weekend" in lower:
        start, end = _weekend(today, upcoming=True)
        return _span("next weekend", start, end)
    if "this weekend" in lower:
        start, end = _weekend(today, upcoming=False)
        return _span("this weekend", start, end)
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
