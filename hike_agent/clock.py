"""The hiker's calendar day.

Relative dates and the word "today" follow the timezone the request named.
When the request does not name one, the default is America/Los_Angeles.
The server's own clock is not used.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "America/Los_Angeles"
_FALLBACK = timezone(timedelta(hours=-8))


def resolve_timezone(name: str | None) -> tuple[datetime.tzinfo, str]:
    """Return a timezone and the name that was actually used."""

    raw = (name or "").strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(raw), raw
    except Exception:
        try:
            return ZoneInfo(DEFAULT_TIMEZONE), DEFAULT_TIMEZONE
        except Exception:
            return _FALLBACK, DEFAULT_TIMEZONE


def today_in(name: str | None = None) -> date:
    """Today's date in the hiker's timezone."""

    zone, _used = resolve_timezone(name)
    return datetime.now(zone).date()
