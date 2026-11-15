"""Client-supplied coordinates.

The hike server never looks up its own IP address. A caller can pass
``{"latitude": ..., "longitude": ...}`` or include a decimal pair in the text.
"""

from __future__ import annotations

import re
from typing import Any

_PAIR = re.compile(r"(?P<lat>[+-]?\d{1,2}\.\d+)\s*,\s*(?P<lon>[+-]?\d{1,3}\.\d+)")


def parse_coordinates(value: Any) -> tuple[float, float] | None:
    """Return ``(latitude, longitude)`` or ``None`` when ``value`` has no pair."""

    if isinstance(value, dict):
        return _valid(value.get("latitude", value.get("lat")), value.get("longitude", value.get("lon", value.get("lng"))))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _valid(value[0], value[1])
    if isinstance(value, str):
        match = _PAIR.search(value)
        if match:
            return _valid(match.group("lat"), match.group("lon"))
    return None


def _valid(latitude: Any, longitude: Any) -> tuple[float, float] | None:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon
