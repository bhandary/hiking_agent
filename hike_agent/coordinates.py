"""Client-supplied coordinates.

The hike server never looks up its own IP address. A caller can pass
``{"latitude": ..., "longitude": ...}`` or include a decimal pair in the text.
"""

from __future__ import annotations

import re
from typing import Any

# Both numbers may be 1–3 digits. A lookaround keeps a 3-digit longitude from
# matching on its last two digits (``-105.271`` must not become ``5.271``).
_PAIR = re.compile(
    r"(?<![\d.])(?P<lat>[+-]?\d{1,3}\.\d+)\s*,\s*(?P<lon>[+-]?\d{1,3}\.\d+)(?![\d.])"
)


def strip_coordinates(text: str) -> str:
    """Remove decimal pairs so they are not treated as part of a place name."""

    return _PAIR.sub(" ", text or "")


def parse_coordinates(value: Any) -> tuple[float, float] | None:
    """Return ``(latitude, longitude)`` or ``None`` when ``value`` has no pair.

    Longitude-first text such as ``-105.271, 40.015`` is rejected. Slicing it
    into a different in-range point would send the hiker to the wrong place.
    """

    if isinstance(value, dict):
        return _valid(value.get("latitude", value.get("lat")), value.get("longitude", value.get("lon", value.get("lng"))))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _valid(value[0], value[1])
    if isinstance(value, str):
        for match in _PAIR.finditer(value):
            found = _valid(match.group("lat"), match.group("lon"))
            if found:
                return found
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
