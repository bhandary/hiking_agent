"""Location agent: turn a place name or client coordinates into a map point.

``me``, ``here``, and ``near me`` do not use this machine's IP address. On a
hosted server that address is the host, not the hiker. Those queries need a
city or park name, or coordinates the client sent.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.tools import tool

from hike_agent.config import Settings, get_settings
from hike_agent.coordinates import parse_coordinates, strip_coordinates
from hike_agent.extract import is_activity_phrase
from hike_agent.services.http import get_json as default_get_json
from hike_agent.services.states import state_code

AGENT_NAME = "location"
AGENT_BRIEF = "Resolve a place name, or client-provided coordinates, to a map point and a US state code."

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NEAR_ME = {
    "",
    "me",
    "here",
    "current",
    "current location",
    "near me",
    "around me",
    "nearby",
    "close to me",
    "my location",
}
NEED_LOCATION = "Tell me a city or park name. I don't use this server's location."
COULD_NOT_FIND = "I couldn't find that place, try a city/park name."
GetJson = Callable[..., Any]
_NEAR_WORDS = re.compile(r"\b(?:at|near|around|nearby|close\s+to)\b", re.IGNORECASE)


def _from_geocode_hit(query: str, hit: dict[str, Any]) -> dict[str, Any]:
    admin1 = hit.get("admin1") or ""
    country = hit.get("country") or ""
    name = hit.get("name") or query
    label_parts = [part for part in (name, admin1, country) if part]
    return {
        "query": query,
        "name": name,
        "label": ", ".join(label_parts),
        "latitude": float(hit["latitude"]),
        "longitude": float(hit["longitude"]),
        "state": admin1 or None,
        "state_code": state_code(admin1),
        "country": country or None,
        "source": "open-meteo",
        "note": "",
    }


def _from_coordinates(
    query: str,
    coords: tuple[float, float],
    *,
    settings: Settings,
    get_json: GetJson,
) -> dict[str, Any]:
    """Use the client's point, and reverse-geocode it so park search has a real name."""

    latitude, longitude = coords
    place = {
        "query": query or "me",
        "name": None,
        "label": f"{latitude:.4f}, {longitude:.4f}",
        "latitude": latitude,
        "longitude": longitude,
        "state": None,
        "state_code": None,
        "country": None,
        "source": "client",
        "note": "",
    }
    try:
        payload = get_json(
            REVERSE_URL,
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "addressdetails": 1,
                "zoom": 12,
            },
            timeout=settings.http_timeout,
            user_agent=settings.user_agent,
        )
    except Exception:  # noqa: BLE001 - coordinates still answer weather; park search skips a fake name
        return place
    address = payload.get("address") if isinstance(payload, dict) else None
    if not isinstance(address, dict):
        return place
    city = ""
    for key in ("city", "town", "village", "hamlet", "municipality"):
        if address.get(key):
            city = str(address[key])
            break
    state = address.get("state") or ""
    country = address.get("country") or ""
    if not city:
        return place
    place["name"] = city
    place["state"] = state or None
    place["state_code"] = state_code(state)
    place["country"] = country or None
    place["label"] = ", ".join(part for part in (city, state, country) if part)
    return place


def _empty(query: str, note: str, source: str = "none") -> dict[str, Any]:
    return {
        "query": query,
        "name": None,
        "label": query,
        "latitude": None,
        "longitude": None,
        "state": None,
        "state_code": None,
        "country": None,
        "source": source,
        "note": note,
    }


def _is_near_me(text: str) -> bool:
    lowered = " ".join(re.sub(r"[?!.,;:]+", " ", text.lower()).split())
    if lowered in NEAR_ME:
        return True
    without_coords = strip_coordinates(lowered)
    without_coords = re.sub(
        r"\b(?:day\s*hikes?|dayhike|overnight(?:\s+hikes?)?|hikes?|hiking)\b",
        " ",
        without_coords,
    )
    without_coords = _NEAR_WORDS.sub(" ", without_coords)
    without_coords = " ".join(without_coords.split())
    return without_coords in NEAR_ME


def resolve_place(
    query: str,
    *,
    settings: Settings | None = None,
    get_json: GetJson | None = None,
    coordinates: Any = None,
) -> dict[str, Any]:
    """Resolve ``query`` to a place dict. Network helpers are injectable for tests."""

    settings = settings or get_settings()
    get_json = get_json or default_get_json
    text = (query or "").strip()
    coords = parse_coordinates(coordinates) or parse_coordinates(text)
    if _is_near_me(text) or (not text and coords):
        if coords:
            return _from_coordinates(text or "me", coords, settings=settings, get_json=get_json)
        return _empty(text or "me", NEED_LOCATION, source="needs-location")
    if not text or is_activity_phrase(text):
        return _empty(text, NEED_LOCATION, source="needs-location")

    try:
        payload = get_json(
            GEOCODE_URL,
            params={"name": text, "count": 5, "language": "en", "format": "json"},
            timeout=settings.http_timeout,
            user_agent=settings.user_agent,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty(text, f"Geocoding failed ({type(exc).__name__}).", source="error")
    results = payload.get("results") if isinstance(payload, dict) else None
    for hit in results or []:
        if hit.get("latitude") is None or hit.get("longitude") is None:
            continue
        return _from_geocode_hit(text, hit)
    return _empty(text, f"No coordinates found for {text!r}.")


@tool
def resolve_place_tool(query: str) -> str:
    """Resolve a place name to coordinates. 'me' needs a city name or client coordinates."""

    import json

    return json.dumps(resolve_place(query))
