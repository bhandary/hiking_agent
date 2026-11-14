"""Location agent: turn a place name or 'me' into coordinates and a state code."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import tool

from hike_agent.config import Settings, get_settings
from hike_agent.services.http import get_json as default_get_json
from hike_agent.services.states import state_code

AGENT_NAME = "location"
AGENT_BRIEF = "Resolve a place name, or this machine's public IP, to coordinates and a US state code."

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
IP_QUERIES = {"", "me", "here", "current", "current location"}
GetJson = Callable[..., Any]


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


def _lookup_ip() -> dict[str, Any] | None:
    import geocoder

    result = geocoder.ip("me")
    if not result.ok or not result.latlng:
        return None
    latitude, longitude = result.latlng
    state_name = result.state
    city = result.city or "Current location"
    country = result.country
    label_parts = [part for part in (city, state_name, country) if part]
    return {
        "query": "me",
        "name": city,
        "label": ", ".join(label_parts) or "Current location",
        "latitude": float(latitude),
        "longitude": float(longitude),
        "state": state_name,
        "state_code": state_code(state_name),
        "country": country,
        "source": "ip",
        "note": "",
    }


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


def resolve_place(
    query: str,
    *,
    settings: Settings | None = None,
    get_json: GetJson | None = None,
    ip_lookup: Callable[[], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Resolve ``query`` to a place dict. Network helpers are injectable for tests."""

    settings = settings or get_settings()
    get_json = get_json or default_get_json
    text = (query or "").strip()
    if text.lower() in IP_QUERIES:
        try:
            found = (ip_lookup or _lookup_ip)()
        except Exception as exc:  # noqa: BLE001 - IP geolocation is a best-effort fallback
            return _empty(text or "me", f"IP location failed ({type(exc).__name__}).", source="ip")
        if not found:
            return _empty(text or "me", "IP location did not return coordinates.", source="ip")
        found["query"] = text or "me"
        return found

    try:
        payload = get_json(
            GEOCODE_URL,
            params={"name": text, "count": 5, "language": "en", "format": "json"},
            timeout=settings.http_timeout,
            user_agent=settings.user_agent,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty(text, f"Geocoding failed ({type(exc).__name__}).")
    results = payload.get("results") if isinstance(payload, dict) else None
    for hit in results or []:
        if hit.get("latitude") is None or hit.get("longitude") is None:
            continue
        return _from_geocode_hit(text, hit)
    return _empty(text, f"No coordinates found for {text!r}.")


@tool
def resolve_place_tool(query: str) -> str:
    """Resolve a place name, or 'me' for this machine, to coordinates and a US state code."""

    import json

    return json.dumps(resolve_place(query))
