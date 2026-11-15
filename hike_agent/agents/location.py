"""Location agent: turn a place name or client coordinates into a map point.

``me``, ``here``, and ``near me`` do not use this machine's IP address. On a
hosted server that address is the host, not the hiker. Those queries need a
city or park name, or coordinates the client sent.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from langchain_core.tools import tool

from hike_agent.config import Settings, get_settings
from hike_agent.coordinates import parse_coordinates, strip_coordinates
from hike_agent.extract import is_activity_phrase
from hike_agent.services.geo import haversine_km
from hike_agent.services.http import get_json as default_get_json
from hike_agent.services.nps import search_parks_by_name
from hike_agent.services.states import STATE_ABBREVIATIONS, state_code

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
ParkSearch = Callable[[str], list[dict[str, Any]]]
_NEAR_WORDS = re.compile(r"\b(?:at|near|around|nearby|close\s+to)\b", re.IGNORECASE)
_GENERIC_PLACE_WORDS = {
    "and",
    "area",
    "around",
    "close",
    "day",
    "hike",
    "hikes",
    "hiking",
    "historic",
    "historical",
    "lakeshore",
    "memorial",
    "monument",
    "national",
    "near",
    "of",
    "park",
    "parks",
    "preserve",
    "recreation",
    "recreational",
    "reserve",
    "seashore",
    "site",
    "sites",
    "state",
    "the",
    "united",
    "us",
    "weekend",
}
_QUERY_FILLER = _GENERIC_PLACE_WORDS | {
    "friday",
    "monday",
    "next",
    "saturday",
    "sunday",
    "this",
    "thursday",
    "today",
    "tomorrow",
    "tuesday",
    "wednesday",
}
_FEATURE_SCORE = {
    "PPLC": 40,
    "PPLA": 34,
    "PPLA2": 28,
    "PPLA3": 22,
    "PPLA4": 16,
    "PPL": 12,
    "PPLX": 8,
    "STLMT": 10,
    "PRK": 24,
    "RESN": 20,
    "RESV": 20,
    "FRST": 20,
    "VAL": 18,
    "MT": 16,
    "MTS": 16,
    "LK": 14,
}
_STATE_NAMES = {abbrev: name for name, abbrev in STATE_ABBREVIATIONS.items()}


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
    search_parks: ParkSearch | None = None,
) -> dict[str, Any]:
    """Resolve ``query`` to a place dict. Network helpers are injectable for tests.

    A place that names a national park uses that park's coordinates. Other
    names are ranked across the geocoder's hits instead of taking the first one.
    """

    settings = settings or get_settings()
    own_client = get_json is None
    get_json = get_json or default_get_json
    text = (query or "").strip()
    coords = parse_coordinates(coordinates) or parse_coordinates(text)
    if _is_near_me(text) or (not text and coords):
        if coords:
            return _from_coordinates(text or "me", coords, settings=settings, get_json=get_json)
        return _empty(text or "me", NEED_LOCATION, source="needs-location")
    if not text or is_activity_phrase(text):
        return _empty(text, NEED_LOCATION, source="needs-location")

    parks = _parks_for_query(text, settings=settings, get_json=get_json, search_parks=search_parks, own_client=own_client)
    matched = _matching_park(text, parks)
    if matched is not None:
        return _from_park(text, matched)

    try:
        payload = get_json(
            GEOCODE_URL,
            params={"name": text, "count": 10, "language": "en", "format": "json"},
            timeout=settings.http_timeout,
            user_agent=settings.user_agent,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty(text, f"Geocoding failed ({type(exc).__name__}).", source="error")
    results = payload.get("results") if isinstance(payload, dict) else None
    chosen = _best_geocode_hit(text, results or [], parks)
    if chosen is not None:
        return _from_geocode_hit(text, chosen)
    return _empty(text, f"No coordinates found for {text!r}.")


def _parks_for_query(
    text: str,
    *,
    settings: Settings,
    get_json: GetJson,
    search_parks: ParkSearch | None,
    own_client: bool,
) -> list[dict[str, Any]]:
    if search_parks is not None:
        try:
            return list(search_parks(text) or [])
        except Exception:  # noqa: BLE001 - a park miss still leaves geocoding
            return []
    if not own_client:
        return []
    try:
        return search_parks_by_name(
            text,
            mcp_url=settings.nps_mcp_url,
            mcp_enabled=settings.nps_mcp_enabled,
            api_key=settings.nps_api_key,
            get_json=get_json,
            timeout=settings.http_timeout,
            user_agent=settings.user_agent,
        )
    except Exception:  # noqa: BLE001
        return []


def _distinctive_tokens(name: str, *, filler: set[str]) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return [word for word in words if word not in filler and len(word) > 2]


def _matching_park(query: str, parks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The NPS park this query names, or nothing when the query is a town."""

    query_tokens = _distinctive_tokens(query, filler=_QUERY_FILLER)
    code = re.sub(r"[^a-z0-9]", "", (query or "").lower())
    best: dict[str, Any] | None = None
    best_score = 0
    for park in parks:
        if park.get("latitude") is None or park.get("longitude") is None:
            continue
        park_code = re.sub(r"[^a-z0-9]", "", str(park.get("parkCode") or park.get("park_code") or "").lower())
        park_tokens = _distinctive_tokens(str(park.get("fullName") or park.get("name") or ""), filler=_GENERIC_PLACE_WORDS)
        if code and code == park_code:
            score = 100
        elif query_tokens and park_tokens and set(query_tokens) == set(park_tokens):
            score = 100
        else:
            continue
        designation = str(park.get("designation") or "").lower()
        if "national park" in designation:
            score += 5
        if score > best_score:
            best = park
            best_score = score
    return best


def _from_park(query: str, park: dict[str, Any]) -> dict[str, Any]:
    states = str(park.get("states") or park.get("state") or "")
    code = state_code(states.split(",")[0].strip()) or ""
    name = str(park.get("fullName") or park.get("name") or query)
    state_name = _STATE_NAMES.get(code, "")
    return {
        "query": query,
        "name": name,
        "label": ", ".join(part for part in (name, state_name, "United States") if part),
        "latitude": float(park["latitude"]),
        "longitude": float(park["longitude"]),
        "state": state_name or None,
        "state_code": code or None,
        "country": "United States",
        "source": "nps",
        "note": "",
    }


def _best_geocode_hit(query: str, hits: list[Any], parks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Rank geocoder hits by country, state, population, feature, and park distance."""

    best: dict[str, Any] | None = None
    best_score = float("-inf")
    for hit in hits:
        if not isinstance(hit, dict) or hit.get("latitude") is None or hit.get("longitude") is None:
            continue
        score = _geocode_score(query, hit, parks)
        if score > best_score:
            best = hit
            best_score = score
    return best


def _geocode_score(query: str, hit: dict[str, Any], parks: list[dict[str, Any]]) -> float:
    score = 0.0
    country = str(hit.get("country_code") or hit.get("country") or "")
    if country in {"US", "USA", "United States"}:
        score += 50
    admin = str(hit.get("admin1") or "")
    if admin and admin.lower() in query.lower():
        score += 40
    population = hit.get("population") or 0
    try:
        population = float(population)
    except (TypeError, ValueError):
        population = 0
    if population > 0:
        score += min(math.log10(population) * 15, 80)
    score += _FEATURE_SCORE.get(str(hit.get("feature_code") or "").upper(), 0)
    distances: list[float] = []
    for park in parks:
        if park.get("latitude") is None or park.get("longitude") is None:
            continue
        distances.append(
            haversine_km(
                float(hit["latitude"]),
                float(hit["longitude"]),
                float(park["latitude"]),
                float(park["longitude"]),
            )
        )
    if distances:
        score += max(0.0, 120 - min(distances))
    return score


@tool
def resolve_place_tool(query: str) -> str:
    """Resolve a place name to coordinates. 'me' needs a city name or client coordinates."""

    import json

    return json.dumps(resolve_place(query))
