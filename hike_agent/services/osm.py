"""Keyless OpenStreetMap fallback for trails when NPS data is unreachable.

Nominatim answers a bounded text search quickly. Overpass can return tagged
hiking routes, and is only used when Nominatim has little to offer, because
public Overpass instances are often slow.
"""

from __future__ import annotations

from typing import Any

import requests

from hike_agent.services.geo import haversine_km
from hike_agent.services.nps import park_record, rank_parks

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Nominatim's bounded text search also returns shops and offices whose names
# merely contain "park". Keep outdoor place types and drop the rest.
_OUTDOOR_TYPES = {
    "path",
    "footway",
    "pedestrian",
    "hiking",
    "route",
    "trail",
    "nature_reserve",
    "national_park",
    "park",
    "peak",
    "wood",
    "forest",
    "fell",
    "cliff",
}
_SKIP_NAME_PARTS = ("office", "village", "headquarters", "headquarter", "visitor center")


def _usable_outdoor_place(name: str, place_type: str) -> bool:
    lowered = name.lower()
    if any(part in lowered for part in _SKIP_NAME_PARTS):
        return False
    if place_type and place_type not in _OUTDOOR_TYPES:
        return False
    return True


def _nominatim(query: str, lat: float, lon: float, *, timeout: float, user_agent: str, get_json: Any) -> list[dict[str, Any]]:
    delta = 0.35
    payload = get_json(
        NOMINATIM_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "limit": 6,
            "viewbox": f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}",
            "bounded": 1,
        },
        timeout=timeout,
        user_agent=user_agent,
        headers={"Accept": "application/json"},
    )
    parks: list[dict[str, Any]] = []
    for item in payload or []:
        name = item.get("name") or ""
        place_type = item.get("type") or ""
        if not name or not _usable_outdoor_place(name, place_type):
            continue
        parks.append(
            park_record(
                {
                    "name": name,
                    "description": item.get("display_name") or "",
                    "designation": item.get("type") or "openstreetmap",
                    "latitude": item.get("lat"),
                    "longitude": item.get("lon"),
                    "url": item.get("osm_type") and f"https://www.openstreetmap.org/{item['osm_type']}/{item.get('osm_id')}",
                    "activities": [name] if "trail" in (item.get("type") or "") or "trail" in name.lower() else [],
                }
            )
        )
    return parks


def _overpass(lat: float, lon: float, *, timeout: float, user_agent: str) -> list[dict[str, Any]]:
    query = f"""
[out:json][timeout:12];
(
  relation["route"="hiking"](around:25000,{lat},{lon});
  way["highway"="path"]["name"~"trail",i](around:8000,{lat},{lon});
);
out tags center 12;
""".strip()
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={"User-Agent": user_agent, "Accept": "application/json"},
        timeout=min(timeout, 18),
    )
    response.raise_for_status()
    parks: list[dict[str, Any]] = []
    for element in response.json().get("elements") or []:
        tags = element.get("tags") or {}
        name = tags.get("name")
        if not name:
            continue
        center = element.get("center") or {}
        latitude = element.get("lat", center.get("lat"))
        longitude = element.get("lon", center.get("lon"))
        kind = tags.get("route") or tags.get("highway") or "openstreetmap"
        element_id = element.get("id")
        osm_type = {"relation": "relation", "way": "way", "node": "node"}.get(element.get("type"), "way")
        parks.append(
            park_record(
                {
                    "name": name,
                    "description": tags.get("description") or f"OpenStreetMap {kind}",
                    "designation": kind,
                    "latitude": latitude,
                    "longitude": longitude,
                    "url": f"https://www.openstreetmap.org/{osm_type}/{element_id}" if element_id else "",
                    "activities": [name],
                }
            )
        )
    return parks


def nearby_outdoor_places(
    latitude: float,
    longitude: float,
    *,
    get_json: Any,
    timeout: float,
    user_agent: str,
    overpass: Any | None = None,
) -> list[dict[str, Any]]:
    """Return named trails and parks near a point, closest first."""

    found: list[dict[str, Any]] = []
    errors: list[str] = []
    for query in ("hiking", "national park"):
        try:
            found.extend(_nominatim(query, latitude, longitude, timeout=timeout, user_agent=user_agent, get_json=get_json))
        except requests.RequestException:
            errors.append(query)
    named = {park["name"] for park in found}
    if len(named) < 3:
        try:
            extra = (overpass or _overpass)(latitude, longitude, timeout=timeout, user_agent=user_agent)
            for park in extra:
                if park["name"] not in named:
                    found.append(park)
                    named.add(park["name"])
        except requests.RequestException:
            pass
    # Drop anything Nominatim placed far outside the search box.
    nearby = []
    for park in found:
        plat, plon = park.get("latitude"), park.get("longitude")
        if plat is None or plon is None:
            continue
        if haversine_km(latitude, longitude, plat, plon) <= 80:
            nearby.append(park)
    return rank_parks(nearby, latitude, longitude, limit=4)
