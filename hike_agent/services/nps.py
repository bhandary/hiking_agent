"""Direct National Park Service API client.

This is the same parks and things-to-do API the original script called. Alerts
are included now because the planner needs closure and hazard notes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import requests

from hike_agent.services.geo import haversine_km

NPS_BASE = "https://developer.nps.gov/api/v1"
GetJson = Callable[..., Any]


def _truncate(text: str | None, limit: int = 320) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def hike_titles(items: list[dict[str, Any]], limit: int = 6) -> list[str]:
    """Prefer hiking and trail activities, then fall back to the first titles."""

    preferred: list[str] = []
    fallback: list[str] = []
    for item in items:
        title = (item.get("title") or item.get("name") or "").strip()
        if not title:
            continue
        tags = item.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        haystack = " ".join([title.lower(), *[str(tag).lower() for tag in tags]])
        if any(word in haystack for word in ("hike", "hiking", "trail")):
            preferred.append(title)
        else:
            fallback.append(title)
    chosen = preferred or fallback[:3]
    seen: list[str] = []
    for title in chosen:
        if title not in seen:
            seen.append(title)
        if len(seen) >= limit:
            break
    return seen


def alert_lines(items: list[dict[str, Any]], limit: int = 4) -> list[str]:
    lines: list[str] = []
    for item in items:
        title = (item.get("title") or item.get("headline") or item.get("name") or "").strip()
        category = (item.get("category") or item.get("severity") or "").strip()
        if not title and not category:
            continue
        line = f"{category}: {title}".strip(": ")
        if line not in lines:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def park_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one NPS park, MCP park summary, or similar object."""

    latitude = _float_or_none(raw.get("latitude"))
    longitude = _float_or_none(raw.get("longitude"))
    activities = raw.get("activities") or []
    activity_names: list[str] = []
    if activities and isinstance(activities[0], str):
        activity_names = [name for name in activities if isinstance(name, str)]
    elif isinstance(activities, list):
        activity_names = hike_titles(activities)
    return {
        "name": raw.get("fullName") or raw.get("name") or "Unnamed park",
        "park_code": raw.get("parkCode") or raw.get("park_code"),
        "description": _truncate(raw.get("description") or ""),
        "designation": raw.get("designation") or "",
        "latitude": latitude,
        "longitude": longitude,
        "url": raw.get("url") or "",
        "trails": activity_names[:6],
        "alerts": alert_lines(raw.get("alerts") or []),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# A day-hike search should not surface a park hundreds of kilometres away
# just because nearer rows were missing coordinates.
NEARBY_RADIUS_KM = 160.0


def rank_parks(
    parks: list[dict[str, Any]],
    latitude: float | None,
    longitude: float | None,
    limit: int = 4,
    radius_km: float | None = NEARBY_RADIUS_KM,
) -> list[dict[str, Any]]:
    """Closest parks that have coordinates, capped for the planner.

    Rows without a latitude and longitude are left out. They used to be given
    a dummy distance, which let far parks and coordinate-less parks fill the
    list. When the hiker's point is known, parks outside ``radius_km`` are
    left out too.
    """

    located = [
        park
        for park in parks
        if park.get("latitude") is not None and park.get("longitude") is not None
    ]
    if latitude is not None and longitude is not None and radius_km is not None:
        located = [
            park
            for park in located
            if haversine_km(latitude, longitude, float(park["latitude"]), float(park["longitude"])) <= radius_km
        ]

    def sort_key(park: dict[str, Any]) -> tuple[float, int, str]:
        if latitude is None or longitude is None:
            distance = 0.0
        else:
            distance = haversine_km(latitude, longitude, float(park["latitude"]), float(park["longitude"]))
        national = 0 if "national park" in (park.get("designation") or "").lower() else 1
        return (distance, national, park.get("name") or "")

    return sorted(located, key=sort_key)[:limit]


def fetch_parks_bundle(
    *,
    api_key: str,
    state_code: str | None,
    latitude: float | None,
    longitude: float | None,
    get_json: GetJson,
    timeout: float,
    user_agent: str,
) -> list[dict[str, Any]]:
    """Fetch parks, hiking-oriented things to do, and alerts for one state."""

    if not state_code:
        return []
    parks_payload = get_json(
        f"{NPS_BASE}/parks",
        params={"stateCode": state_code, "limit": 50, "api_key": api_key},
        timeout=timeout,
        user_agent=user_agent,
    )
    parks = [park_record(item) for item in parks_payload.get("data") or []]
    selected = rank_parks(parks, latitude, longitude, limit=4)
    for park in selected:
        code = park.get("park_code")
        if not code:
            continue
        things = get_json(
            f"{NPS_BASE}/thingstodo",
            params={"parkCode": code, "limit": 20, "api_key": api_key},
            timeout=timeout,
            user_agent=user_agent,
        )
        alerts = get_json(
            f"{NPS_BASE}/alerts",
            params={"parkCode": code, "limit": 10, "api_key": api_key},
            timeout=timeout,
            user_agent=user_agent,
        )
        park["trails"] = hike_titles(things.get("data") or []) or park["trails"]
        park["alerts"] = alert_lines(alerts.get("data") or [])
    return selected


_PARK_HEADING = re.compile(r"^### (?P<name>.+)$", re.MULTILINE)
_PARK_META = re.compile(
    r"\*\*parkCode:\*\*\s*(?P<code>\w+)\s*\|\s*\*\*(?P<designation>[^*]+)\*\*\s*\|\s*\*\*States:\*\*\s*(?P<states>[A-Z]{2}(?:,\s*[A-Z]{2})*)"
)
_PARK_COORDINATES = re.compile(
    r"\*\*Coordinates:\*\*\s*(?P<lat>-?\d+(?:\.\d+)?),\s*(?P<lon>-?\d+(?:\.\d+)?)"
)


def parks_from_find_text(text: str) -> list[dict[str, Any]]:
    """Parks from the hosted NPS MCP's markdown, including coordinates."""

    parks: list[dict[str, Any]] = []
    for chunk in re.split(r"\n(?=### )", text or ""):
        heading = _PARK_HEADING.search(chunk)
        if not heading:
            continue
        meta = _PARK_META.search(chunk)
        coords = _PARK_COORDINATES.search(chunk)
        parks.append(
            {
                "fullName": heading.group("name").strip(),
                "parkCode": meta.group("code") if meta else None,
                "designation": meta.group("designation").strip() if meta else "",
                "states": meta.group("states") if meta else "",
                "latitude": float(coords.group("lat")) if coords else None,
                "longitude": float(coords.group("lon")) if coords else None,
            }
        )
    return parks


def search_parks_by_name(
    query: str,
    *,
    mcp_url: str = "",
    mcp_enabled: bool = False,
    api_key: str = "",
    get_json: GetJson | None = None,
    timeout: float = 12,
    user_agent: str = "hike-agent/0.1",
) -> list[dict[str, Any]]:
    """Find NPS parks by name. The hosted MCP works without an API key."""

    text = (query or "").strip()
    if not text:
        return []
    parks: list[dict[str, Any]] = []
    if mcp_enabled and mcp_url:
        try:
            parks = parks_from_find_text(_hosted_find_parks(mcp_url, text, timeout=timeout, user_agent=user_agent))
        except (requests.RequestException, ValueError, TypeError):
            parks = []
    if parks:
        return parks
    if not api_key or get_json is None:
        return []
    try:
        payload = get_json(
            f"{NPS_BASE}/parks",
            params={"q": text, "limit": 10, "api_key": api_key},
            timeout=timeout,
            user_agent=user_agent,
        )
    except (requests.RequestException, ValueError, TypeError):
        return []
    found: list[dict[str, Any]] = []
    for item in (payload or {}).get("data") or []:
        if not isinstance(item, dict):
            continue
        found.append(
            {
                "fullName": item.get("fullName") or item.get("name"),
                "parkCode": item.get("parkCode"),
                "designation": item.get("designation") or "",
                "states": item.get("states") or "",
                "latitude": _float_or_none(item.get("latitude")),
                "longitude": _float_or_none(item.get("longitude")),
            }
        )
    return found


def _hosted_find_parks(url: str, query: str, *, timeout: float, user_agent: str) -> str:
    response = requests.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "nps_find_parks", "arguments": {"query": query, "limit": 8}},
        },
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": user_agent,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.text
    payload: dict[str, Any] | None = None
    if body.lstrip().startswith("{"):
        payload = response.json()
    else:
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and "result" in event:
                payload = event
    if not payload:
        return ""
    result = payload.get("result") or {}
    parts: list[str] = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts)
