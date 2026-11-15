"""Direct National Park Service API client.

This is the same parks and things-to-do API the original script called. Alerts
are included now because the planner needs closure and hazard notes.
"""

from __future__ import annotations

from typing import Any, Callable

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


def rank_parks(parks: list[dict[str, Any]], latitude: float | None, longitude: float | None, limit: int = 4) -> list[dict[str, Any]]:
    """Closest national parks first, then other NPS sites, capped for the planner."""

    def sort_key(park: dict[str, Any]) -> tuple[float, int, str]:
        park_lat = park.get("latitude")
        park_lon = park.get("longitude")
        if latitude is None or longitude is None or park_lat is None or park_lon is None:
            distance = 1_000_000.0
        else:
            distance = haversine_km(latitude, longitude, park_lat, park_lon)
        national = 0 if "national park" in (park.get("designation") or "").lower() else 1
        return (distance, national, park.get("name") or "")

    return sorted(parks, key=sort_key)[:limit]


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
