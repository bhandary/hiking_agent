"""Parks and trails agent.

Order of sources:
1. Local National Park Service MCP (stdio) when ``NPS_API_KEY`` is set.
2. Hosted National Park Service MCP.
3. The original NPS REST API, when ``NPS_API_KEY`` is set.
4. OpenStreetMap. Those rows are labeled ``not verified by NPS``.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import requests
from langchain_core.tools import tool

from hike_agent.config import Settings, get_settings
from hike_agent.mcp_bridge import call_mcp_tool, call_mcp_tools, nps_stdio_target
from hike_agent.services.http import get_json as default_get_json
from hike_agent.services.nps import (
    alert_lines,
    fetch_parks_bundle,
    hike_titles,
    park_record,
    rank_parks,
)
from hike_agent.services.osm import nearby_outdoor_places

AGENT_NAME = "parks"
AGENT_BRIEF = "Find nearby national parks, hiking activities, and current park alerts."

McpCall = Callable[[Any, str, dict[str, Any], float], dict[str, Any]]
PARK_HEADING = re.compile(r"^###\s+(?P<name>.+)$", re.MULTILINE)
PARK_CODE = re.compile(r"\*\*parkCode:\*\*\s+(?P<code>[A-Za-z0-9]{3,6})")
NPS_SOURCES = frozenset({"nps-mcp-local", "nps-mcp", "nps-api"})
NOT_VERIFIED = "not verified by NPS"
_NOT_A_PARK_QUERY = {"provided location", "me", "here", "current", "current location", "near me", "my location"}


def _park_query_name(name: Any) -> str:
    """City or park text safe to send as an NPS query. Placeholders are not."""

    text = str(name or "").strip()
    if not text or text.lower() in _NOT_A_PARK_QUERY:
        return ""
    return text


def _items(structured: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(structured, list):
        return [item for item in structured if isinstance(item, dict)]
    if not isinstance(structured, dict):
        return []
    for key in keys:
        value = structured.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def parks_from_mcp_result(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Read parks out of an MCP tool result, structured JSON first, then markdown."""

    if not result or not result.get("ok"):
        return []
    structured = result.get("structured")
    items = _items(structured, "parks", "data")
    if items:
        return [park_record(item) for item in items]
    text = result.get("text") or ""
    parks: list[dict[str, Any]] = []
    for match in PARK_HEADING.finditer(text):
        window = text[match.end() : match.end() + 400]
        code_match = PARK_CODE.search(window)
        parks.append(
            park_record(
                {
                    "fullName": match.group("name").strip(),
                    "parkCode": code_match.group("code") if code_match else None,
                }
            )
        )
    return parks


def _apply_activity_payload(park: dict[str, Any], activities: dict[str, Any] | None, alerts: dict[str, Any] | None) -> None:
    activity_items = _items((activities or {}).get("structured"), "activities", "thingsToDo", "data")
    alert_items = _items((alerts or {}).get("structured"), "alerts", "data")
    trails = hike_titles(activity_items)
    if not trails and activities and activities.get("ok"):
        titled = []
        for line in (activities.get("text") or "").splitlines():
            if line.startswith("### "):
                titled.append({"title": line[4:].strip()})
        trails = hike_titles(titled)
    if trails:
        park["trails"] = trails
    parsed_alerts = alert_lines(alert_items)
    if parsed_alerts:
        park["alerts"] = parsed_alerts


def _attach_mcp_details(
    parks: list[dict[str, Any]],
    mcp_call: McpCall,
    target: str,
    timeout: float,
) -> None:
    selected = [park for park in parks if park.get("park_code")]
    calls = []
    for park in selected:
        code = park["park_code"]
        calls.append(("nps_get_activities", {"parkCode": code, "query": "hiking", "limit": 6}))
        calls.append(("nps_get_alerts", {"parkCode": code, "limit": 5}))
    if not calls:
        return
    if mcp_call is call_mcp_tool:
        results = call_mcp_tools(target, calls, timeout)
    else:
        results = [mcp_call(target, name, arguments, timeout) for name, arguments in calls]
    for index, park in enumerate(selected):
        activities = results[index * 2] if index * 2 < len(results) else None
        alerts = results[index * 2 + 1] if index * 2 + 1 < len(results) else None
        _apply_activity_payload(park, activities, alerts)


def _report(source: str, note: str, parks: list[dict[str, Any]]) -> dict[str, Any]:
    verified = source in NPS_SOURCES
    for park in parks:
        park["source"] = source
        park["nps_verified"] = verified
    return {"source": source, "note": note, "nps_verified": verified, "parks": parks}


def _mcp_attempts(settings: Settings) -> list[tuple[str, Any]]:
    """Local stdio first when a key is set, then the hosted server."""

    attempts: list[tuple[str, Any]] = []
    if settings.nps_mcp_enabled and settings.nps_api_key:
        attempts.append(("nps-mcp-local", nps_stdio_target(settings)))
    if settings.nps_mcp_enabled and settings.nps_mcp_url:
        attempts.append(("nps-mcp", settings.nps_mcp_url))
    return attempts


def find_hikes(
    place: dict[str, Any],
    *,
    settings: Settings | None = None,
    mcp_call: McpCall | None = None,
    get_json: Callable[..., Any] | None = None,
    overpass: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Return ``{source, note, parks}`` for a resolved place."""

    settings = settings or get_settings()
    mcp_call = mcp_call or call_mcp_tool
    get_json = get_json or default_get_json
    latitude = place.get("latitude")
    longitude = place.get("longitude")
    if latitude is None or longitude is None:
        return _report("none", "No coordinates, so park search was skipped.", [])

    notes: list[str] = []
    search_name = _park_query_name(place.get("name"))
    arguments: dict[str, Any] = {"limit": 8}
    if place.get("state_code"):
        arguments["stateCode"] = place["state_code"]
    if search_name:
        arguments["query"] = search_name
    # A coordinate-only place has no city. Querying NPS for a placeholder such as
    # "Provided location", or with no query at all, returns parks for the wrong place.
    if not search_name and not place.get("state_code"):
        notes.append("No place name or state, so the park-name search was skipped.")
        attempts: list[tuple[str, Any]] = []
    else:
        attempts = _mcp_attempts(settings)
    for source_name, target in attempts:
        found = mcp_call(target, "nps_find_parks", arguments, settings.nps_mcp_timeout)
        parks = rank_parks(parks_from_mcp_result(found), latitude, longitude, limit=4)
        if parks:
            _attach_mcp_details(parks, mcp_call, target, settings.nps_mcp_timeout)
            return _report(source_name, "", parks)
        error = str((found or {}).get("error") or "no parks returned")
        label = "Local NPS MCP" if source_name == "nps-mcp-local" else "Hosted NPS MCP"
        notes.append(f"{label} unavailable ({error[:180]}).")

    if settings.nps_api_key:
        try:
            parks = fetch_parks_bundle(
                api_key=settings.nps_api_key,
                state_code=place.get("state_code"),
                latitude=latitude,
                longitude=longitude,
                get_json=get_json,
                timeout=settings.http_timeout,
                user_agent=settings.user_agent,
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "error"
            notes.append(f"NPS API returned HTTP {status}.")
            parks = []
        except requests.RequestException:
            notes.append("NPS API request failed.")
            parks = []
        except ValueError:
            notes.append("NPS API returned data that was not JSON.")
            parks = []
        if parks:
            return _report("nps-api", "", parks)
        if not place.get("state_code"):
            notes.append("NPS API needs a US state code, and this place did not resolve to one.")
    else:
        notes.append("NPS_API_KEY is not set, so the local NPS MCP server and the National Park Service API were skipped.")

    try:
        osm_parks = nearby_outdoor_places(
            latitude,
            longitude,
            get_json=get_json,
            timeout=settings.http_timeout,
            user_agent=settings.user_agent,
            overpass=overpass,
        )
    except requests.RequestException:
        osm_parks = []
        notes.append("OpenStreetMap lookup failed.")
    if osm_parks:
        note = " ".join(notes)
        note = (
            f"{note} Results are from OpenStreetMap and are {NOT_VERIFIED}."
        ).strip()
        return _report("openstreetmap", note, osm_parks)

    return _report("none", " ".join(notes) or "No parks or trails were found.", [])


@tool
def find_hikes_tool(place_json: str) -> str:
    """Find parks and trails for a place JSON object with latitude, longitude, and state_code."""

    import json

    place = json.loads(place_json)
    return json.dumps(find_hikes(place))
