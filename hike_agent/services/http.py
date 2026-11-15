"""Small HTTP helper shared by the direct API fallbacks."""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_USER_AGENT = "hike-agent/0.1 (educational hike planner)"


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Any:
    """GET a JSON document. Raises ``requests.RequestException`` on failure."""

    request_headers = {"User-Agent": user_agent, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    response = requests.get(url, params=params, headers=request_headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
