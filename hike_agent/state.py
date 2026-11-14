"""Shared state for the hike-planning graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class HikeState(TypedDict, total=False):
    """Fields produced as the graph moves from a place name to a recommendation.

    ``errors`` and ``trace`` use a list reducer so the parks and weather nodes
    can write them in the same step.
    """

    query: str
    place: dict[str, Any] | None
    parks: dict[str, Any] | None
    conditions: dict[str, Any] | None
    recommendation: str
    planner: str
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]
