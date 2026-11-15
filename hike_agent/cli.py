"""Command-line entrypoint for the hike planner."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from hike_agent.config import get_settings
from hike_agent.extract import extract_hike_request
from hike_agent.graph import build_graph
from hike_agent.model import make_complete, warn_model_failure


def plan_hike(query: str, *, complete=None, settings=None, when=None) -> dict[str, Any]:
    """Run the graph for one place query and return the final state."""

    settings = settings or get_settings()
    if complete is None:
        try:
            complete = make_complete(settings)
        except Exception as exc:  # noqa: BLE001
            warn_model_failure(exc)
            complete = None
    graph = build_graph(complete=complete)
    return graph.invoke({"query": query, "when": when or {}, "errors": [], "trace": []})


def format_sources(state: dict[str, Any]) -> str:
    place = state.get("place") or {}
    parks = state.get("parks") or {}
    conditions = state.get("conditions") or {}
    return (
        f"Sources: location={place.get('source')}, "
        f"parks={parks.get('source')}, "
        f"weather={conditions.get('source')}, "
        f"planner={state.get('planner')}"
    )


def _follow_up(state: dict[str, Any], complete) -> None:
    if complete is None:
        print("Follow-up questions need a reachable model. The recommendation above is the full answer.", file=sys.stderr)
        return
    history = state.get("recommendation") or ""
    while True:
        try:
            question = input("\nFollow-up question (exit to quit) > ")
        except EOFError:
            return
        if question.strip().lower() in {"exit", "quit", ""}:
            return
        prompt = f"Prior recommendation:\n{history}\n\nFollow-up question:\n{question}"
        try:
            answer = complete(prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"Model unavailable: {exc}", file=sys.stderr)
            return
        print(answer)
        history += f"\n\nUser: {question}\nAgent: {answer}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan a hike with location, parks, weather, and planner agents.",
    )
    parser.add_argument(
        "place",
        nargs="?",
        help='Place to plan around, for example "Boulder, Colorado". Use "me" for this machine\'s IP location.',
    )
    parser.add_argument("--json", action="store_true", help="Print the graph state as JSON.")
    parser.add_argument("--chat", action="store_true", help="Ask follow-up questions after the recommendation.")
    args = parser.parse_args(argv)
    if not args.place:
        parser.print_help(sys.stderr)
        return 2

    settings = get_settings()
    complete = None
    try:
        complete = make_complete(settings)
    except Exception as exc:  # noqa: BLE001
        warn_model_failure(exc)

    extracted = extract_hike_request(args.place)
    try:
        state = plan_hike(
            extracted["place"],
            complete=complete,
            settings=settings,
            when=extracted.get("when") or {},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Hike planner failed: {exc}", file=sys.stderr)
        return 1

    for line in state.get("trace") or []:
        print(line, file=sys.stderr)
    for error in state.get("errors") or []:
        print(f"note: {error}", file=sys.stderr)

    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        print(state.get("recommendation") or "No recommendation was produced.")
        print()
        print(format_sources(state))

    if args.chat:
        _follow_up(state, complete)
    return 0
