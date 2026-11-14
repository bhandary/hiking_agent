"""Compatibility entrypoint. Prefer ``python -m hike_agent``."""

from hike_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
