"""Fakes shared by agent tests."""


def mcp_down(*_args, **_kwargs):
    return {"ok": False, "text": "", "structured": None, "error": "mcp down"}
