"""Call MCP tools through LangChain's built-in MCP adapter.

``langchain-mcp-adapters`` (``MultiServerMCPClient``) is unmaintained. LangChain
1.4+ moved that client into ``langchain.mcp.MCPAdapter``, which is what this
module uses. Tool calls are best-effort: any failure returns ``ok: False`` so
the parks and weather agents can fall back to direct HTTP.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

from hike_agent.config import Settings


def nps_stdio_target(settings: Settings) -> dict[str, Any]:
    """Local National Park Service MCP server. Requires ``NPS_API_KEY`` in the child env."""

    return {
        "mcpServers": {
            "nps": {
                "command": settings.nps_mcp_command,
                "args": list(settings.nps_mcp_args),
                "env": {"NPS_API_KEY": settings.nps_api_key},
                "timeout": int(settings.nps_mcp_timeout * 1000),
            }
        }
    }


def weather_mcp_target(settings: Settings) -> dict[str, Any]:
    """Stdio config for the keyless NOAA / Open-Meteo weather server."""

    return {
        "mcpServers": {
            "weather": {
                "command": settings.weather_mcp_command,
                "args": list(settings.weather_mcp_args),
                "env": {"ENABLED_TOOLS": "basic", "LOG_LEVEL": "ERROR"},
                "timeout": int(settings.weather_mcp_timeout * 1000),
            }
        }
    }


def match_tool(tools: list[Any], name: str) -> Any | None:
    """Find a tool by exact name, or by a server-prefixed ``server_tool`` name."""

    for tool in tools:
        if getattr(tool, "name", None) == name:
            return tool
    suffix = f"_{name}"
    for tool in tools:
        tool_name = getattr(tool, "name", "")
        if tool_name.endswith(suffix):
            return tool
    return None


def text_from_content(content: Any) -> str:
    """Flatten MCP / LangChain content blocks into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                block_text = getattr(block, "text", None)
                parts.append(str(block_text if block_text is not None else block))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def structured_from_artifact(artifact: Any) -> Any:
    """Read ``structuredContent`` off an MCP tool artifact."""

    if artifact is None:
        return None
    if isinstance(artifact, dict):
        if "structured_content" in artifact:
            return artifact["structured_content"]
        return artifact
    return getattr(artifact, "structured_content", None)


def interpret_tool_result(result: Any) -> dict[str, Any]:
    """Normalize a LangChain MCP tool return value."""

    artifact = None
    content = result
    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
    text = text_from_content(content)
    structured = structured_from_artifact(artifact)
    if isinstance(structured, dict) and structured.get("error"):
        error = structured["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        return {"ok": False, "text": text, "structured": structured, "error": message or text or "MCP tool error"}
    if text.lower().startswith("error:"):
        return {"ok": False, "text": text, "structured": structured, "error": text}
    return {"ok": True, "text": text, "structured": structured, "error": None}


def _missing_tool(tools: list[Any], tool_name: str) -> dict[str, Any]:
    available = ", ".join(getattr(item, "name", "?") for item in tools) or "(none)"
    return {
        "ok": False,
        "text": "",
        "structured": None,
        "error": f"MCP tool {tool_name} not found. Available: {available}",
    }


async def _invoke_many(target: Any, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    # Importing langchain.mcp warns once per process that the namespace is beta.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*langchain.mcp.*")
        from langchain.mcp import MCPAdapter

    async with MCPAdapter(target) as adapter:
        tools = await adapter.list_tools()
        results: list[dict[str, Any]] = []
        for tool_name, arguments in calls:
            tool = match_tool(tools, tool_name)
            if tool is None:
                results.append(_missing_tool(tools, tool_name))
                continue
            try:
                results.append(interpret_tool_result(await tool.ainvoke(arguments)))
            except Exception as exc:  # noqa: BLE001 - one bad tool should not drop the others
                results.append(
                    {
                        "ok": False,
                        "text": "",
                        "structured": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results


def call_mcp_tools(
    target: Any,
    calls: list[tuple[str, dict[str, Any]]],
    timeout: float,
) -> list[dict[str, Any]]:
    """Call several tools on one MCP server. Never raises.

    One connection is reused so a stdio server is not started once per tool.
    """

    if not calls:
        return []
    try:
        return asyncio.run(asyncio.wait_for(_invoke_many(target, calls), timeout))
    except Exception as exc:  # noqa: BLE001 - fallback depends on a total failure signal
        detail = str(exc).strip()
        error = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
        failure = {"ok": False, "text": "", "structured": None, "error": error}
        return [failure for _ in calls]


def call_mcp_tool(
    target: Any,
    tool_name: str,
    arguments: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Call one MCP tool. Never raises; connection and timeout failures are ``ok: False``."""

    return call_mcp_tools(target, [(tool_name, arguments)], timeout)[0]
