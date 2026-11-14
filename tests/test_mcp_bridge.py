"""Unit tests for MCP result handling. No server is started."""

from types import SimpleNamespace

from hike_agent.mcp_bridge import interpret_tool_result, match_tool, weather_mcp_target


def test_match_tool_accepts_exact_and_prefixed_names():
    tools = [SimpleNamespace(name="weather_get_forecast"), SimpleNamespace(name="get_alerts")]
    assert match_tool(tools, "get_alerts").name == "get_alerts"
    assert match_tool(tools, "get_forecast").name == "weather_get_forecast"
    assert match_tool(tools, "missing") is None


def test_interpret_success_tuple_with_structured_content():
    artifact = SimpleNamespace(structured_content={"parks": [{"fullName": "Zion"}]})
    content = [{"type": "text", "text": "### Zion"}]
    parsed = interpret_tool_result((content, artifact))
    assert parsed["ok"] is True
    assert parsed["text"] == "### Zion"
    assert parsed["structured"]["parks"][0]["fullName"] == "Zion"


def test_interpret_structured_error():
    artifact = SimpleNamespace(structured_content={"error": {"message": "Status: 503"}})
    parsed = interpret_tool_result(([{"type": "text", "text": "Error: 503"}], artifact))
    assert parsed["ok"] is False
    assert "503" in parsed["error"]


def test_weather_target_uses_the_pinned_server(settings):
    target = weather_mcp_target(settings)
    server = target["mcpServers"]["weather"]
    assert server["command"] == "npx"
    assert "@dangahagan/weather-mcp@1.35.17" in server["args"]
    assert server["env"]["ENABLED_TOOLS"] == "basic"
    assert server["env"]["LOG_LEVEL"] == "ERROR"
