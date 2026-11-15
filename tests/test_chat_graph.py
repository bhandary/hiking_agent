"""The chat graph turns one human message into the planner's reply."""

from langchain_core.messages import HumanMessage

from hike_agent.chat_graph import build_chat_graph, latest_human_text


def test_latest_human_text_skips_the_assistant():
    messages = [
        {"role": "user", "content": "hikes near Boulder"},
        {"type": "ai", "content": "earlier plan"},
    ]
    assert latest_human_text(messages) == "hikes near Boulder"


def test_chat_graph_extracts_the_place_and_returns_the_plan():
    seen = {}

    def plan(query, *, when=None, **kwargs):
        seen["query"] = query
        seen["when"] = when
        return {"recommendation": f"Plan for {query}."}

    graph = build_chat_graph(plan)
    state = graph.invoke(
        {"messages": [HumanMessage(content="hikes near Boulder this weekend")]},
    )
    assert seen["query"] == "Boulder"
    assert seen["when"]["start"]
    assert "this weekend" in seen["when"]["label"]
    reply = state["messages"][-1]
    assert reply.type == "ai"
    assert reply.content == "Plan for Boulder."
