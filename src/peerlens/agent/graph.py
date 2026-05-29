from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ..config import settings
from ..schemas.agent import AgentState
from .nodes import (
    analyze_node,
    cross_validate_node,
    query_parse_node,
    report_node,
    retrieve_node,
)
from .tools import ALL_TOOLS

_NODE_ORDER = ["retrieve", "analyze", "cross_validate", "report"]


def _has_tool_calls(state: AgentState) -> bool:
    msgs = state.messages
    if not msgs:
        return False
    last = msgs[-1]
    return isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", None))


def _route_retrieve(state: AgentState) -> Literal["tools", "analyze"]:
    return "tools" if _has_tool_calls(state) else "analyze"


def _route_analyze(state: AgentState) -> Literal["tools", "cross_validate"]:
    return "tools" if _has_tool_calls(state) else "cross_validate"


def _route_cross_validate(state: AgentState) -> Literal["tools", "report"]:
    return "tools" if _has_tool_calls(state) else "report"


def _route_report(state: AgentState) -> Literal["tools", "__end__"]:
    return "tools" if _has_tool_calls(state) else END


def _route_after_tools(state: AgentState) -> str:
    """Return to whichever node last triggered a tool call."""
    node = state.active_node
    if node in _NODE_ORDER:
        return node
    return "retrieve"


def build_graph():
    tool_node = ToolNode(
        ALL_TOOLS,
        handle_tool_errors=True,  # convert exceptions to ToolMessages, don't crash
    )

    graph = StateGraph(AgentState)

    graph.add_node("query_parse", query_parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("cross_validate", cross_validate_node)
    graph.add_node("report", report_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("query_parse")
    graph.add_edge("query_parse", "retrieve")

    graph.add_conditional_edges("retrieve", _route_retrieve)
    graph.add_conditional_edges("analyze", _route_analyze)
    graph.add_conditional_edges("cross_validate", _route_cross_validate)
    graph.add_conditional_edges("report", _route_report)

    # Tools route back to the node that called them
    graph.add_conditional_edges("tools", _route_after_tools)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
