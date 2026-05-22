from __future__ import annotations

import concurrent.futures
from typing import Generator

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from ..config import settings
from ..schemas.agent import AgentState
from ..schemas.tools import AnalysisReport
from .graph import get_graph

_RECURSION_LIMIT = settings.agent.max_iterations * 3  # LangGraph step budget
_TIMEOUT_SECONDS = 180  # hard wall-clock timeout


def _invoke(query: str) -> AgentState | None:
    graph = get_graph()
    initial = AgentState(
        messages=[HumanMessage(content=query)],
        raw_query=query,
    )
    try:
        result = graph.invoke(
            initial,
            config={"recursion_limit": _RECURSION_LIMIT},
        )
        return result
    except GraphRecursionError:
        return None


def run_agent(query: str, session_id: str = "") -> AnalysisReport | None:
    """Run agent with a hard timeout. Returns None if timed out or recursion exceeded."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_invoke, query)
        try:
            final = future.result(timeout=_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return None

    if final is None:
        return None
    if isinstance(final, dict):
        return final.get("report")
    return getattr(final, "report", None)


def stream_agent(query: str, session_id: str = "") -> Generator[dict, None, None]:
    """Stream agent state snapshots. Stops after _TIMEOUT_SECONDS."""
    graph = get_graph()
    initial = AgentState(
        messages=[HumanMessage(content=query)],
        raw_query=query,
    )
    try:
        for event in graph.stream(
            initial,
            config={"recursion_limit": _RECURSION_LIMIT},
            stream_mode="values",
        ):
            yield event
    except GraphRecursionError:
        yield {"error": "Agent reached iteration limit and was stopped."}
    except Exception as e:
        yield {"error": str(e)}
