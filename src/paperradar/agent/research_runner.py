from __future__ import annotations

import concurrent.futures
from typing import Generator

from langchain_core.messages import HumanMessage

from ..schemas.survey import ResearchAgentState, SurveyReport
from .research_graph import get_research_graph

_TIMEOUT_SECONDS = 240
_RECURSION_LIMIT = 45


def _invoke(refined_query: str, focus: dict) -> ResearchAgentState | None:
    from langgraph.errors import GraphRecursionError

    graph = get_research_graph()
    init = ResearchAgentState(
        messages=[HumanMessage(content=refined_query)],
        raw_query=refined_query,
        refined_query=refined_query,
        focus=focus,
    )
    try:
        return graph.invoke(init, config={"recursion_limit": _RECURSION_LIMIT})
    except GraphRecursionError:
        return None


def run_research_agent(refined_query: str, focus: dict) -> SurveyReport | None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_invoke, refined_query, focus)
        try:
            final = future.result(timeout=_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return None
    if final is None:
        return None
    return final.get("survey_report") if isinstance(final, dict) else getattr(final, "survey_report", None)


def stream_research_agent(refined_query: str, focus: dict) -> Generator[dict, None, None]:
    from langgraph.errors import GraphRecursionError

    graph = get_research_graph()
    init = ResearchAgentState(
        messages=[HumanMessage(content=refined_query)],
        raw_query=refined_query,
        refined_query=refined_query,
        focus=focus,
    )
    try:
        for event in graph.stream(init, config={"recursion_limit": _RECURSION_LIMIT}, stream_mode="values"):
            yield event
    except GraphRecursionError:
        yield {"error": "Research agent hit recursion limit — try a more specific query."}
    except Exception as e:
        yield {"error": str(e)}
