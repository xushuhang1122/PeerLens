from __future__ import annotations

import concurrent.futures
from typing import Generator

from langchain_core.messages import HumanMessage

from ..schemas.survey import ResearchAgentState, SurveyReport
from .research_graph import get_research_graph

_TIMEOUT_SECONDS = 240
_RECURSION_LIMIT = 45


def _get_memory_context(query_text: str) -> str | None:
    """Pre-graph: retrieve relevant history and build memory context string. Silent on failure."""
    try:
        from ..memory.agent_memory import AgentMemoryManager
        mgr = AgentMemoryManager()
        sessions = mgr.retrieve_relevant(query_text[:1000], n_results=3)
        return mgr.build_memory_context(sessions, "research") if sessions else None
    except Exception:
        return None


def _save_session_async(query_text: str, report: SurveyReport) -> None:
    """Async fire-and-forget: persist session to memory store and save report to disk."""
    def _save():
        try:
            from ..memory.agent_memory import AgentMemoryManager
            AgentMemoryManager().save_session("research", query_text, report)
        except Exception:
            pass
        try:
            from ..utils.report_saver import save_survey_report
            save_survey_report(report, query_text)
        except Exception:
            pass
    concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(_save)


def _invoke(refined_query: str, focus: dict, memory_context: str | None = None) -> ResearchAgentState | None:
    from langgraph.errors import GraphRecursionError

    graph = get_research_graph()
    init = ResearchAgentState(
        messages=[HumanMessage(content=refined_query)],
        raw_query=refined_query,
        refined_query=refined_query,
        focus=focus,
        memory_context=memory_context,
    )
    try:
        return graph.invoke(init, config={"recursion_limit": _RECURSION_LIMIT})
    except GraphRecursionError:
        return None


def run_research_agent(refined_query: str, focus: dict) -> SurveyReport | None:
    memory_context = _get_memory_context(refined_query)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_invoke, refined_query, focus, memory_context)
        try:
            final = future.result(timeout=_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return None
    if final is None:
        return None
    report = final.get("survey_report") if isinstance(final, dict) else getattr(final, "survey_report", None)
    if report:
        _save_session_async(refined_query, report)
    return report


def stream_research_agent(refined_query: str, focus: dict) -> Generator[dict, None, None]:
    from langgraph.errors import GraphRecursionError

    memory_context = _get_memory_context(refined_query)
    graph = get_research_graph()
    init = ResearchAgentState(
        messages=[HumanMessage(content=refined_query)],
        raw_query=refined_query,
        refined_query=refined_query,
        focus=focus,
        memory_context=memory_context,
    )
    final_report = None
    try:
        for event in graph.stream(init, config={"recursion_limit": _RECURSION_LIMIT}, stream_mode="values"):
            report = event.get("survey_report") if isinstance(event, dict) else None
            if report:
                final_report = report
            yield event
    except GraphRecursionError:
        yield {"error": "Research agent hit recursion limit — try a more specific query."}
    except Exception as e:
        yield {"error": str(e)}
    if final_report:
        _save_session_async(refined_query, final_report)
