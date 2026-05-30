from __future__ import annotations

import concurrent.futures
from datetime import datetime
from typing import Generator

from langchain_core.messages import HumanMessage

from ..schemas.diagnosis import DiagnosisReport, DiagnosisState
from .diagnosis_graph import get_diagnosis_graph

_TIMEOUT_SECONDS = 600
_RECURSION_LIMIT = 45


def _get_memory_context(paper_text: str) -> str | None:
    """Pre-graph: retrieve relevant history and build memory context string. Silent on failure."""
    try:
        from ..memory.agent_memory import AgentMemoryManager
        mgr = AgentMemoryManager()
        sessions = mgr.retrieve_relevant(paper_text[:1000], n_results=3)
        return mgr.build_memory_context(sessions, "diagnosis") if sessions else None
    except Exception:
        return None


def _save_session_async(paper_text: str, report: DiagnosisReport) -> None:
    """Async fire-and-forget: compress and persist session to memory store."""
    def _save():
        try:
            from ..memory.agent_memory import AgentMemoryManager
            AgentMemoryManager().save_session("diagnosis", paper_text, report)
        except Exception:
            pass
    concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(_save)


def _invoke(paper_text: str, target_venue: str, memory_context: str | None = None) -> DiagnosisState | None:
    from langgraph.errors import GraphRecursionError

    graph = get_diagnosis_graph()
    init = DiagnosisState(
        messages=[HumanMessage(content="Diagnose my paper.")],
        paper_text=paper_text,
        target_venue=target_venue,
        memory_context=memory_context,
    )
    try:
        return graph.invoke(init, config={"recursion_limit": _RECURSION_LIMIT})
    except GraphRecursionError:
        return None
    except Exception:
        return None


def run_diagnosis_agent(paper_text: str, target_venue: str = "") -> DiagnosisReport | None:
    memory_context = _get_memory_context(paper_text)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_invoke, paper_text, target_venue, memory_context)
        try:
            final = future.result(timeout=_TIMEOUT_SECONDS)
        except (concurrent.futures.TimeoutError, Exception):
            future.cancel()
            return None
    if final is None:
        return None
    report = final.get("report") if isinstance(final, dict) else getattr(final, "report", None)
    if report:
        _save_session_async(paper_text, report)
    return report


def stream_diagnosis_agent(paper_text: str, target_venue: str = "") -> Generator[dict, None, None]:
    from langgraph.errors import GraphRecursionError
    from .diagnosis_graph import init_run_logger

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_run_logger(run_id)

    memory_context = _get_memory_context(paper_text)
    graph = get_diagnosis_graph()
    init = DiagnosisState(
        messages=[HumanMessage(content="Diagnose my paper.")],
        paper_text=paper_text,
        target_venue=target_venue,
        memory_context=memory_context,
    )
    final_report = None
    try:
        for event in graph.stream(init, config={"recursion_limit": _RECURSION_LIMIT}, stream_mode="values"):
            report = event.get("report") if isinstance(event, dict) else None
            if report:
                final_report = report
            yield event
    except GraphRecursionError:
        yield {"error": "Diagnosis agent hit recursion limit."}
    except Exception as e:
        yield {"error": str(e)}
    if final_report:
        _save_session_async(paper_text, final_report)
