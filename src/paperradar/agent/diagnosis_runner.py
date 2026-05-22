from __future__ import annotations

import concurrent.futures
from typing import Generator

from langchain_core.messages import HumanMessage

from ..schemas.diagnosis import DiagnosisReport, DiagnosisState
from .diagnosis_graph import get_diagnosis_graph

_TIMEOUT_SECONDS = 300
_RECURSION_LIMIT = 45


def _invoke(paper_text: str, target_venue: str) -> DiagnosisState | None:
    from langgraph.errors import GraphRecursionError

    graph = get_diagnosis_graph()
    init = DiagnosisState(
        messages=[HumanMessage(content="Diagnose my paper.")],
        paper_text=paper_text,
        target_venue=target_venue,
    )
    try:
        return graph.invoke(init, config={"recursion_limit": _RECURSION_LIMIT})
    except GraphRecursionError:
        return None


def run_diagnosis_agent(paper_text: str, target_venue: str = "") -> DiagnosisReport | None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_invoke, paper_text, target_venue)
        try:
            final = future.result(timeout=_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return None
    if final is None:
        return None
    return final.get("report") if isinstance(final, dict) else getattr(final, "report", None)


def stream_diagnosis_agent(paper_text: str, target_venue: str = "") -> Generator[dict, None, None]:
    from langgraph.errors import GraphRecursionError

    graph = get_diagnosis_graph()
    init = DiagnosisState(
        messages=[HumanMessage(content="Diagnose my paper.")],
        paper_text=paper_text,
        target_venue=target_venue,
    )
    try:
        for event in graph.stream(init, config={"recursion_limit": _RECURSION_LIMIT}, stream_mode="values"):
            yield event
    except GraphRecursionError:
        yield {"error": "Diagnosis agent hit recursion limit."}
    except Exception as e:
        yield {"error": str(e)}
