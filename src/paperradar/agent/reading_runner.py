from __future__ import annotations

from typing import Generator

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..schemas.reading import ReadingReport, ReadingState
from .reading_graph import get_reading_graph


def start_reading_agent(
    input_mode: str,
    thread_id: str,
    pdf_bytes: bytes | None = None,
    url: str = "",
    paper_title: str = "",
    paper_text: str = "",
    paper_id: str = "",
    paper_authors: list[str] | None = None,
    paper_venue: str = "",
    source_url: str = "",
    memory_context: str | None = None,
) -> Generator[dict, None, None]:
    """
    Stream graph events from parse_input through deep_read.
    Stops at the interrupt in discuss_node, awaiting the first user message.
    """
    graph = get_reading_graph()
    init = ReadingState(
        messages=[HumanMessage(content="Start reading.")],
        input_mode=input_mode,
        pdf_bytes=pdf_bytes,
        url=url,
        paper_title=paper_title,
        paper_text=paper_text,
        paper_id=paper_id,
        paper_authors=paper_authors or [],
        paper_venue=paper_venue,
        source_url=source_url,
        memory_context=memory_context,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 60,
    }
    try:
        for event in graph.stream(init, config=config, stream_mode="values"):
            yield event
    except Exception as e:
        yield {"error": str(e)}


def resume_discussion(
    thread_id: str,
    user_message: str,
) -> Generator[dict, None, None]:
    """
    Resume from interrupt in discuss_node with the user's message.
    Streams until the next interrupt (next turn) or END.
    """
    graph = get_reading_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 60,
    }
    try:
        for event in graph.stream(
            Command(resume=user_message),
            config=config,
            stream_mode="values",
        ):
            yield event
    except Exception as e:
        yield {"error": str(e)}


def get_reading_report(thread_id: str) -> ReadingReport | None:
    """Retrieve the ReadingReport from checkpointer state after deep_read completes."""
    graph = get_reading_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
        values = snapshot.values if hasattr(snapshot, "values") else {}
        return values.get("report")
    except Exception:
        return None
