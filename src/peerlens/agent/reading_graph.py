from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from ..config import settings
from ..schemas.reading import (
    MemoryConnection,
    ReadingReport,
    ReadingState,
    ReviewerPerspective,
)
from ..utils.pdf_parser import extract_body_text, extract_title_abstract

_llm = ChatOpenAI(
    model=settings.llm.model,
    temperature=0.1,
    api_key=settings.llm.openai_api_key,
    **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
)

_checkpointer = MemorySaver()

_DONE_PHRASES = {
    "exit", "quit", "done", "bye", "end", "stop", "finish",
    "结束", "退出", "完成", "谢谢", "好了", "够了",
}

# ------------------------------------------------------------------
# Helper: extract JSON from LLM response (handles markdown fences)
# ------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts[1:]:
            if part.startswith("json"):
                part = part[4:]
            part = part.strip()
            if part:
                try:
                    return json.loads(part)
                except Exception:
                    pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError("No JSON object found in response")


# ------------------------------------------------------------------
# Nodes
# ------------------------------------------------------------------

def parse_input_node(state: ReadingState) -> dict[str, Any]:
    updates: dict[str, Any] = {"active_node": "parse_input", "iteration": state.iteration + 1}
    if state.input_mode == "pdf" and state.pdf_bytes:
        full_text = _extract_pdf_full(state.pdf_bytes)
        meta = extract_title_abstract(full_text)
        updates["paper_text"] = extract_body_text(full_text, max_words=10_000)
        if not state.paper_title:
            updates["paper_title"] = meta.get("title", "")
        updates["messages"] = [AIMessage(content=f"Extracted PDF text: {len(updates['paper_text'].split())} words")]
    return updates


def _extract_pdf_full(pdf_bytes: bytes) -> str:
    """Extract all pages of a PDF into plain text (no word limit)."""
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_openreview_node(state: ReadingState) -> dict[str, Any]:
    from .tools_reading import fetch_paper_from_openreview

    url = state.url
    m = re.search(r"id=([A-Za-z0-9_-]+)", url)
    forum_id = m.group(1) if m else url.strip("/").split("/")[-1]

    result = fetch_paper_from_openreview(forum_id)
    return {
        "active_node": "fetch_openreview",
        "paper_id": result.get("paper_id", forum_id),
        "paper_title": result.get("title", ""),
        "paper_authors": result.get("authors", []),
        "paper_venue": result.get("venue", ""),
        "paper_text": result.get("full_text", result.get("abstract", "")),
        "source_url": result.get("source_url", url),
        "messages": [AIMessage(content=f"Fetched from OpenReview: {result.get('title', '(untitled)')}")]
    }


def fetch_arxiv_node(state: ReadingState) -> dict[str, Any]:
    from .tools_reading import fetch_paper_from_arxiv

    result = fetch_paper_from_arxiv(state.url)
    return {
        "active_node": "fetch_arxiv",
        "paper_id": result.get("paper_id", ""),
        "paper_title": result.get("title", ""),
        "paper_authors": result.get("authors", []),
        "paper_venue": result.get("venue", ""),
        "paper_text": result.get("full_text", result.get("abstract", "")),
        "source_url": result.get("source_url", state.url),
        "messages": [AIMessage(content=f"Fetched from ArXiv: {result.get('title', '(untitled)')}")]
    }


def inject_reviews_node(state: ReadingState) -> dict[str, Any]:
    """Try to find this paper's reviews in the local DB by paper_id or title match."""
    from .tools import get_paper_reviews as _local_reviews
    from .tools_remote import resolve_tool
    from ..schemas.tools import GetPaperReviewsOutput

    _reviews_tool = resolve_tool("get_paper_reviews", _local_reviews)
    updates: dict[str, Any] = {"active_node": "inject_reviews"}
    msgs: list[Any] = []

    paper_ids: list[str] = []
    if state.paper_id:
        paper_ids = [state.paper_id]
    elif state.paper_title:
        # Try title-based lookup via vector search with string similarity guard
        try:
            from difflib import SequenceMatcher
            from ..store.chroma import ChromaManager
            from ..retrieval.embedder import Embedder
            embedder = Embedder()
            chroma = ChromaManager()
            query_emb = embedder.embed_query(state.paper_title)
            result = chroma.query_content(query_emb, n_results=1)
            if result.get("ids") and result["ids"][0]:
                candidate_id = result["ids"][0][0]
                meta = result.get("metadatas", [[]])[0]
                candidate_title = meta[0].get("title", "") if meta else ""
                sim = SequenceMatcher(
                    None,
                    state.paper_title.lower(),
                    candidate_title.lower(),
                ).ratio()
                if sim >= 0.6:
                    paper_ids = [candidate_id]
        except Exception:
            pass

    if paper_ids:
        try:
            raw = _reviews_tool.invoke({"paper_ids": paper_ids})
            reviews_out = GetPaperReviewsOutput(**raw) if isinstance(raw, dict) else raw
            flat: list[dict] = []
            for r in reviews_out.results:
                for rev in r.reviews:
                    flat.append(rev)
            updates["paper_reviews"] = flat
            msgs.append(AIMessage(content=f"Loaded {len(flat)} reviews for this paper"))
        except Exception as e:
            msgs.append(AIMessage(content=f"Review injection skipped: {e}"))
    else:
        msgs.append(AIMessage(content="No local reviews found for this paper"))

    updates["messages"] = msgs
    return updates


_DEEP_READ_SYSTEM_BASE = """\
You are an expert academic reading assistant with deep knowledge of machine learning and AI research.
Your task is to produce a thorough, structured reading report for the given paper.

Respond with ONLY valid JSON matching this schema exactly:
{{
  "tldr": "One to two sentences summarizing the paper.",
  "problem_statement": "What problem does this paper solve and why does it matter?",
  "core_contributions": ["contribution 1", "contribution 2", ...],
  "methodology_summary": "High-level description of the technical approach.",
  "key_innovations": ["innovation 1", "innovation 2", ...],
  "datasets_and_baselines": "What datasets and baselines were used?",
  "main_results": "Key quantitative results and findings.",
  "ablations": "Ablation study summary (empty string if none).",
  "limitations": ["limitation 1", "limitation 2", ...],
  "open_questions": ["question 1", "question 2", ...],
  {reviewer_schema}
  {memory_schema}
}}

Be specific and technical. Avoid vague statements."""

_REVIEWER_SCHEMA_WITH_DATA = (
    '"reviewer_perspectives": [\n'
    '    {\n'
    '      "reviewer_id": "Reviewer 1",\n'
    '      "stance": "positive",\n'
    '      "key_points": ["point 1", "point 2"]\n'
    '    }\n'
    '  ],'
)

_REVIEWER_SCHEMA_EMPTY = '"reviewer_perspectives": [],  // no reviewer data available — always return empty array'

_MEMORY_SCHEMA_WITH_DATA = (
    '"memory_connections": [\n'
    '    {\n'
    '      "session_id": "<exact session_id from the research history provided above>",\n'
    '      "agent_type": "<diagnosis|research|reading>",\n'
    '      "connection_description": "Specific, concrete description of how this paper relates to that session.",\n'
    '      "related_input_summary": "<brief description of the related session>",\n'
    '      "relevance_score": 0.85\n'
    '    }\n'
    '  ]'
)

_MEMORY_SCHEMA_EMPTY = '"memory_connections": []  // no research history provided — always return empty array'


def _build_deep_read_system(has_reviews: bool, has_memory: bool) -> str:
    reviewer_schema = _REVIEWER_SCHEMA_WITH_DATA if has_reviews else _REVIEWER_SCHEMA_EMPTY
    memory_schema = _MEMORY_SCHEMA_WITH_DATA if has_memory else _MEMORY_SCHEMA_EMPTY
    return _DEEP_READ_SYSTEM_BASE.format(
        reviewer_schema=reviewer_schema,
        memory_schema=memory_schema,
    )


def deep_read_node(state: ReadingState) -> dict[str, Any]:
    paper_text = extract_body_text(state.paper_text, max_words=10_000) if state.paper_text else ""
    title = state.paper_title or "(untitled)"
    venue = state.paper_venue or ""
    authors = ", ".join(state.paper_authors[:5]) if state.paper_authors else ""

    # Format reviews section
    if state.paper_reviews:
        review_lines = []
        for i, rev in enumerate(state.paper_reviews[:6], 1):
            reviewer_id = f"Reviewer {i}"
            strengths = rev.get("strengths", "")
            weaknesses = rev.get("weaknesses", "")
            rating = rev.get("rating") or rev.get("overall") or ""
            review_lines.append(
                f"[{reviewer_id}]{' Rating: ' + str(rating) if rating else ''}\n"
                f"Strengths: {strengths[:400]}\n"
                f"Weaknesses: {weaknesses[:400]}"
            )
        review_section = "\n\n[Reviewer Perspectives]\n" + "\n---\n".join(review_lines)
    else:
        review_section = "\n\n[Reviewer Perspectives]: NOT AVAILABLE"

    # Format memory section
    if state.memory_context:
        memory_section = f"\n\n[Your Research History]\n{state.memory_context}"
    else:
        memory_section = "\n\n[Your Research History]: NOT AVAILABLE"

    user_content = (
        f"[Paper]\n"
        f"Title: {title}\n"
        f"Venue: {venue}\n"
        f"Authors: {authors}\n\n"
        f"{paper_text}"
        f"{review_section}"
        f"{memory_section}"
    )

    system_prompt = _build_deep_read_system(
        has_reviews=bool(state.paper_reviews),
        has_memory=bool(state.memory_context),
    )

    try:
        response = _llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        data = _parse_json(raw)

        reviewer_perspectives = [
            ReviewerPerspective(**rp) for rp in data.get("reviewer_perspectives", [])
        ]
        memory_connections = []
        for mc in data.get("memory_connections", []):
            try:
                memory_connections.append(MemoryConnection(
                    agent_type=mc.get("agent_type", "diagnosis"),
                    session_id=mc.get("session_id", ""),
                    timestamp=datetime.utcnow(),
                    connection_description=mc.get("connection_description", ""),
                    related_input_summary=mc.get("related_input_summary", ""),
                    relevance_score=float(mc.get("relevance_score", 0.5)),
                ))
            except Exception:
                continue

        report = ReadingReport(
            paper_title=title,
            authors=state.paper_authors,
            venue=venue,
            source_url=state.source_url,
            tldr=data.get("tldr", ""),
            problem_statement=data.get("problem_statement", ""),
            core_contributions=data.get("core_contributions", []),
            methodology_summary=data.get("methodology_summary", ""),
            key_innovations=data.get("key_innovations", []),
            datasets_and_baselines=data.get("datasets_and_baselines", ""),
            main_results=data.get("main_results", ""),
            ablations=data.get("ablations", ""),
            limitations=data.get("limitations", []),
            open_questions=data.get("open_questions", []),
            reviewer_perspectives=reviewer_perspectives,
            memory_connections=memory_connections,
        )
    except Exception as e:
        report = ReadingReport(
            paper_title=title,
            authors=state.paper_authors,
            venue=venue,
            source_url=state.source_url,
            tldr=f"Failed to parse reading report: {e}",
            problem_statement="",
        )

    return {
        "report": report,
        "discussion_active": True,
        "active_node": "deep_read",
        "messages": [AIMessage(content=f"Deep reading complete: {title}")],
    }


_DISCUSSION_SYSTEM_TEMPLATE = """\
You are an expert academic discussion partner who has deeply read the following paper.
You have access to:
1. The full paper content
2. Real reviewer comments (if available)
3. The user's research history (if available)

You can discuss the paper from multiple angles:
- Explain technical details clearly
- Argue as devil's advocate using reviewer criticisms
- Connect insights to the user's own research
- Suggest follow-up papers or experiments

Keep responses focused, informative, and intellectually engaging.

---
[Paper Title]: {title}
[Venue]: {venue}

[Paper Body]:
{paper_text}

[Reviewer Comments]:
{reviews}

[User Research History]:
{memory_context}

[Reading Report Summary]:
{report_summary}
"""


def _build_discussion_system(state: ReadingState) -> str:
    paper_text = extract_body_text(state.paper_text, max_words=8_000) if state.paper_text else ""
    reviews = ""
    if state.paper_reviews:
        lines = []
        for i, rev in enumerate(state.paper_reviews[:4], 1):
            w = rev.get("weaknesses", "")[:300]
            s = rev.get("strengths", "")[:200]
            lines.append(f"Reviewer {i}: Strengths: {s} | Weaknesses: {w}")
        reviews = "\n".join(lines)
    memory_context = state.memory_context or "(none)"
    report_summary = ""
    if state.report:
        report_summary = (
            f"TL;DR: {state.report.tldr}\n"
            f"Limitations: {'; '.join(state.report.limitations[:3])}\n"
            f"Open questions: {'; '.join(state.report.open_questions[:3])}"
        )
    return _DISCUSSION_SYSTEM_TEMPLATE.format(
        title=state.paper_title or "(untitled)",
        venue=state.paper_venue or "",
        paper_text=paper_text,
        reviews=reviews or "(no reviewer comments available)",
        memory_context=memory_context,
        report_summary=report_summary or "(report not yet generated)",
    )


def discuss_node(state: ReadingState) -> dict[str, Any] | Command:
    user_input: str = interrupt("Awaiting discussion input")

    if user_input.strip().lower() in _DONE_PHRASES:
        return Command(goto=END)

    system_prompt = _build_discussion_system(state)
    context_messages = [SystemMessage(content=system_prompt)]
    # Include recent conversation history (last 10 turns)
    recent = state.messages[-20:] if len(state.messages) > 20 else state.messages
    context_messages += [m for m in recent if isinstance(m, (HumanMessage, AIMessage))]
    context_messages.append(HumanMessage(content=user_input))

    response = _llm.invoke(context_messages)
    reply = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "messages": [HumanMessage(content=user_input), AIMessage(content=reply)],
        "active_node": "discuss",
        "iteration": state.iteration + 1,
    }


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------

def _route_after_parse(state: ReadingState) -> str:
    return {
        "pdf": "inject_reviews",
        "openreview_url": "fetch_openreview",
        "arxiv_url": "fetch_arxiv",
        "topic_query_selected": "inject_reviews",
    }.get(state.input_mode, "inject_reviews")


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

_compiled_graph: Any = None


def build_reading_graph():
    graph = StateGraph(ReadingState)
    graph.add_node("parse_input", parse_input_node)
    graph.add_node("fetch_openreview", fetch_openreview_node)
    graph.add_node("fetch_arxiv", fetch_arxiv_node)
    graph.add_node("inject_reviews", inject_reviews_node)
    graph.add_node("deep_read", deep_read_node)
    graph.add_node("discuss", discuss_node)

    graph.set_entry_point("parse_input")
    graph.add_conditional_edges(
        "parse_input",
        _route_after_parse,
        {
            "inject_reviews": "inject_reviews",
            "fetch_openreview": "fetch_openreview",
            "fetch_arxiv": "fetch_arxiv",
        },
    )
    graph.add_edge("fetch_openreview", "inject_reviews")
    graph.add_edge("fetch_arxiv", "inject_reviews")
    graph.add_edge("inject_reviews", "deep_read")
    graph.add_edge("deep_read", "discuss")
    graph.add_edge("discuss", "discuss")  # self-loop; Command(goto=END) overrides

    return graph.compile(checkpointer=_checkpointer)


def get_reading_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_reading_graph()
    return _compiled_graph
