from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from ..config import settings
from ..schemas.survey import ResearchAgentState, SurveyReport, SurveySection

_llm = ChatOpenAI(
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    api_key=settings.llm.openai_api_key,
    **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
)

# ------------------------------------------------------------------
# Nodes  — direct tool invocation, no ToolNode routing
# ------------------------------------------------------------------

def retrieve_node(state: ResearchAgentState) -> dict[str, Any]:
    from .tools import search_papers as _local_search_papers
    from .tools_remote import resolve_tool
    from ..schemas.tools import SearchPapersOutput

    _search = resolve_tool("search_papers", _local_search_papers)

    focus = state.focus or {}
    query = state.refined_query or state.raw_query
    conferences = focus.get("conferences") or ["NeurIPS", "ICML", "ICLR"]
    years = focus.get("years") or [2022, 2023, 2024, 2025]
    decisions = focus.get("decisions") or None

    try:
        raw = _search.invoke({
            "query": query,
            "conference_filter": conferences,
            "year_filter": [int(y) for y in years],
            "decision_filter": decisions,
            "top_k": 30,
        })
        results = SearchPapersOutput(**raw) if isinstance(raw, dict) else raw
    except Exception as e:
        from ..schemas.tools import SearchPapersOutput
        return {
            "search_results": SearchPapersOutput(results=[], total_found=0, query=query),
            "messages": [AIMessage(content=f"Search failed: {e}")],
            "iteration": state.iteration + 1,
        }

    return {
        "search_results": results,
        "messages": [AIMessage(content=f"Retrieved {results.total_found} papers for: {query}")],
        "iteration": state.iteration + 1,
    }


def analyze_node(state: ResearchAgentState) -> dict[str, Any]:
    from .tools import analyze_temporal_distribution as _local_temporal
    from .tools_remote import resolve_tool
    from ..schemas.tools import TemporalAnalysis

    _temporal_tool = resolve_tool("analyze_temporal_distribution", _local_temporal)

    focus = state.focus or {}
    query = state.refined_query or state.raw_query
    conferences = focus.get("conferences") or ["NeurIPS", "ICML", "ICLR"]
    years = focus.get("years") or [2022, 2023, 2024, 2025]

    updates: dict[str, Any] = {"iteration": state.iteration + 1}
    msgs: list[Any] = []

    try:
        raw_t = _temporal_tool.invoke({
            "topic": query,
            "conferences": conferences,
            "years": [int(y) for y in years],
        })
        temporal = TemporalAnalysis(**raw_t) if isinstance(raw_t, dict) else raw_t
        updates["temporal_analysis"] = temporal
        msgs.append(AIMessage(content=f"Temporal trend: {temporal.trend}"))
    except Exception as e:
        msgs.append(AIMessage(content=f"Temporal analysis skipped: {e}"))

    updates["messages"] = msgs
    return updates


def synthesize_survey_node(state: ResearchAgentState) -> dict[str, Any]:
    topic = state.refined_query or state.raw_query

    # Build paper list for LLM context
    key_papers = []
    papers_block = ""
    no_papers_found = True
    if state.search_results and state.search_results.results:
        key_papers = state.search_results.results[:15]
        no_papers_found = False
        lines = []
        for i, p in enumerate(key_papers):
            abstract = p.abstract[:300] if p.abstract else "(abstract not available)"
            lines.append(
                f"[{i+1}] ID={p.paper_id}\n"
                f"    Title: {p.title}\n"
                f"    Venue: {p.conference} {p.year} ({p.decision})\n"
                f"    Area: {p.primary_area}\n"
                f"    Abstract: {abstract}"
            )
        papers_block = "\n\n".join(lines)

    temporal_block = ""
    if state.temporal_analysis:
        t = state.temporal_analysis
        temporal_block = (
            f"Trend: {t.trend}. Peak: {t.peak_conference} {t.peak_year}. {t.summary}"
        )

    memory_section = ""
    if state.memory_context:
        memory_section = (
            f"\n== YOUR RESEARCH HISTORY ==\n{state.memory_context}\n"
            "In submission_advice, reference relevant past work or diagnoses if applicable.\n"
        )

    if no_papers_found:
        synthesis_prompt = f"""You are writing a topic overview for a researcher.

Topic: "{topic}"

The local paper database has NO results for this query.
Your only job is to write a 2-3 sentence background and 3 open research questions.
Do NOT write survey sections. Do NOT cite any papers or invent any references.

Output ONLY valid JSON:
{{
  "title": "Overview: {topic}",
  "background": "2-3 sentences explaining the research area and why it matters.",
  "sections": [],
  "open_questions": [
    "Specific open question 1",
    "Specific open question 2",
    "Specific open question 3"
  ],
  "submission_advice": "No papers were found in the local database for this topic. Consider crawling relevant conferences (NeurIPS, ICML, ICLR, etc.) via the Library page first, then re-run the survey."
}}
"""
    else:
        synthesis_prompt = f"""You are writing a mini literature survey on: "{topic}"

== PAPERS IN DATABASE ==
{papers_block}

== TREND DATA ==
{temporal_block or "(not available)"}
{memory_section}
== TASK ==
Write a structured literature survey. Output a JSON object with this EXACT schema:

{{
  "title": "A Survey of {topic}",
  "background": "2-3 sentences explaining the problem and why it matters",
  "sections": [
    {{
      "heading": "Section title (e.g. 'Core Approaches', 'Key Findings', 'Evaluation Methods')",
      "content": "3-5 sentences describing this theme. Mention specific paper titles and venues explicitly. Reference them by their [N] index number.",
      "cited_paper_ids": ["paper_id_1", "paper_id_2"]
    }}
  ],
  "open_questions": [
    "Concrete open research question 1",
    "Concrete open research question 2",
    "Concrete open research question 3"
  ],
  "submission_advice": "2-3 sentences of actionable advice for someone wanting to publish in this area"
}}

Requirements:
- Write 3-5 sections covering different facets (methodology landscape, evaluation, trends, limitations, applications, etc.)
- In each section content, explicitly name paper titles and authors/venues. Do NOT be vague.
- Cite at least 8 different papers across all sections using their [N] index number.
- cited_paper_ids MUST only contain IDs that appear in the == PAPERS IN DATABASE == list above. Do NOT invent IDs.
- Open questions should be specific and actionable, not generic.
"""

    response = _llm.invoke(synthesis_prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    valid_ids = {p.paper_id for p in key_papers}

    survey: SurveyReport
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])
        # Validate cited_paper_ids against actual search results
        for s in data.get("sections", []):
            s["cited_paper_ids"] = [
                pid for pid in s.get("cited_paper_ids", []) if pid in valid_ids
            ]
        sections = [SurveySection(**s) for s in data.get("sections", [])]
        survey = SurveyReport(
            title=data.get("title", f"Survey: {topic}"),
            background=data.get("background", ""),
            key_papers=key_papers,
            sections=sections,
            open_questions=data.get("open_questions", []),
            submission_advice=data.get("submission_advice", ""),
            used_training_data=no_papers_found,
        )
    except Exception:
        survey = SurveyReport(
            title=f"Survey: {topic}",
            background="The agent encountered an error generating the survey. Please try again with a more specific query.",
            key_papers=key_papers,
            used_training_data=no_papers_found,
        )

    return {
        "survey_report": survey,
        "messages": [AIMessage(content=f"Survey complete: {survey.title}")],
    }


# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

_compiled_graph: Any = None


def build_research_graph():
    graph = StateGraph(ResearchAgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("synthesize_survey", synthesize_survey_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "synthesize_survey")
    graph.add_edge("synthesize_survey", END)

    return graph.compile()


def get_research_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_research_graph()
    return _compiled_graph
