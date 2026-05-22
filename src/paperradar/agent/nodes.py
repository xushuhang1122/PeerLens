from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..schemas.agent import AgentState
from .tools import ALL_TOOLS

_llm = ChatOpenAI(
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    api_key=settings.llm.openai_api_key,
    **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
)
_llm_with_tools = _llm.bind_tools(ALL_TOOLS)

_SYSTEM_PROMPT = """You are PaperRadar, an expert research assistant for ML/AI academic papers.
Local database: NeurIPS, ICML, ICLR, ACL, EMNLP, AISTATS, UAI, CoRL, COLM (2022-2025).

Available tools: search_papers, get_paper_reviews, cluster_reviews,
analyze_temporal_distribution, identify_research_gaps, get_user_context,
web_search, check_and_crawl_new_conference, discover_conference, generate_report.

Rules:
- Be concise. Call at most 3 tools per turn.
- If search_papers returns results, proceed to generate_report. Do not loop.
- Only call web_search / check_and_crawl_new_conference when the user explicitly asks about a specific new conference release.
- Always finish with generate_report.
"""

_MAX_TOOL_CALLS = settings.agent.max_tool_retries * 3  # hard ceiling


def _extract_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end]) if start >= 0 else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _over_budget(state: AgentState) -> bool:
    return state.tool_call_count >= _MAX_TOOL_CALLS or state.iteration >= settings.agent.max_iterations


def query_parse_node(state: AgentState) -> dict[str, Any]:
    if _over_budget(state):
        return {"error_log": state.error_log + ["Budget exceeded in query_parse"]}

    prompt = (
        f"Parse this research query into structured intent.\n"
        f"Query: {state.raw_query}\n\n"
        "Return JSON with keys:\n"
        "- topic: main research topic\n"
        "- features: list chosen from [search, temporal, clustering, gap, user_context, crawl_check]\n"
        "  (keep it minimal — for a simple question, only include 'search')\n"
        "- decision_preference: list of oral/spotlight/poster or null\n"
        "- conferences: list of conference names or null\n"
        "- years: list of ints or null\n"
        "- domain: primary_area string for clustering/gap or null"
    )
    resp = _llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    parsed = _extract_json(resp.content if isinstance(resp.content, str) else "")
    return {"parsed_intent": parsed, "iteration": state.iteration + 1}


def retrieve_node(state: AgentState) -> dict[str, Any]:
    if _over_budget(state):
        return {"error_log": state.error_log + ["Budget exceeded in retrieve — forcing report"]}

    intent = state.parsed_intent or {}
    features = intent.get("features", [])

    if not features or "search" in features or "crawl_check" in features:
        messages = list(state.messages) + [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User query: {state.raw_query}\n"
                    f"Intent: {json.dumps(intent)}\n\n"
                    "Call search_papers now. Use decision/conference/year filters from intent if specified. "
                    "Do not call web_search unless the user explicitly asked about a new conference release."
                )
            ),
        ]
        response = _llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "tool_call_count": state.tool_call_count + 1,
            "active_node": "retrieve",
        }
    return {}


def analyze_node(state: AgentState) -> dict[str, Any]:
    if _over_budget(state):
        return {}

    intent = state.parsed_intent or {}
    features = intent.get("features", [])
    needs_analysis = any(f in features for f in ["temporal", "clustering", "gap"])
    if not needs_analysis:
        return {}

    domain = intent.get("domain") or intent.get("topic", "")
    instructions = []
    if "temporal" in features:
        instructions.append("Call analyze_temporal_distribution.")
    if "clustering" in features:
        instructions.append("Call cluster_reviews.")
    if "gap" in features:
        instructions.append("Call identify_research_gaps.")

    messages = list(state.messages) + [
        HumanMessage(
            content=(
                f"Now run the requested analysis for: {state.raw_query}\n"
                f"Domain: {domain}\n"
                + "\n".join(instructions)
                + "\nCall at most 2 tools in this step."
            )
        )
    ]
    response = _llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "tool_call_count": state.tool_call_count + 1,
        "active_node": "analyze",
    }


def cross_validate_node(state: AgentState) -> dict[str, Any]:
    if _over_budget(state):
        return {}

    messages = list(state.messages) + [
        HumanMessage(
            content=(
                "Call get_user_context once to load user preferences, "
                "then note any overlap with found papers. Do not call any other tools."
            )
        )
    ]
    response = _llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "tool_call_count": state.tool_call_count + 1,
        "active_node": "cross_validate",
    }


def report_node(state: AgentState) -> dict[str, Any]:
    messages = list(state.messages) + [
        HumanMessage(
            content=(
                f"Generate the final report for: '{state.raw_query}'\n"
                "Call generate_report exactly once with all findings. "
                "Include paper citations and actionable recommendations."
            )
        )
    ]
    response = _llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "iteration": state.iteration + 1,
        "active_node": "report",
    }
