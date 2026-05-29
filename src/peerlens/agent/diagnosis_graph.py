from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from ..config import settings
from ..schemas.diagnosis import (
    Contribution,
    DiagnosisReport,
    DiagnosisState,
    EvidenceUpdate,
    Finding,
    PaperModel,
    WritingIssue,
)
from ..schemas.tools import PaperResult

_llm = ChatOpenAI(
    model=settings.llm.model,
    temperature=0.2,
    api_key=settings.llm.openai_api_key,
    **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
)


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------

def _call_llm_json(prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = _llm.invoke(prompt)
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                for part in parts[1:]:
                    part = part[4:].strip() if part.startswith("json") else part.strip()
                    if part:
                        try:
                            return json.loads(part)
                        except Exception:
                            pass
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception:
            if attempt == retries - 1:
                return {}
    return {}


# ---------------------------------------------------------------------------
# Node 1: preprocess (Stage 0)
# ---------------------------------------------------------------------------

def preprocess_node(state: DiagnosisState) -> dict[str, Any]:
    from ..utils.pdf_parser import preprocess_paper_text, split_into_sections

    result = preprocess_paper_text(state.paper_text)
    clean_text = result["clean_text"]
    sections = split_into_sections(clean_text, max_words_per_section=2000)

    return {
        "paper_text": clean_text,
        "noise_log": result["noise_log"],
        "is_chunked": result["is_chunked"],
        "paper_sections": dict(sections),
        "messages": [AIMessage(content=(
            f"Preprocessed: {len(result['noise_log'])} noise tokens removed, "
            f"{'chunked' if result['is_chunked'] else 'single pass'}, "
            f"{len(sections)} sections"
        ))],
        "iteration": state.iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node 2: understand (Stage 1)
# ---------------------------------------------------------------------------

_UNDERSTAND_PROMPT = """You are analyzing an academic ML/AI paper. Build a structural model.
Do NOT evaluate, diagnose, or criticize — only describe.

Paper text:
{text}

Respond with ONLY valid JSON:
{{
  "detected_domain": "concise research area phrase",
  "detected_keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
  "core_claim": "one sentence: what this paper claims to show or achieve",
  "contributions": [
    {{
      "id": "C1",
      "statement": "contribution as stated by the authors",
      "method_location": "section name where the method is described, or empty",
      "experiment_location": "section name where experiments validate this, or empty",
      "result_location": "section name where results for this appear, or empty"
    }}
  ],
  "section_summaries": {{
    "Introduction": "2-3 sentence summary",
    "Methods": "2-3 sentence summary",
    "Experiments": "2-3 sentence summary"
  }},
  "key_claims": ["explicit quantitative or qualitative result claimed by the paper"]
}}
"""

_MERGE_PROMPT = """You received partial structural analyses of chunks of the same paper.
Merge them into one unified structural model. Deduplicate contributions and key claims.

Chunk analyses:
{chunks_json}

Respond with ONLY valid JSON in the same format as the input chunks.
"""


def _understand_single(text: str) -> dict:
    prompt = _UNDERSTAND_PROMPT.format(text=text[:12000])
    return _call_llm_json(prompt)


def understand_node(state: DiagnosisState) -> dict[str, Any]:
    if state.is_chunked and state.paper_sections:
        sections = list(state.paper_sections.values())
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_understand_single, s): i for i, s in enumerate(sections)}
            chunk_results = [None] * len(sections)
            for fut in as_completed(futures):
                chunk_results[futures[fut]] = fut.result()
        valid = [r for r in chunk_results if r]
        if len(valid) > 1:
            merge_prompt = _MERGE_PROMPT.format(chunks_json=json.dumps(valid, ensure_ascii=False))
            data = _call_llm_json(merge_prompt)
        else:
            data = valid[0] if valid else {}
    else:
        data = _understand_single(state.paper_text)

    domain = data.get("detected_domain", "machine learning")
    keywords = data.get("detected_keywords", [])
    abstract = data.get("core_claim", "")

    contributions: list[Contribution] = []
    for c in data.get("contributions", []):
        try:
            contributions.append(Contribution(**c))
        except Exception:
            pass

    paper_model = PaperModel(
        core_claim=data.get("core_claim", ""),
        contributions=contributions,
        section_summaries=data.get("section_summaries", {}),
        key_claims=data.get("key_claims", []),
    )

    return {
        "detected_domain": domain,
        "detected_keywords": keywords,
        "paper_abstract_clean": abstract,
        "paper_model": paper_model,
        "messages": [AIMessage(content=(
            f"Paper understood: domain={domain}, "
            f"{len(contributions)} contributions, "
            f"{len(paper_model.key_claims)} key claims"
        ))],
        "iteration": state.iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node 3: search similar papers (unchanged)
# ---------------------------------------------------------------------------

def search_node(state: DiagnosisState) -> dict[str, Any]:
    from .tools import search_papers as _local_search_papers
    from .tools_remote import resolve_tool
    from ..schemas.tools import SearchPapersOutput

    _search = resolve_tool("search_papers", _local_search_papers)

    domain = state.detected_domain
    keywords = " ".join(state.detected_keywords[:6])
    query = f"{domain} {keywords}".strip()

    updates: dict[str, Any] = {"iteration": state.iteration + 1}
    msgs: list[Any] = []

    try:
        raw = _search.invoke({"query": query, "top_k": 20})
        result = SearchPapersOutput(**raw) if isinstance(raw, dict) else raw
        all_papers: list[PaperResult] = result.results

        accepted = [p for p in all_papers if p.decision in ("oral", "spotlight", "poster", "accepted")]
        rejected = [p for p in all_papers if p.decision == "rejected"]

        updates["similar_accepted"] = accepted
        updates["similar_rejected"] = rejected
        msgs.append(AIMessage(
            content=f"Found {len(accepted)} accepted and {len(rejected)} rejected similar papers"
        ))
    except Exception as e:
        msgs.append(AIMessage(content=f"Paper search failed: {e}"))

    updates["messages"] = msgs
    return updates


# ---------------------------------------------------------------------------
# Node 4: cluster reviewer comments (unchanged)
# ---------------------------------------------------------------------------

def review_analysis_node(state: DiagnosisState) -> dict[str, Any]:
    from .tools import cluster_reviews as _local_cluster, get_paper_reviews as _local_reviews
    from .tools_remote import resolve_tool
    from ..schemas.tools import ClusterAnalysis, GetPaperReviewsOutput

    _cluster = resolve_tool("cluster_reviews", _local_cluster)
    _reviews_tool = resolve_tool("get_paper_reviews", _local_reviews)

    domain = state.detected_domain
    updates: dict[str, Any] = {"iteration": state.iteration + 1}
    msgs: list[Any] = []

    try:
        raw_p = _cluster.invoke({"primary_area": domain, "n_clusters": 5})
        patterns = ClusterAnalysis(**raw_p) if isinstance(raw_p, dict) else raw_p
        updates["review_patterns"] = patterns
        msgs.append(AIMessage(content=f"Cluster analysis: {len(patterns.clusters)} clusters found"))
    except Exception as e:
        msgs.append(AIMessage(content=f"Cluster analysis skipped: {e}"))

    accepted_ids = [p.paper_id for p in state.similar_accepted[:5]]
    rejected_ids = [p.paper_id for p in state.similar_rejected[:5]]
    paper_ids = accepted_ids + rejected_ids

    if paper_ids:
        try:
            raw_r = _reviews_tool.invoke({"paper_ids": paper_ids})
            reviews = GetPaperReviewsOutput(**raw_r) if isinstance(raw_r, dict) else raw_r
            excerpts = []
            for r in reviews.results[:8]:
                text = r.reviews[0].get("text", "") if r.reviews else ""
                if text:
                    excerpts.append(text[:600])
            updates["retrieved_reviews"] = {"count": len(reviews.results), "excerpts": excerpts}
            msgs.append(AIMessage(content=f"Retrieved reviews for {len(reviews.results)} papers"))
        except Exception as e:
            msgs.append(AIMessage(content=f"Review retrieval skipped: {e}"))

    updates["messages"] = msgs
    return updates


# ---------------------------------------------------------------------------
# Node 5: diagnose (Stage 2)
# ---------------------------------------------------------------------------

_DIAGNOSE_PROMPT = """You are an expert academic reviewer. Analyze this paper's structural model
and identify specific problems. Do NOT repeat any problem twice.

PAPER MODEL:
Core claim: {core_claim}

Contributions:
{contributions_block}

Key claims:
{key_claims_block}

Section summaries:
{section_summaries_block}

CONTEXT FROM DATABASE:
Similar accepted papers:
{accepted_summary}

Similar rejected papers:
{rejected_summary}

Reviewer concern clusters in this domain:
{cluster_summary}

INSTRUCTIONS:
For each contribution and key claim, identify specific problems.
Use the section summaries — do not invent content not described there.

Each finding must have:
- nature: EXACTLY one of: content_missing | expression_issue | design_flaw
- repair_cost: EXACTLY one of: one_day_revision | needs_experiment | needs_redesign
- confidence: EXACTLY one of: high | medium | low
- confidence_reason: required when confidence is medium or low (e.g. "may be addressed in appendix not extracted")

Output ONLY valid JSON:
{{
  "findings": [
    {{
      "id": "F1",
      "related_contribution": "C1",
      "problem": "one sentence describing the specific issue",
      "nature": "content_missing",
      "repair_cost": "needs_experiment",
      "confidence": "high",
      "confidence_reason": ""
    }}
  ],
  "executive_summary": "5 sentences max: (1) is the core idea solid? (2) biggest risk for reviewers (3) second biggest risk (4) what the authors must do first (5) overall prognosis"
}}
"""


def diagnose_node(state: DiagnosisState) -> dict[str, Any]:
    model = state.paper_model or PaperModel()

    contributions_block = "\n".join(
        f"  {c.id}: {c.statement}" for c in model.contributions
    ) or "  (none extracted)"

    key_claims_block = "\n".join(
        f"  - {kc}" for kc in model.key_claims
    ) or "  (none extracted)"

    section_summaries_block = "\n".join(
        f"  [{name}]: {summary}" for name, summary in model.section_summaries.items()
    ) or "  (none extracted)"

    accepted_summary = "\n".join(
        f"  - {p.title} ({p.conference} {p.year}): {p.abstract[:150]}"
        for p in state.similar_accepted[:5]
    ) or "  (none found)"

    rejected_summary = "\n".join(
        f"  - {p.title} ({p.conference} {p.year}): {p.abstract[:150]}"
        for p in state.similar_rejected[:5]
    ) or "  (none found)"

    cluster_summary = ""
    if state.review_patterns:
        from ..schemas.tools import ClusterAnalysis
        if isinstance(state.review_patterns, ClusterAnalysis):
            cluster_summary = "\n".join(
                f"  - {c.label}: {c.criticism_pattern}"
                for c in state.review_patterns.clusters[:5]
            )
    cluster_summary = cluster_summary or "  (not available)"

    prompt = _DIAGNOSE_PROMPT.format(
        core_claim=model.core_claim or "(not extracted)",
        contributions_block=contributions_block,
        key_claims_block=key_claims_block,
        section_summaries_block=section_summaries_block,
        accepted_summary=accepted_summary,
        rejected_summary=rejected_summary,
        cluster_summary=cluster_summary,
    )

    data = _call_llm_json(prompt)

    findings: list[Finding] = []
    for f in data.get("findings", []):
        try:
            findings.append(Finding(**f))
        except Exception:
            pass

    executive_summary = data.get("executive_summary", "")

    return {
        "findings": findings,
        "executive_summary": executive_summary,
        "messages": [AIMessage(content=f"Diagnosis complete: {len(findings)} findings")],
        "iteration": state.iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node 6: forensics (Stage 3)
# ---------------------------------------------------------------------------

_FORENSICS_PROMPT = """You are verifying specific findings about a section of an academic paper.

SECTION: {section_name}
{section_text}

FINDINGS TO VERIFY (only for non-high-confidence):
{findings_block}

TASKS:
1. For each finding listed, find a quote from the section text that either supports or contradicts it.
2. Also identify writing issues IN THIS SECTION — typos, broken references, unclear sentences.
   Each writing issue MUST include an exact quote from the section text.

Output ONLY valid JSON:
{{
  "evidence_updates": [
    {{
      "finding_id": "F1",
      "evidence_quote": "exact quote from section that is relevant",
      "verdict": "confirmed | refuted | inconclusive"
    }}
  ],
  "writing_issues": [
    {{
      "quote": "exact text from section",
      "issue": "description of the problem",
      "suggestion": "specific fix"
    }}
  ]
}}
"""


def forensics_node(state: DiagnosisState) -> dict[str, Any]:
    model = state.paper_model or PaperModel()
    findings = state.findings

    non_high = [f for f in findings if f.confidence != "high"]

    contrib_map = {c.id: c for c in model.contributions}

    section_to_findings: dict[str, list[Finding]] = {}
    for f in non_high:
        contrib = contrib_map.get(f.related_contribution)
        section_name = ""
        if contrib:
            section_name = (
                contrib.method_location
                or contrib.experiment_location
                or contrib.result_location
            )
        if not section_name:
            section_name = "__default__"
        section_to_findings.setdefault(section_name, []).append(f)

    default_section_name = next(iter(model.section_summaries), "") if model.section_summaries else ""
    if "__default__" in section_to_findings and not default_section_name:
        del section_to_findings["__default__"]

    all_evidence: list[EvidenceUpdate] = []
    all_writing: list[WritingIssue] = []

    for section_name, sec_findings in section_to_findings.items():
        resolved = section_name if section_name != "__default__" else default_section_name
        section_text = model.section_summaries.get(resolved, "")
        if not section_text:
            for key, val in model.section_summaries.items():
                if resolved.lower() in key.lower():
                    section_text = val
                    break
        if not section_text:
            section_text = "(section text not available)"

        findings_block = "\n".join(
            f"  {f.id}: {f.problem} [confidence={f.confidence}]" for f in sec_findings
        )

        prompt = _FORENSICS_PROMPT.format(
            section_name=resolved or section_name,
            section_text=section_text[:4000],
            findings_block=findings_block,
        )

        data = _call_llm_json(prompt)

        for eu in data.get("evidence_updates", []):
            try:
                all_evidence.append(EvidenceUpdate(**eu))
            except Exception:
                pass

        for wi in data.get("writing_issues", []):
            try:
                obj = WritingIssue(**wi)
                if obj.quote.strip():
                    all_writing.append(obj)
            except Exception:
                pass

    n_findings_checked = sum(len(v) for v in section_to_findings.values())
    return {
        "evidence_updates": all_evidence,
        "writing_issues_stage3": all_writing,
        "messages": [AIMessage(content=(
            f"Forensics: checked {n_findings_checked} findings across "
            f"{len(section_to_findings)} sections, "
            f"{len(all_writing)} writing issues found"
        ))],
        "iteration": state.iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node 7: synthesize (pure assembly, no LLM)
# ---------------------------------------------------------------------------

def synthesize_node(state: DiagnosisState) -> dict[str, Any]:
    refuted_ids = {u.finding_id for u in state.evidence_updates if u.verdict == "refuted"}

    findings: list[Finding] = []
    for f in state.findings:
        if f.id in refuted_ids:
            reason = (f.confidence_reason + " [counterevidence found in text]").strip()
            f = f.model_copy(update={"confidence": "low", "confidence_reason": reason})
        findings.append(f)

    report = DiagnosisReport(
        detected_domain=state.detected_domain,
        detected_keywords=state.detected_keywords,
        executive_summary=state.executive_summary,
        findings=findings,
        evidence_updates=state.evidence_updates,
        writing_issues=state.writing_issues_stage3,
        similar_accepted=state.similar_accepted[:5],
        similar_rejected=state.similar_rejected[:5],
    )

    return {
        "report": report,
        "messages": [AIMessage(content=f"Report assembled: {len(findings)} findings, domain={state.detected_domain}")],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_compiled_graph: Any = None


def build_diagnosis_graph():
    graph = StateGraph(DiagnosisState)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("understand", understand_node)
    graph.add_node("search", search_node)
    graph.add_node("review_analysis", review_analysis_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("forensics", forensics_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "understand")
    graph.add_edge("understand", "search")
    graph.add_edge("search", "review_analysis")
    graph.add_edge("review_analysis", "diagnose")
    graph.add_edge("diagnose", "forensics")
    graph.add_edge("forensics", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def get_diagnosis_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_diagnosis_graph()
    return _compiled_graph
