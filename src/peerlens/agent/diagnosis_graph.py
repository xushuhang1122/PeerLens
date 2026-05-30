from __future__ import annotations

import json
import time
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
from ..utils.run_logger import RunLogger

_llm = ChatOpenAI(
    model=settings.llm.model,
    temperature=0.2,
    api_key=settings.llm.openai_api_key,
    **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
)

# Module-level logger — reset at the start of each diagnosis run.
_run_logger: RunLogger | None = None


def init_run_logger(run_id: str) -> RunLogger:
    global _run_logger
    _run_logger = RunLogger(run_id)
    return _run_logger


# ---------------------------------------------------------------------------
# Shared retry helper (records token usage + elapsed to run logger)
# ---------------------------------------------------------------------------

def _call_llm_json(prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = _llm.invoke(prompt)
            elapsed = time.time() - t0

            if _run_logger:
                usage = (resp.response_metadata or {}).get("token_usage", {})
                _run_logger.record_llm(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    elapsed_s=elapsed,
                )

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
    prompt = _UNDERSTAND_PROMPT.format(text=text)
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
            f"{len(paper_model.key_claims)} key claims, "
            f"paper_text={len(state.paper_text.split())} words"
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

    accepted: list[PaperResult] = []
    rejected: list[PaperResult] = []

    try:
        raw = _search.invoke({
            "query": query,
            "top_k": 20,
            "decision_filter": ["oral", "spotlight", "poster", "accepted"],
        })
        result = SearchPapersOutput(**raw) if isinstance(raw, dict) else raw
        accepted = result.results
    except Exception as e:
        msgs.append(AIMessage(content=f"Accepted paper search failed: {e}"))

    try:
        raw = _search.invoke({
            "query": query,
            "top_k": 10,
            "decision_filter": ["rejected"],
        })
        result = SearchPapersOutput(**raw) if isinstance(raw, dict) else raw
        rejected = result.results
    except Exception as e:
        msgs.append(AIMessage(content=f"Rejected paper search failed: {e}"))

    updates["similar_accepted"] = accepted
    updates["similar_rejected"] = rejected
    msgs.append(AIMessage(
        content=f"Found {len(accepted)} accepted and {len(rejected)} rejected similar papers"
    ))

    updates["messages"] = msgs
    return updates


# ---------------------------------------------------------------------------
# Node 4: cluster reviewer comments + relevance filtering
# ---------------------------------------------------------------------------

def _filter_by_relevance(core_claim: str, papers: list, top_k: int = 3) -> list:
    """LLM-rank papers by relevance to core_claim; return top_k."""
    if len(papers) <= top_k:
        return papers
    candidates = "\n".join(
        f"  {i+1}. {p.title}: {p.abstract[:120]}" for i, p in enumerate(papers)
    )
    prompt = (
        f"Current paper core claim: {core_claim}\n\n"
        f"Rate each candidate paper's relevance (0.0-1.0) to the current paper.\n"
        f"Candidates:\n{candidates}\n\n"
        f"IMPORTANT: Output ONLY a single line of JSON, nothing else:\n"
        f"{{\"scores\": [float, float, ...]}}  <- one float per candidate, in order"
    )
    data = _call_llm_json(prompt)
    scores = data.get("scores", [])
    if len(scores) == len(papers):
        ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:top_k]]
    return papers[:top_k]


def review_analysis_node(state: DiagnosisState) -> dict[str, Any]:
    from .tools import cluster_reviews as _local_cluster, get_paper_reviews as _local_reviews
    from .tools_remote import resolve_tool
    from ..schemas.tools import ClusterAnalysis, GetPaperReviewsOutput

    _cluster = resolve_tool("cluster_reviews", _local_cluster)
    _reviews_tool = resolve_tool("get_paper_reviews", _local_reviews)

    domain = state.detected_domain
    updates: dict[str, Any] = {"iteration": state.iteration + 1}
    msgs: list[Any] = []

    # Relevance filtering: keep top-3 most relevant from each list
    core_claim = state.paper_model.core_claim if state.paper_model else ""
    filtered_accepted = state.similar_accepted
    filtered_rejected = state.similar_rejected
    if core_claim:
        try:
            if len(state.similar_accepted) > 3:
                filtered_accepted = _filter_by_relevance(core_claim, state.similar_accepted)
            if len(state.similar_rejected) > 3:
                filtered_rejected = _filter_by_relevance(core_claim, state.similar_rejected)
            updates["similar_accepted"] = filtered_accepted
            updates["similar_rejected"] = filtered_rejected
            msgs.append(AIMessage(content=(
                f"Relevance filter: {len(filtered_accepted)} accepted, "
                f"{len(filtered_rejected)} rejected papers kept"
            )))
        except Exception as e:
            msgs.append(AIMessage(content=f"Relevance filtering skipped: {e}"))

    try:
        raw_p = _cluster.invoke({"primary_area": domain, "n_clusters": 5})
        patterns = ClusterAnalysis(**raw_p) if isinstance(raw_p, dict) else raw_p
        updates["review_patterns"] = patterns
        msgs.append(AIMessage(content=f"Cluster analysis: {len(patterns.clusters)} clusters found"))
    except Exception as e:
        msgs.append(AIMessage(content=f"Cluster analysis skipped: {e}"))

    accepted_id_set = {p.paper_id for p in filtered_accepted}
    paper_ids = [p.paper_id for p in filtered_accepted] + [p.paper_id for p in filtered_rejected]

    if paper_ids:
        try:
            raw_r = _reviews_tool.invoke({"paper_ids": paper_ids})
            reviews = GetPaperReviewsOutput(**raw_r) if isinstance(raw_r, dict) else raw_r
            accepted_excerpts: list[str] = []
            rejected_excerpts: list[str] = []
            for r in reviews.results[:8]:
                text = r.reviews[0].get("text", "") if r.reviews else ""
                if not text:
                    continue
                if r.paper_id in accepted_id_set:
                    accepted_excerpts.append(text[:800])
                else:
                    rejected_excerpts.append(text[:800])
            updates["accepted_review_excerpts"] = accepted_excerpts
            updates["rejected_review_excerpts"] = rejected_excerpts
            updates["retrieved_reviews"] = {
                "count": len(reviews.results),
                "excerpts": accepted_excerpts + rejected_excerpts,
            }
            msgs.append(AIMessage(content=(
                f"Reviews: {len(accepted_excerpts)} accepted, "
                f"{len(rejected_excerpts)} rejected excerpts"
            )))
        except Exception as e:
            msgs.append(AIMessage(content=f"Review retrieval skipped: {e}"))

    updates["messages"] = msgs
    return updates


# ---------------------------------------------------------------------------
# Node 5: diagnose (Stage 2)
# ---------------------------------------------------------------------------

_DIAGNOSE_PROMPT = """You are an expert academic reviewer. Read the full paper and identify specific, grounded problems.

[PAPER STRUCTURE MODEL]
Core claim: {core_claim}

Contributions:
{contributions_block}

Key claims:
{key_claims_block}

[FULL PAPER TEXT]
{paper_text}

[ACCEPTED PAPER REVIEWS]
These are reviews of accepted papers with similar research focus.
Use them to calibrate the evidence standard — what level of proof is required
for a contribution claim to be considered sufficient in this area.
{accepted_reviews_block}

[REJECTED PAPER REVIEWS]
These are reviews of papers that were rejected from similar venues.
Use them to identify dealbreakers — which types of problems consistently cause
rejection so you can flag whether this paper has the same issues.
{rejected_reviews_block}

[REVIEWER CONCERN CLUSTERS IN THIS DOMAIN]
{cluster_summary}

[DIAGNOSIS INSTRUCTIONS]
For each contribution and key claim, identify specific problems grounded in the paper text.
Every problem statement must reference specific content from the paper — not generic criticism.

REPAIR COST DEFINITIONS — use these strictly:
- one_day_revision: fix without new experiments. Examples: missing related work citation,
  incomplete limitations discussion, unclear notation, weak motivation paragraph,
  missing ablation commentary on existing results.
- needs_experiment: fix requires running the model or collecting new data. Examples:
  missing baseline comparison, unvalidated quantitative claim, missing ablation study.
- needs_redesign: fundamental threat to validity. Examples: construct validity problem,
  selection bias in benchmark, methodology flaw that makes the main result uninterpretable.

FINDING COUNT CONSTRAINTS:
- Generate 4 to 7 findings. If more exist, prioritize the most impactful.
- At least 2 findings MUST be one_day_revision.
- Mark exactly one finding as "is_primary": true — the single issue most likely to cause rejection.

CONFIDENCE GUIDANCE:
- high: issue is clearly present (or absent) in the paper text you just read.
- medium: issue might be addressed in parts you could not verify.
- low: you are inferring a problem that may not exist.

When determining repair_cost: reference accepted review standards for what counts as sufficient
evidence in this area.
When marking confidence=high: if rejected papers were rejected for the same reason, that
confirms the finding is a real dealbreaker.

For findings with repair_cost=one_day_revision, also provide action_steps:
- 2 to 4 numbered steps describing exactly what the author should do tomorrow
- Each step must name a specific section, paragraph, equation, or table to edit
- Steps must be concrete enough that the author can act without re-reading the review
- Bad step: "improve the clarity of the method section"
- Good step: "In Section 3, add a sentence after Equation 2 explaining why cosine similarity was chosen over L2 distance, citing the ablation in Table 1"
For findings with repair_cost=needs_experiment or needs_redesign, leave action_steps as [].

Output ONLY valid JSON:
{{
  "findings": [
    {{
      "id": "F1",
      "related_contribution": "C1",
      "problem": "one sentence describing the specific issue, referencing paper content",
      "nature": "content_missing | expression_issue | design_flaw",
      "repair_cost": "one_day_revision | needs_experiment | needs_redesign",
      "confidence": "high | medium | low",
      "confidence_reason": "required when confidence != high",
      "is_primary": false,
      "action_steps": ["Step 1: ...", "Step 2: ..."]
    }}
  ],
  "executive_summary": "5 sentences: (1) is the core idea solid? (2) biggest risk (3) second risk (4) what to do first (5) overall prognosis"
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

    cluster_summary = ""
    if state.review_patterns:
        from ..schemas.tools import ClusterAnalysis
        if isinstance(state.review_patterns, ClusterAnalysis):
            cluster_summary = "\n".join(
                f"  - {c.label}: {c.criticism_pattern}"
                for c in state.review_patterns.clusters[:5]
            )
    cluster_summary = cluster_summary or "  (not available)"

    accepted_reviews_block = "\n---\n".join(state.accepted_review_excerpts[:3]) or "  (none available)"
    rejected_reviews_block = "\n---\n".join(state.rejected_review_excerpts[:3]) or "  (none available)"

    prompt = _DIAGNOSE_PROMPT.format(
        core_claim=model.core_claim or "(not extracted)",
        contributions_block=contributions_block,
        key_claims_block=key_claims_block,
        paper_text=state.paper_text,
        accepted_reviews_block=accepted_reviews_block,
        rejected_reviews_block=rejected_reviews_block,
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

FINDINGS TO VERIFY:
{findings_block}

TASKS:
1. For each finding listed, quote a passage from the section (max 60 words) that either
   supports or contradicts it. Give a verdict.
2. List writing issues in this section — typos, broken references, unclear sentences.
   Each issue MUST have an exact quote (max 20 words). Keep issue description under 20 words.
   Keep suggestion under 20 words. Report at most 3 writing issues.

Output ONLY valid JSON:
{{
  "evidence_updates": [
    {{
      "finding_id": "F1",
      "evidence_quote": "exact quote from section (max 60 words)",
      "verdict": "confirmed | refuted | inconclusive"
    }}
  ],
  "writing_issues": [
    {{
      "quote": "exact text (max 20 words)",
      "issue": "problem description (max 20 words)",
      "suggestion": "specific fix (max 20 words)"
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

    default_section_name = next(iter(state.paper_sections), "") if state.paper_sections else ""
    if "__default__" in section_to_findings and not default_section_name:
        del section_to_findings["__default__"]

    all_evidence: list[EvidenceUpdate] = []
    all_writing: list[WritingIssue] = []

    for section_name, sec_findings in section_to_findings.items():
        resolved = section_name if section_name != "__default__" else default_section_name

        # Prefer original section text; fallback to LLM summary with a note
        section_text = state.paper_sections.get(resolved, "")
        if not section_text:
            for key, val in state.paper_sections.items():
                if resolved.lower() in key.lower():
                    section_text = val
                    break
        using_summary_fallback = False
        if not section_text:
            section_text = model.section_summaries.get(resolved, "")
            if not section_text:
                for key, val in model.section_summaries.items():
                    if resolved.lower() in key.lower():
                        section_text = val
                        break
            using_summary_fallback = bool(section_text)
        if not section_text:
            section_text = "(section text not available)"

        findings_block = "\n".join(
            f"  {f.id}: {f.problem} [confidence={f.confidence}]" for f in sec_findings
        )
        if using_summary_fallback:
            findings_block += "\n  [NOTE: using LLM-generated summary as fallback — quotes may not match verbatim]"

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

    # Save and print log summary
    log_lines: list[str] = []
    if _run_logger:
        log_path = _run_logger.save()
        log_lines = _run_logger.summary_lines()
        log_lines.append(f"Log saved: {log_path}")

    messages = [AIMessage(content=f"Report assembled: {len(findings)} findings, domain={state.detected_domain}")]
    if log_lines:
        messages.append(AIMessage(content="\n".join(log_lines)))

    return {
        "report": report,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_compiled_graph: Any = None


def _timed_node(name: str, fn):
    """Wrap a node function with RunLogger timing."""
    def wrapper(state):
        if _run_logger:
            _run_logger.start_node(name)
        try:
            return fn(state)
        finally:
            if _run_logger:
                _run_logger.end_node()
    wrapper.__name__ = fn.__name__
    return wrapper


def build_diagnosis_graph():
    graph = StateGraph(DiagnosisState)
    graph.add_node("preprocess", _timed_node("preprocess", preprocess_node))
    graph.add_node("understand", _timed_node("understand", understand_node))
    graph.add_node("search", _timed_node("search", search_node))
    graph.add_node("review_analysis", _timed_node("review_analysis", review_analysis_node))
    graph.add_node("diagnose", _timed_node("diagnose", diagnose_node))
    graph.add_node("forensics", _timed_node("forensics", forensics_node))
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
