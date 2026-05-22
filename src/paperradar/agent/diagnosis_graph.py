from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from ..config import settings
from ..schemas.diagnosis import (
    DiagnosisReport,
    DiagnosisSuggestion,
    DiagnosisState,
    SimulatedReview,
)
from ..schemas.tools import PaperResult

_llm = ChatOpenAI(
    model=settings.llm.model,
    temperature=0.2,
    api_key=settings.llm.openai_api_key,
    **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
)


# ------------------------------------------------------------------
# Nodes — direct tool invocation, no ToolNode
# ------------------------------------------------------------------

def detect_node(state: DiagnosisState) -> dict[str, Any]:
    paper_text = state.paper_text[:3000]
    prompt = f"""Analyze this academic paper text and extract:
1. The primary research domain (concise English phrase, e.g. "efficient transformers", "reinforcement learning from human feedback")
2. 5-8 English keywords that capture the topic precisely
3. A clean version of the abstract (first 300 words of the abstract section if present)

Paper text:
{paper_text}

Respond with ONLY a JSON object:
{{
  "detected_domain": "...",
  "detected_keywords": ["kw1", "kw2", ...],
  "paper_abstract_clean": "..."
}}
"""
    response = _llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    detected_domain = ""
    detected_keywords: list[str] = []
    paper_abstract_clean = ""
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])
        detected_domain = data.get("detected_domain", "")
        detected_keywords = data.get("detected_keywords", [])
        paper_abstract_clean = data.get("paper_abstract_clean", "")
    except Exception:
        detected_domain = "machine learning"

    return {
        "detected_domain": detected_domain,
        "detected_keywords": detected_keywords,
        "paper_abstract_clean": paper_abstract_clean,
        "messages": [AIMessage(content=f"Detected domain: {detected_domain}")],
        "iteration": state.iteration + 1,
    }


def search_node(state: DiagnosisState) -> dict[str, Any]:
    from .tools import search_papers

    domain = state.detected_domain
    keywords = " ".join(state.detected_keywords[:6])
    query = f"{domain} {keywords}".strip()

    updates: dict[str, Any] = {"iteration": state.iteration + 1}
    msgs: list[Any] = []

    try:
        result = search_papers.invoke({
            "query": query,
            "top_k": 20,
        })
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


def review_analysis_node(state: DiagnosisState) -> dict[str, Any]:
    from .tools import cluster_reviews, get_paper_reviews

    domain = state.detected_domain
    updates: dict[str, Any] = {"iteration": state.iteration + 1}
    msgs: list[Any] = []

    try:
        patterns = cluster_reviews.invoke({
            "primary_area": domain,
            "n_clusters": 5,
        })
        updates["review_patterns"] = patterns
        msgs.append(AIMessage(content=f"Cluster analysis: {len(patterns.clusters)} clusters found"))
    except Exception as e:
        msgs.append(AIMessage(content=f"Cluster analysis skipped: {e}"))

    accepted_ids = [p.paper_id for p in state.similar_accepted[:5]]
    rejected_ids = [p.paper_id for p in state.similar_rejected[:5]]
    paper_ids = accepted_ids + rejected_ids

    if paper_ids:
        try:
            reviews = get_paper_reviews.invoke({"paper_ids": paper_ids})
            updates["retrieved_reviews"] = {"count": len(reviews.results)}
            msgs.append(AIMessage(content=f"Retrieved reviews for {len(reviews.results)} papers"))
        except Exception as e:
            msgs.append(AIMessage(content=f"Review retrieval skipped: {e}"))

    updates["messages"] = msgs
    return updates


_VENUE_CRITERIA: dict[str, str] = {
    "neurips": (
        "NeurIPS review criteria:\n"
        "- Scoring: overall 1-10 (4=reject, 5=borderline reject, 6=borderline accept, 8=accept, 10=strong accept)\n"
        "- Confidence: 1-5\n"
        "- Sub-scores: soundness 1-4, presentation 1-4, contribution 1-4\n"
        "- Key emphasis: technical soundness, novelty of contribution, empirical rigor, clarity of exposition\n"
        "- NeurIPS especially values: rigorous theoretical or empirical grounding, honest limitations section, reproducibility checklist\n"
        "- Common rejection reasons: incremental contribution, missing baselines, overclaimed results, poor writing"
    ),
    "icml": (
        "ICML review criteria:\n"
        "- Scoring: overall 1-10 (same scale as NeurIPS)\n"
        "- Confidence: 1-5\n"
        "- Sub-scores: soundness 1-4, presentation 1-4, contribution 1-4\n"
        "- Key emphasis: originality, significance, correctness, clarity\n"
        "- ICML especially values: clear problem motivation, strong experimental evaluation with ablations, connections to theory\n"
        "- Common rejection reasons: limited novelty over prior work, weak experimental comparison, unclear writing"
    ),
    "iclr": (
        "ICLR review criteria:\n"
        "- Scoring: 1-10 (1=strong reject, 3=reject, 5=marginally below threshold, 6=marginally above threshold, 8=accept, 10=strong accept)\n"
        "- Confidence: 1-5\n"
        "- Sub-scores: soundness 1-4, presentation 1-4, contribution 1-4\n"
        "- Key emphasis: reproducibility, open science, representation learning and deep learning\n"
        "- ICLR especially values: code/data release, ablation studies, scaling analysis, failure case analysis\n"
        "- Common rejection reasons: no code release, missing ablations, overclaiming, poor reproducibility"
    ),
    "aistats": (
        "AISTATS review criteria:\n"
        "- Scoring: 1-10 with sub-scores for quality, clarity, originality, significance (1-4 each)\n"
        "- Confidence: 1-5\n"
        "- Key emphasis: statistical machine learning, probabilistic methods, algorithms with theoretical guarantees\n"
        "- AISTATS especially values: theoretical depth, connections between statistics and ML\n"
        "- Common rejection reasons: lack of theory, pure empirical work without statistical insight"
    ),
    "uai": (
        "UAI review criteria:\n"
        "- Scoring: 1-10\n"
        "- Key emphasis: uncertainty, probabilistic graphical models, Bayesian methods, causal inference\n"
        "- UAI especially values: principled uncertainty quantification, causal reasoning, exact or approximate inference contributions\n"
        "- Common rejection reasons: ignoring uncertainty, no comparison to Bayesian baselines"
    ),
    "jmlr": (
        "JMLR review criteria (journal format):\n"
        "- No numeric scores; qualitative recommendation: accept / major revision / minor revision / reject\n"
        "- Key emphasis: completeness, theoretical rigor, comprehensive related work, reproducibility\n"
        "- JMLR especially values: thorough proofs, extensive experiments, complete literature review\n"
        "- Common rejection reasons: incomplete theoretical analysis, insufficient experiments, missing proofs"
    ),
    "emnlp": (
        "EMNLP review criteria:\n"
        "- Scoring: overall 1-5 (1=strong reject, 3=borderline, 5=strong accept)\n"
        "- Confidence: 1-5\n"
        "- Sub-scores: soundness/correctness 1-5, excitement/interest 1-5, reproducibility 1-4, dataset/resource release\n"
        "- Key emphasis: NLP tasks, evaluation on standard benchmarks, dataset quality, linguistic analysis\n"
        "- Common rejection reasons: no human evaluation, poor baseline selection, missing error analysis"
    ),
    "acl": (
        "ACL/ACL Rolling Review criteria:\n"
        "- Scoring: 1-5 via ARR (overall recommendation + paper scores)\n"
        "- Confidence: 1-5\n"
        "- Key emphasis: linguistic grounding, evaluation on established NLP benchmarks, clear research questions\n"
        "- ACL especially values: human evaluation, reproducibility, broad impact on NLP\n"
        "- Common rejection reasons: narrow scope, missing ablations, no significance testing"
    ),
    "cvpr": (
        "CVPR review criteria:\n"
        "- Scoring: 1-6 (1=strong reject, 3=borderline, 5=strong accept)\n"
        "- Confidence: 1-5\n"
        "- Key emphasis: computer vision tasks, SOTA performance on standard benchmarks, visual quality\n"
        "- CVPR especially values: new datasets/benchmarks, strong empirical performance, clear visual comparisons\n"
        "- Common rejection reasons: no comparison to recent SOTA, missing ablations, poor figures/visualization"
    ),
    "eccv": (
        "ECCV review criteria:\n"
        "- Scoring: 1-6 (similar to CVPR)\n"
        "- Key emphasis: computer vision, novel methods with clear improvements\n"
        "- Common rejection reasons: incremental over published work, missing standard benchmarks"
    ),
    "aaai": (
        "AAAI review criteria:\n"
        "- Scoring: 1-7 (1=strong reject, 4=neutral, 7=strong accept)\n"
        "- Confidence: 1-5\n"
        "- Key emphasis: broad AI topics, clear problem definition, strong evaluation\n"
        "- Common rejection reasons: vague problem formulation, weak experimental setup"
    ),
}


def _get_venue_criteria(target_venue: str) -> str:
    if not target_venue:
        return ""
    v = target_venue.lower()
    for key, criteria in _VENUE_CRITERIA.items():
        if key in v:
            return criteria
    return (
        f"Target venue: {target_venue}\n"
        "Use your knowledge of this venue's review criteria, scoring scale, and typical reviewer expectations. "
        "Adapt the scoring scale and emphasis areas accordingly."
    )


def diagnose_node(state: DiagnosisState) -> dict[str, Any]:
    domain = state.detected_domain
    abstract = state.paper_abstract_clean or state.paper_text[:800]
    keywords = state.detected_keywords
    target_venue = state.target_venue

    accepted_summary = ""
    if state.similar_accepted:
        accepted_summary = "\n".join(
            f"- {p.title} ({p.conference} {p.year}): {p.abstract[:200]}"
            for p in state.similar_accepted[:5]
        )

    rejected_summary = ""
    if state.similar_rejected:
        rejected_summary = "\n".join(
            f"- {p.title} ({p.conference} {p.year}): {p.abstract[:200]}"
            for p in state.similar_rejected[:5]
        )

    cluster_summary = ""
    if state.review_patterns:
        from ..schemas.tools import ClusterAnalysis
        if isinstance(state.review_patterns, ClusterAnalysis):
            cluster_summary = "\n".join(
                f"- Cluster '{c.label}': {c.criticism_pattern} (avg rating: {c.avg_rating})"
                for c in state.review_patterns.clusters[:5]
            )

    venue_criteria = _get_venue_criteria(target_venue)
    venue_section = ""
    if venue_criteria:
        venue_section = f"\n== TARGET VENUE ==\n{venue_criteria}\n"

    if target_venue:
        score_instructions = (
            f"Use the scoring scale appropriate for {target_venue}. "
            "Adapt overall_scale, confidence_scale, soundness_scale, presentation_scale, contribution_scale "
            "to match this venue's actual review form. "
            "Set score_interpretation to explain what the overall_score means at this venue "
            "(e.g. '5/10 = borderline reject at NeurIPS')."
        )
    else:
        score_instructions = (
            "Use standard ML conference scoring: overall 1-10 (5=borderline), "
            "soundness/presentation/contribution 1-4, confidence 1-5."
        )

    prompt = f"""You are an expert academic reviewer simulating a peer review for an ML/AI paper.
{venue_section}
Paper domain: {domain}
Paper keywords: {', '.join(keywords)}
Paper abstract:
{abstract}

Similar ACCEPTED papers in this area:
{accepted_summary or "(none found in local database)"}

Similar REJECTED papers in this area:
{rejected_summary or "(none found in local database)"}

Reviewer concern clusters in this domain:
{cluster_summary or "(none found)"}

Produce a detailed diagnosis. Output ONLY this JSON:
{{
  "key_reviewer_concerns": ["concern 1", "concern 2", "concern 3"],
  "acceptance_patterns": ["pattern 1", "pattern 2", "pattern 3"],
  "rejection_patterns": ["pattern 1", "pattern 2", "pattern 3"],
  "suggestions": [
    {{
      "aspect": "novelty",
      "reviewer_comment": "The paper claims X but does not sufficiently differentiate from prior work Y.",
      "suggestion": "Add a dedicated comparison section showing concrete improvements over Y in 3 metrics.",
      "priority": "critical"
    }}
  ],
  "overall_assessment": "2-3 sentence overall assessment of where this paper stands in the field.",
  "simulated_review": {{
    "venue": "{target_venue or 'general ML conference'}",
    "overall_score": 5,
    "overall_scale": "1-10",
    "confidence": 3,
    "confidence_scale": "1-5",
    "soundness": 3,
    "soundness_scale": "1-4",
    "presentation": 3,
    "presentation_scale": "1-4",
    "contribution": 2,
    "contribution_scale": "1-4",
    "score_interpretation": "5/10 = borderline at this venue",
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
    "questions": ["clarification question 1", "clarification question 2"],
    "summary": "2-3 sentence reviewer verdict."
  }}
}}

Requirements:
- Include 4-6 suggestions covering novelty, experiments, clarity, related work, reproducibility, theory.
- Each suggestion must include a specific reviewer_comment in the style that reviewer would actually write.
- Priorities: critical = must fix, important = strongly recommended, minor = nice to have.
- {score_instructions}
- Make the simulated_review reflect the specific standards and expectations of the target venue.
"""
    response = _llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    report: DiagnosisReport
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])

        suggestions = [DiagnosisSuggestion(**s) for s in data.get("suggestions", [])]

        sim_rev_data = data.get("simulated_review")
        simulated_review = SimulatedReview(**sim_rev_data) if sim_rev_data else None

        report = DiagnosisReport(
            detected_domain=domain,
            detected_keywords=keywords,
            similar_accepted=state.similar_accepted[:5],
            similar_rejected=state.similar_rejected[:5],
            key_reviewer_concerns=data.get("key_reviewer_concerns", []),
            acceptance_patterns=data.get("acceptance_patterns", []),
            rejection_patterns=data.get("rejection_patterns", []),
            suggestions=suggestions,
            overall_assessment=data.get("overall_assessment", ""),
            simulated_review=simulated_review,
            generated_at=datetime.utcnow(),
        )
    except Exception:
        report = DiagnosisReport(
            detected_domain=domain,
            detected_keywords=keywords,
            similar_accepted=state.similar_accepted[:5],
            similar_rejected=state.similar_rejected[:5],
            overall_assessment=content[:500],
            generated_at=datetime.utcnow(),
        )

    return {
        "report": report,
        "messages": [AIMessage(content=f"Diagnosis complete for domain: {domain}")],
    }


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

_compiled_graph: Any = None


def build_diagnosis_graph():
    graph = StateGraph(DiagnosisState)
    graph.add_node("detect", detect_node)
    graph.add_node("search", search_node)
    graph.add_node("review_analysis", review_analysis_node)
    graph.add_node("diagnose", diagnose_node)

    graph.set_entry_point("detect")
    graph.add_edge("detect", "search")
    graph.add_edge("search", "review_analysis")
    graph.add_edge("review_analysis", "diagnose")
    graph.add_edge("diagnose", END)

    return graph.compile()


def get_diagnosis_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_diagnosis_graph()
    return _compiled_graph
