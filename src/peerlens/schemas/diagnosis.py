from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .tools import PaperResult


# ---------------------------------------------------------------------------
# Stage 1: structural paper model
# ---------------------------------------------------------------------------

class Contribution(BaseModel):
    id: str
    statement: str
    method_location: str = ""
    experiment_location: str = ""
    result_location: str = ""


class PaperModel(BaseModel):
    core_claim: str = ""
    contributions: list[Contribution] = Field(default_factory=list)
    section_summaries: dict[str, str] = Field(default_factory=dict)
    key_claims: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 2: findings
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    id: str
    related_contribution: str = ""
    problem: str
    nature: str       # "content_missing" | "expression_issue" | "design_flaw"
    repair_cost: str  # "one_day_revision" | "needs_experiment" | "needs_redesign"
    confidence: str   # "high" | "medium" | "low"
    confidence_reason: str = ""


# ---------------------------------------------------------------------------
# Stage 3: forensics
# ---------------------------------------------------------------------------

class EvidenceUpdate(BaseModel):
    finding_id: str
    evidence_quote: str = ""
    verdict: str  # "confirmed" | "refuted" | "inconclusive"


class WritingIssue(BaseModel):
    quote: str = ""
    issue: str
    suggestion: str = ""


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

class DiagnosisReport(BaseModel):
    detected_domain: str
    detected_keywords: list[str] = Field(default_factory=list)
    executive_summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    evidence_updates: list[EvidenceUpdate] = Field(default_factory=list)
    writing_issues: list[WritingIssue] = Field(default_factory=list)
    similar_accepted: list[PaperResult] = Field(default_factory=list)
    similar_rejected: list[PaperResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class DiagnosisState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    paper_text: str = ""
    target_venue: str = ""
    # Stage 0 outputs
    noise_log: list[str] = Field(default_factory=list)
    is_chunked: bool = False
    # Stage 1 outputs (also used by search/review_analysis nodes)
    detected_domain: str = ""
    detected_keywords: list[str] = Field(default_factory=list)
    paper_abstract_clean: str = ""
    paper_sections: dict[str, str] = Field(default_factory=dict)
    paper_model: Optional[PaperModel] = None
    # search + review_analysis outputs
    similar_accepted: list[PaperResult] = Field(default_factory=list)
    similar_rejected: list[PaperResult] = Field(default_factory=list)
    review_patterns: Optional[object] = None
    retrieved_reviews: Optional[dict] = None
    # Stage 2 outputs
    findings: list[Finding] = Field(default_factory=list)
    executive_summary: str = ""
    # Stage 3 outputs
    evidence_updates: list[EvidenceUpdate] = Field(default_factory=list)
    writing_issues_stage3: list[WritingIssue] = Field(default_factory=list)
    # Final
    report: Optional[DiagnosisReport] = None
    memory_context: Optional[str] = None
    active_node: str = "preprocess"
    tool_call_count: int = 0
    iteration: int = 0
