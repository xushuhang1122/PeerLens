from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .tools import PaperResult


class DiagnosisSuggestion(BaseModel):
    aspect: str
    reviewer_comment: str = ""
    suggestion: str
    priority: str  # "critical" | "important" | "minor"


class SimulatedReview(BaseModel):
    venue: str = ""
    # Verbal recommendation label, optionally with venue-specific score.
    # e.g. "Weak Accept", "Reject", "6/10 — marginally above threshold at ICLR"
    recommendation: str
    confidence: int
    confidence_scale: str = "1-5"
    soundness: int
    soundness_scale: str = "1-4"
    presentation: int
    presentation_scale: str = "1-4"
    contribution: int
    contribution_scale: str = "1-4"
    score_interpretation: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    summary: str = ""


class DiagnosisReport(BaseModel):
    detected_domain: str
    detected_keywords: list[str] = Field(default_factory=list)
    similar_accepted: list[PaperResult] = Field(default_factory=list)
    similar_rejected: list[PaperResult] = Field(default_factory=list)
    key_reviewer_concerns: list[str] = Field(default_factory=list)
    acceptance_patterns: list[str] = Field(default_factory=list)
    rejection_patterns: list[str] = Field(default_factory=list)
    suggestions: list[DiagnosisSuggestion] = Field(default_factory=list)
    overall_assessment: str = ""
    simulated_review: Optional[SimulatedReview] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class DiagnosisState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    paper_text: str = ""
    target_venue: str = ""
    detected_domain: str = ""
    detected_keywords: list[str] = Field(default_factory=list)
    paper_abstract_clean: str = ""
    similar_accepted: list[PaperResult] = Field(default_factory=list)
    similar_rejected: list[PaperResult] = Field(default_factory=list)
    review_patterns: Optional[object] = None
    retrieved_reviews: Optional[dict] = None
    report: Optional[DiagnosisReport] = None
    memory_context: Optional[str] = None
    active_node: str = "detect"
    tool_call_count: int = 0
    iteration: int = 0
