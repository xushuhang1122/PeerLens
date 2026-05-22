from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

DecisionType = Literal["oral", "spotlight", "poster", "accepted", "rejected", "unknown"]


class Paper(BaseModel):
    id: str
    number: Optional[int] = None
    title: str
    authors: list[str] = []
    abstract: str = ""
    keywords: list[str] = []
    primary_area: str = ""
    venue: str = ""
    decision: DecisionType = "unknown"
    forum_url: str = ""
    conference: str = ""
    year: int = 0
    crawled_at: datetime = Field(default_factory=datetime.utcnow)


class Review(BaseModel):
    id: str
    forum_id: str
    paper_id: str
    reviewer_id: str = ""
    rating: Optional[float] = None
    confidence: Optional[float] = None
    summary: str = ""
    soundness: str = ""
    presentation: str = ""
    contribution: str = ""
    strengths: str = ""
    weaknesses: str = ""
    questions: str = ""
    full_text: str = ""
    invitation: str = ""


class DecisionNote(BaseModel):
    id: str
    forum_id: str
    paper_id: str
    decision: str = ""
    comment: str = ""
    invitation: str = ""
