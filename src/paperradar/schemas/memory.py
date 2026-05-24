from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class EpisodicEvent(BaseModel):
    event_id: Optional[int] = None
    event_type: Literal["query", "view", "like", "dislike", "report_generated", "push_sent"]
    content: str
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    feedback: Optional[Literal["up", "down"]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = ""


class UserPreferenceEmbedding(BaseModel):
    doc_id: str
    text: str
    source: Literal["liked_paper", "liked_topic", "query_history"]
    paper_id: Optional[str] = None
    timestamp: str = ""


class AgentSession(BaseModel):
    session_id: str
    agent_type: Literal["diagnosis", "research", "reading"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_summary: str
    output_summary: str
    key_findings: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    full_report_path: str = ""
