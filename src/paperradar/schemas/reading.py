from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class MemoryConnection(BaseModel):
    agent_type: Literal["diagnosis", "research", "reading"]
    session_id: str
    timestamp: datetime
    connection_description: str
    related_input_summary: str
    relevance_score: float = 0.0


class ReviewerPerspective(BaseModel):
    reviewer_id: str
    stance: Literal["positive", "negative", "mixed"]
    key_points: list[str] = Field(default_factory=list)


class ReadingReport(BaseModel):
    paper_title: str
    authors: list[str] = Field(default_factory=list)
    venue: str = ""
    source_url: str = ""
    tldr: str
    problem_statement: str
    core_contributions: list[str] = Field(default_factory=list)
    methodology_summary: str = ""
    key_innovations: list[str] = Field(default_factory=list)
    datasets_and_baselines: str = ""
    main_results: str = ""
    ablations: str = ""
    limitations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    reviewer_perspectives: list[ReviewerPerspective] = Field(default_factory=list)
    memory_connections: list[MemoryConnection] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ReadingState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    input_mode: Literal["pdf", "openreview_url", "arxiv_url", "topic_query_selected"] = "pdf"
    pdf_bytes: Optional[bytes] = None
    url: str = ""
    paper_text: str = ""
    paper_title: str = ""
    paper_id: str = ""
    paper_authors: list[str] = Field(default_factory=list)
    paper_venue: str = ""
    source_url: str = ""
    paper_reviews: list[dict] = Field(default_factory=list)
    memory_context: Optional[str] = None
    report: Optional[ReadingReport] = None
    discussion_active: bool = False
    active_node: str = "parse_input"
    iteration: int = 0
