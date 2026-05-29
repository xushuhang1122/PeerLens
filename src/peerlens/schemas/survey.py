from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .tools import PaperResult, SearchPapersOutput
from .tools import TemporalAnalysis, GapReport


class SurveySection(BaseModel):
    heading: str
    content: str
    cited_paper_ids: list[str] = Field(default_factory=list)


class SurveyReport(BaseModel):
    title: str
    background: str
    key_papers: list[PaperResult] = Field(default_factory=list)
    sections: list[SurveySection] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    submission_advice: str = ""
    used_training_data: bool = False
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchAgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    raw_query: str = ""
    refined_query: str = ""
    focus: dict = Field(default_factory=dict)
    search_results: Optional[SearchPapersOutput] = None
    temporal_analysis: Optional[TemporalAnalysis] = None
    gap_report: Optional[GapReport] = None
    survey_report: Optional[SurveyReport] = None
    memory_context: Optional[str] = None
    active_node: str = "refine"
    tool_call_count: int = 0
    iteration: int = 0
    error_log: list[str] = Field(default_factory=list)
