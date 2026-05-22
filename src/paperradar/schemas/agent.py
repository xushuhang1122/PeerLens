from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .tools import (
    AnalysisReport,
    ClusterAnalysis,
    GapReport,
    SearchPapersOutput,
    TemporalAnalysis,
    UserProfile,
)


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    raw_query: str = ""
    parsed_intent: Optional[dict] = None

    search_results: Optional[SearchPapersOutput] = None
    retrieved_reviews: Optional[dict] = None

    temporal_analysis: Optional[TemporalAnalysis] = None
    gap_report: Optional[GapReport] = None
    cluster_analysis: Optional[ClusterAnalysis] = None

    user_profile: Optional[UserProfile] = None

    report: Optional[AnalysisReport] = None

    tool_call_count: int = 0
    error_log: list[str] = Field(default_factory=list)
    iteration: int = 0
    # tracks which node last invoked tools so ToolNode can route back correctly
    active_node: str = "retrieve"
