from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# search_papers
# ---------------------------------------------------------------------------

class SearchPapersInput(BaseModel):
    query: str = Field(..., description="Natural language search query")
    decision_filter: Optional[list[str]] = Field(
        None, description="oral/spotlight/poster/accepted/rejected"
    )
    conference_filter: Optional[list[str]] = Field(None)
    year_filter: Optional[list[int]] = Field(None)
    top_k: int = Field(default=20, ge=1, le=100)


class PaperResult(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]
    primary_area: str
    venue: str
    decision: str
    forum_url: str
    year: int
    conference: str
    rrf_score: float
    bm25_rank: Optional[int] = None
    content_vec_rank: Optional[int] = None
    review_vec_rank: Optional[int] = None


class SearchPapersOutput(BaseModel):
    results: list[PaperResult]
    total_found: int
    query: str


# ---------------------------------------------------------------------------
# get_paper_reviews
# ---------------------------------------------------------------------------

class GetPaperReviewsInput(BaseModel):
    paper_ids: list[str] = Field(..., description="OpenReview paper IDs")


class ReviewResult(BaseModel):
    paper_id: str
    paper_title: str
    reviews: list[dict]
    avg_rating: Optional[float]
    avg_confidence: Optional[float]
    decision: str


class GetPaperReviewsOutput(BaseModel):
    results: list[ReviewResult]


# ---------------------------------------------------------------------------
# cluster_reviews
# ---------------------------------------------------------------------------

class ClusterReviewsInput(BaseModel):
    primary_area: str = Field(..., description="Research area to cluster")
    n_clusters: int = Field(default=5, ge=2, le=20)


class ClusterInfo(BaseModel):
    cluster_id: int
    label: str
    top_terms: list[str]
    representative_quotes: list[str]
    paper_count: int
    avg_rating: Optional[float]
    criticism_pattern: str


class ClusterAnalysis(BaseModel):
    primary_area: str
    n_clusters: int
    clusters: list[ClusterInfo]
    high_frequency_rejections: list[str]


# ---------------------------------------------------------------------------
# analyze_temporal_distribution
# ---------------------------------------------------------------------------

class AnalyzeTemporalInput(BaseModel):
    topic: str
    conferences: list[str] = Field(default=["NeurIPS", "ICML", "ICLR"])
    years: list[int] = Field(default=[2022, 2023, 2024, 2025])


class YearConferenceBucket(BaseModel):
    conference: str
    year: int
    paper_count: int
    oral_count: int
    spotlight_count: int
    poster_count: int
    top_papers: list[str]


class TemporalAnalysis(BaseModel):
    topic: str
    distribution: list[YearConferenceBucket]
    trend: str
    peak_year: Optional[int]
    peak_conference: Optional[str]
    summary: str


# ---------------------------------------------------------------------------
# identify_research_gaps
# ---------------------------------------------------------------------------

class IdentifyGapsInput(BaseModel):
    domain: str = Field(..., description="Research domain to analyze")
    min_cluster_density: float = Field(
        default=0.1, description="Min relative density to count as covered"
    )


class ResearchGap(BaseModel):
    gap_description: str
    evidence: list[str]
    suggested_angle: str


class GapReport(BaseModel):
    domain: str
    gaps: list[ResearchGap]
    covered_areas: list[str]
    sparse_areas: list[str]
    rejection_patterns: list[str]
    submission_advice: str


# ---------------------------------------------------------------------------
# get_user_context
# ---------------------------------------------------------------------------

class GetUserContextInput(BaseModel):
    pass


class UserProfile(BaseModel):
    recent_queries: list[str]
    liked_paper_ids: list[str]
    liked_topics: list[str]
    disliked_topics: list[str]
    preference_vector_summary: str


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query to find information about new conference results")


class WebSearchOutput(BaseModel):
    results: list[dict]
    query: str


# ---------------------------------------------------------------------------
# check_and_crawl_new_conference
# ---------------------------------------------------------------------------

class CheckAndCrawlInput(BaseModel):
    conference: str = Field(..., description="Conference name, e.g. NeurIPS, ICLR, ICML")
    year: int = Field(..., description="Conference year")
    decisions: Optional[list[str]] = Field(
        None, description="Decision types to crawl. None means all available."
    )


class CheckAndCrawlOutput(BaseModel):
    conference: str
    year: int
    already_existed: bool
    crawl_started: bool
    paper_count: int
    message: str


# ---------------------------------------------------------------------------
# discover_conference
# ---------------------------------------------------------------------------

class DiscoverConferenceInput(BaseModel):
    venue_id: str = Field(
        ...,
        description=(
            "OpenReview venue_id to probe, e.g. 'aclweb.org/ACL/2024/Conference'. "
            "Use {year} placeholder or a specific year."
        ),
    )


class ConferenceFieldInfo(BaseModel):
    venue_id: str
    paper_count: int
    content_fields: list[str]
    decision_patterns: list[str]
    review_invitation_pattern: Optional[str]
    sample_titles: list[str]
    filterable_summary: str


class DiscoverConferenceOutput(BaseModel):
    found: bool
    venue_id: str
    info: Optional[ConferenceFieldInfo] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class GenerateReportInput(BaseModel):
    findings: dict


class AnalysisReport(BaseModel):
    title: str
    executive_summary: str
    search_results: Optional[SearchPapersOutput] = None
    temporal_analysis: Optional[TemporalAnalysis] = None
    gap_report: Optional[GapReport] = None
    cluster_analysis: Optional[ClusterAnalysis] = None
    personalized_recommendations: list[PaperResult] = []
    generated_at: datetime = Field(default_factory=datetime.utcnow)
