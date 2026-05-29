from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool
from openai import OpenAI

from ..analysis.clustering import ReviewClusterer
from ..analysis.gap_detector import GapDetector
from ..analysis.temporal import TemporalAnalyzer
from ..config import settings
from ..crawl.discover import discover_venue
from ..crawl.pipeline import CrawlPipeline, is_crawl_running
from ..memory.episodic import EpisodicMemory
from ..memory.semantic import SemanticMemory
from ..retrieval.hybrid_search import HybridSearcher
from ..schemas.tools import (
    AnalysisReport,
    AnalyzeTemporalInput,
    CheckAndCrawlInput,
    CheckAndCrawlOutput,
    ClusterAnalysis,
    ClusterReviewsInput,
    DiscoverConferenceInput,
    DiscoverConferenceOutput,
    GapReport,
    GenerateReportInput,
    GetPaperReviewsInput,
    GetPaperReviewsOutput,
    GetUserContextInput,
    IdentifyGapsInput,
    ReviewResult,
    SearchPapersInput,
    SearchPapersOutput,
    TemporalAnalysis,
    UserProfile,
    WebSearchInput,
    WebSearchOutput,
)

_searcher: HybridSearcher | None = None
_temporal: TemporalAnalyzer | None = None
_clusterer: ReviewClusterer | None = None
_gap: GapDetector | None = None
_episodic: EpisodicMemory | None = None
_semantic: SemanticMemory | None = None
_pipeline: CrawlPipeline | None = None
_llm: OpenAI | None = None


def _get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


def _get_temporal() -> TemporalAnalyzer:
    global _temporal
    if _temporal is None:
        _temporal = TemporalAnalyzer()
    return _temporal


def _get_clusterer() -> ReviewClusterer:
    global _clusterer
    if _clusterer is None:
        _clusterer = ReviewClusterer()
    return _clusterer


def _get_gap() -> GapDetector:
    global _gap
    if _gap is None:
        _gap = GapDetector()
    return _gap


def _get_episodic() -> EpisodicMemory:
    global _episodic
    if _episodic is None:
        _episodic = EpisodicMemory()
    return _episodic


def _get_semantic() -> SemanticMemory:
    global _semantic
    if _semantic is None:
        _semantic = SemanticMemory()
    return _semantic


def _get_pipeline() -> CrawlPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CrawlPipeline()
    return _pipeline


def _get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        _llm = OpenAI(
            api_key=settings.llm.openai_api_key,
            **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
        )
    return _llm


@tool(args_schema=SearchPapersInput)
def search_papers(
    query: str,
    decision_filter: Optional[list[str]] = None,
    conference_filter: Optional[list[str]] = None,
    year_filter: Optional[list[int]] = None,
    top_k: int = 20,
) -> SearchPapersOutput:
    """Search academic papers using hybrid BM25 + vector retrieval with optional decision/venue filters."""
    inp = SearchPapersInput(
        query=query,
        decision_filter=decision_filter,
        conference_filter=conference_filter,
        year_filter=year_filter,
        top_k=top_k,
    )
    return _get_searcher().search(inp)


@tool(args_schema=GetPaperReviewsInput)
def get_paper_reviews(paper_ids: list[str]) -> GetPaperReviewsOutput:
    """Fetch stored review data for given paper IDs from the local database."""
    from ..store.chroma import ChromaManager
    chroma = ChromaManager()  # returns singleton
    results: list[ReviewResult] = []
    for pid in paper_ids:
        data = chroma.get_all_review_embeddings(where={"paper_id": pid})
        metas = data.get("metadatas") or []
        docs = data.get("documents") or []
        if metas:
            meta = metas[0]
            results.append(
                ReviewResult(
                    paper_id=pid,
                    paper_title=meta.get("title", ""),
                    reviews=[{"text": docs[0]} if docs else {}],
                    avg_rating=meta.get("avg_rating"),
                    avg_confidence=None,
                    decision=meta.get("decision", ""),
                )
            )
    return GetPaperReviewsOutput(results=results)


@tool(args_schema=ClusterReviewsInput)
def cluster_reviews(primary_area: str, n_clusters: int = 5) -> ClusterAnalysis:
    """Cluster review comments for a research area using K-Means to identify common criticism patterns."""
    return _get_clusterer().cluster(ClusterReviewsInput(primary_area=primary_area, n_clusters=n_clusters))


@tool(args_schema=AnalyzeTemporalInput)
def analyze_temporal_distribution(
    topic: str,
    conferences: list[str] = ["NeurIPS", "ICML", "ICLR"],
    years: list[int] = [2022, 2023, 2024, 2025],
) -> TemporalAnalysis:
    """Analyze how a research topic's presence evolves over years and conferences."""
    return _get_temporal().analyze(
        AnalyzeTemporalInput(topic=topic, conferences=conferences, years=years)
    )


@tool(args_schema=IdentifyGapsInput)
def identify_research_gaps(
    domain: str, min_cluster_density: float = 0.1
) -> GapReport:
    """Identify under-explored research areas and common rejection patterns in a domain."""
    return _get_gap().detect(IdentifyGapsInput(domain=domain, min_cluster_density=min_cluster_density))


@tool(args_schema=GetUserContextInput)
def get_user_context() -> UserProfile:
    """Load the user's research history, liked papers, and semantic preference summary."""
    recent_queries = _get_episodic().get_recent_queries(20)
    liked_ids = _get_episodic().get_liked_paper_ids(50)
    disliked_ids = _get_episodic().get_disliked_paper_ids(20)
    pref_summary = _get_semantic().get_preference_summary()
    return UserProfile(
        recent_queries=recent_queries,
        liked_paper_ids=liked_ids,
        liked_topics=[],
        disliked_topics=[],
        preference_vector_summary=pref_summary,
    )


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> WebSearchOutput:
    """Search the web for information about new conference paper releases or current events.
    Returns a best-effort answer synthesized by the LLM based on its training knowledge,
    or a real search result if the configured endpoint supports it."""
    from langchain_openai import ChatOpenAI
    from ..config import settings as _s
    chat = ChatOpenAI(
        model=_s.llm.model, temperature=0.3, api_key=_s.llm.openai_api_key,
        **({"base_url": _s.llm.base_url} if _s.llm.base_url else {}),
    )
    resp = chat.invoke(
        f"Answer this web search query as accurately as possible, "
        f"focusing on ML conference announcements and paper releases:\n\n{query}"
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return WebSearchOutput(results=[{"text": content}], query=query)


@tool(args_schema=CheckAndCrawlInput)
def check_and_crawl_new_conference(
    conference: str,
    year: int,
    decisions: Optional[list[str]] = None,
) -> CheckAndCrawlOutput:
    """Check if a conference/year is in the local database. If not, start async crawl and indexing."""
    already_exists = _get_pipeline().check_local(conference, year)
    if already_exists:
        return CheckAndCrawlOutput(
            conference=conference,
            year=year,
            already_existed=True,
            crawl_started=False,
            paper_count=0,
            message=f"{conference} {year} already in local database.",
        )

    if is_crawl_running(conference, year):
        return CheckAndCrawlOutput(
            conference=conference,
            year=year,
            already_existed=False,
            crawl_started=False,
            paper_count=0,
            message=f"Crawl for {conference} {year} is already in progress.",
        )

    def _on_complete(count: int) -> None:
        _get_episodic().log_crawl(conference, year, None, count, "success")

    _get_pipeline().run_async(
        conference=conference,
        year=year,
        decision=decisions[0] if decisions and len(decisions) == 1 else None,
        on_complete=_on_complete,
    )

    return CheckAndCrawlOutput(
        conference=conference,
        year=year,
        already_existed=False,
        crawl_started=True,
        paper_count=0,
        message=(
            f"Crawl for {conference} {year} started in background. "
            "Results will be available once indexing completes."
        ),
    )


@tool(args_schema=GenerateReportInput)
def generate_report(findings: dict) -> AnalysisReport:
    """Synthesize all collected findings into a structured analysis report with citations."""
    prompt = (
        "You are a research assistant. Generate a structured analysis report based on these findings:\n"
        f"{json.dumps(findings, indent=2, default=str)}\n\n"
        "Return JSON with keys: title, executive_summary. "
        "Be concise, cite specific papers when available."
    )
    resp = _get_llm().chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=settings.llm.max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        parsed = {}

    return AnalysisReport(
        title=parsed.get("title", "Research Analysis Report"),
        executive_summary=parsed.get("executive_summary", ""),
    )


@tool(args_schema=DiscoverConferenceInput)
def discover_conference(venue_id: str) -> DiscoverConferenceOutput:
    """Probe OpenReview to discover what data is available for a venue_id.
    Returns available content fields, decision/venue patterns, and review invitation patterns.
    Use this before crawling an unknown conference to understand its structure."""
    return discover_venue(venue_id)


ALL_TOOLS = [
    search_papers,
    get_paper_reviews,
    cluster_reviews,
    analyze_temporal_distribution,
    identify_research_gaps,
    get_user_context,
    web_search,
    check_and_crawl_new_conference,
    discover_conference,
    generate_report,
]
