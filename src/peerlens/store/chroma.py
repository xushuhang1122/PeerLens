from __future__ import annotations

from typing import Any, Optional

import chromadb

from ..config import settings
from ..schemas.memory import UserPreferenceEmbedding
from ..schemas.paper import Paper, Review


def _join(lst: list[str]) -> str:
    return ", ".join(lst)


def _paper_doc(paper: Paper) -> str:
    kw = _join(paper.keywords)
    return f"Title: {paper.title}\nAbstract: {paper.abstract}\nKeywords: {kw}"


def _paper_meta(paper: Paper) -> dict:
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "conference": paper.conference,
        "year": paper.year,
        "decision": paper.decision,
        "primary_area": paper.primary_area,
        "forum_url": paper.forum_url,
        "authors": _join(paper.authors),
        "keywords": _join(paper.keywords),
    }


def _review_doc(title: str, reviews: list[Review]) -> str:
    parts = [f"Paper: {title}", "Reviews:"]
    for r in reviews:
        parts.append(r.full_text[:2000])
        parts.append("---")
    return "\n".join(parts)


def _review_meta(paper: Paper, reviews: list[Review]) -> dict:
    ratings = [r.rating for r in reviews if r.rating is not None]
    confidences = [r.confidence for r in reviews if r.confidence is not None]
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "conference": paper.conference,
        "year": paper.year,
        "decision": paper.decision,
        "primary_area": paper.primary_area,
        "avg_rating": sum(ratings) / len(ratings) if ratings else 0.0,
        "review_count": len(reviews),
    }


_chroma_instance: "ChromaManager | None" = None


class ChromaManager:
    """Process-level singleton: all callers share one ChromaDB PersistentClient."""

    def __new__(cls) -> "ChromaManager":
        global _chroma_instance
        if _chroma_instance is None:
            _chroma_instance = super().__new__(cls)
        return _chroma_instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._client = chromadb.PersistentClient(path=settings.chroma.persist_dir)
        self._content = self._client.get_or_create_collection(
            settings.chroma.col_papers_content,
            metadata={"hnsw:space": "cosine"},
        )
        self._reviews = self._client.get_or_create_collection(
            settings.chroma.col_papers_reviews,
            metadata={"hnsw:space": "cosine"},
        )
        self._prefs = self._client.get_or_create_collection(
            settings.chroma.col_user_preferences,
            metadata={"hnsw:space": "cosine"},
        )
        self._agent_memory = self._client.get_or_create_collection(
            settings.chroma.col_agent_memory,
            metadata={"hnsw:space": "cosine"},
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Papers content collection
    # ------------------------------------------------------------------

    def upsert_papers_content(
        self, papers: list[Paper], embeddings: list[list[float]]
    ) -> None:
        if not papers:
            return
        self._content.upsert(
            ids=[p.id for p in papers],
            embeddings=embeddings,
            documents=[_paper_doc(p) for p in papers],
            metadatas=[_paper_meta(p) for p in papers],
        )

    def query_content(
        self,
        query_embedding: list[float],
        n_results: int = 50,
        where: Optional[dict] = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._content.query(**kwargs)

    def get_content_by_ids(self, ids: list[str]) -> dict:
        return self._content.get(
            ids=ids, include=["documents", "metadatas", "embeddings"]
        )

    def get_all_content_embeddings(self, where: Optional[dict] = None) -> dict:
        kwargs: dict[str, Any] = {"include": ["embeddings", "metadatas"]}
        if where:
            kwargs["where"] = where
        return self._content.get(**kwargs)

    def count_content(self, where: Optional[dict] = None) -> int:
        if where:
            return self._content.count()
        return self._content.count()

    def has_conference_data(self, conference: str, year: int) -> bool:
        result = self._content.get(
            where={"$and": [{"conference": conference}, {"year": year}]},
            limit=1,
            include=[],
        )
        return len(result["ids"]) > 0

    # ------------------------------------------------------------------
    # Papers reviews collection
    # ------------------------------------------------------------------

    def upsert_papers_reviews(
        self,
        papers: list[Paper],
        reviews_by_paper: dict[str, list[Review]],
        embeddings: list[list[float]],
    ) -> None:
        valid_papers = [p for p in papers if p.id in reviews_by_paper]
        if not valid_papers:
            return
        self._reviews.upsert(
            ids=[p.id for p in valid_papers],
            embeddings=embeddings,
            documents=[_review_doc(p.title, reviews_by_paper[p.id]) for p in valid_papers],
            metadatas=[_review_meta(p, reviews_by_paper[p.id]) for p in valid_papers],
        )

    def query_reviews(
        self,
        query_embedding: list[float],
        n_results: int = 50,
        where: Optional[dict] = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._reviews.query(**kwargs)

    def get_all_review_embeddings(self, where: Optional[dict] = None) -> dict:
        kwargs: dict[str, Any] = {"include": ["embeddings", "metadatas", "documents"]}
        if where:
            kwargs["where"] = where
        return self._reviews.get(**kwargs)

    # ------------------------------------------------------------------
    # User preferences collection
    # ------------------------------------------------------------------

    def upsert_user_preference(
        self, pref: UserPreferenceEmbedding, embedding: list[float]
    ) -> None:
        self._prefs.upsert(
            ids=[pref.doc_id],
            embeddings=[embedding],
            documents=[pref.text],
            metadatas=[{
                "source": pref.source,
                "paper_id": pref.paper_id or "",
                "timestamp": pref.timestamp,
            }],
        )

    def get_all_preference_embeddings(self) -> dict:
        return self._prefs.get(include=["embeddings", "metadatas"])

    def query_preferences(
        self, query_embedding: list[float], n_results: int = 20
    ) -> dict:
        return self._prefs.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

    # ------------------------------------------------------------------
    # Agent memory collection
    # ------------------------------------------------------------------

    def upsert_agent_memory(
        self,
        session_id: str,
        embedding: list[float],
        document: str,
        metadata: dict,
    ) -> None:
        self._agent_memory.upsert(
            ids=[session_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def query_agent_memory(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._agent_memory.query(**kwargs)
