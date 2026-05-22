from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from ..config import settings
from ..schemas.paper import Paper, Review
from ..store.bm25_index import BM25Index
from ..store.chroma import ChromaManager
from ..store.sqlite_memory import EpisodicStore
from ..retrieval.embedder import Embedder
from .openreview import OpenReviewClient
from .reviews import ReviewFetcher

_running_crawls: dict[str, bool] = {}
_crawl_lock = threading.Lock()


def _crawl_key(conference: str, year: int) -> str:
    return f"{conference}_{year}"


def is_crawl_running(conference: str, year: int) -> bool:
    return _running_crawls.get(_crawl_key(conference, year), False)


class CrawlPipeline:
    def __init__(self) -> None:
        self._client = OpenReviewClient()
        self._review_fetcher = ReviewFetcher()
        self._chroma = ChromaManager()
        self._bm25 = BM25Index()
        self._embedder = Embedder()
        self._store = EpisodicStore()

    def check_local(self, conference: str, year: int) -> bool:
        """Return True if this conference/year is already in ChromaDB."""
        return self._chroma.has_conference_data(conference, year)

    def run_sync(
        self,
        conference: str,
        year: int,
        decision: Optional[str] = None,
        on_complete: Optional[Callable[[int], None]] = None,
    ) -> int:
        """Crawl, embed, and index papers. Returns paper count."""
        key = _crawl_key(conference, year)
        with _crawl_lock:
            if _running_crawls.get(key):
                return 0
            _running_crawls[key] = True

        try:
            papers = self._client.fetch_papers(conference, year, decision)
            if not papers:
                self._store.log_crawl(conference, year, decision, 0, "success")
                return 0

            self._save_raw(papers, conference, year)
            self._index_papers(papers)
            self._index_reviews(papers)
            self._store.log_crawl(conference, year, decision, len(papers), "success")

            if on_complete:
                on_complete(len(papers))
            return len(papers)
        except Exception as e:
            self._store.log_crawl(conference, year, decision, 0, "failed")
            raise
        finally:
            with _crawl_lock:
                _running_crawls[key] = False

    def run_async(
        self,
        conference: str,
        year: int,
        decision: Optional[str] = None,
        on_complete: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Start crawl in a background thread (non-blocking)."""
        thread = threading.Thread(
            target=self.run_sync,
            args=(conference, year, decision, on_complete),
            daemon=True,
            name=f"crawl-{conference}-{year}",
        )
        thread.start()

    def _save_raw(self, papers: list[Paper], conference: str, year: int) -> None:
        out_dir = Path(settings.raw_data_dir) / f"{conference.lower()}_{year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "papers.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([p.model_dump(mode="json") for p in papers], f, indent=2)

    def _index_papers(self, papers: list[Paper]) -> None:
        print(f"Embedding {len(papers)} papers (content)...")
        docs = [
            f"Title: {p.title}\nAbstract: {p.abstract}\nKeywords: {', '.join(p.keywords)}"
            for p in papers
        ]
        embeddings = self._embedder.embed(docs)
        self._chroma.upsert_papers_content(papers, embeddings)

        # Rebuild BM25 index including new papers
        existing = self._chroma.get_all_content_embeddings()
        all_ids = existing.get("ids", [])
        all_metas = existing.get("metadatas", [])
        all_papers_rebuilt = [
            Paper(
                id=m["paper_id"],
                title=m["title"],
                abstract="",
                keywords=m["keywords"].split(", ") if m.get("keywords") else [],
                conference=m.get("conference", ""),
                year=int(m.get("year", 0)),
                decision=m.get("decision", "unknown"),  # type: ignore[arg-type]
                primary_area=m.get("primary_area", ""),
                forum_url=m.get("forum_url", ""),
            )
            for m in all_metas
        ]
        self._bm25.build(all_papers_rebuilt)

    def _index_reviews(self, papers: list[Paper]) -> None:
        paper_ids = [p.id for p in papers]
        print(f"Fetching reviews for {len(paper_ids)} papers (async)...")
        reviews_by_paper = self._review_fetcher.fetch_reviews_for_papers(paper_ids)

        papers_with_reviews = [p for p in papers if reviews_by_paper.get(p.id)]
        if not papers_with_reviews:
            return

        print(f"Embedding {len(papers_with_reviews)} review sets...")
        docs = []
        for p in papers_with_reviews:
            reviews = reviews_by_paper[p.id]
            parts = [f"Paper: {p.title}", "Reviews:"]
            for r in reviews:
                parts.append(r.full_text[:2000])
                parts.append("---")
            docs.append("\n".join(parts))

        embeddings = self._embedder.embed(docs)
        self._chroma.upsert_papers_reviews(papers_with_reviews, reviews_by_paper, embeddings)

    def check_local_by_venue(self, venue_id: str) -> bool:
        """Check if any papers with this exact venue_id string are in ChromaDB."""
        result = self._chroma._content.get(
            where={"venue": {"$contains": venue_id.split("/")[0]}},
            limit=1,
            include=[],
        )
        return len(result.get("ids", [])) > 0

    def run_sync_custom(
        self,
        venue_id: str,
        label: str,
        on_complete: Optional[Callable[[int], None]] = None,
    ) -> int:
        """Crawl an arbitrary venue_id not in the preset conference list."""
        key = f"custom_{venue_id}"
        with _crawl_lock:
            if _running_crawls.get(key):
                return 0
            _running_crawls[key] = True

        try:
            papers = list(self._client.paginate(venue_id))
            from .openreview import _extract_paper
            parsed = []
            for note in papers:
                # Derive conference/year from venue_id string if possible
                parts = venue_id.split("/")
                conf = parts[0].split(".")[-1] if parts else label
                year_str = next((p for p in parts if p.isdigit() and len(p) == 4), "0")
                p = _extract_paper(note, conf, int(year_str))
                if p:
                    p.conference = label
                    parsed.append(p)

            if not parsed:
                return 0

            self._save_raw(parsed, label.replace(" ", "_"), 0)
            self._index_papers(parsed)
            self._index_reviews(parsed)

            if on_complete:
                on_complete(len(parsed))
            return len(parsed)
        finally:
            with _crawl_lock:
                _running_crawls[key] = False

    def run_async_custom(
        self,
        venue_id: str,
        label: str,
        on_complete: Optional[Callable[[int], None]] = None,
    ) -> None:
        thread = threading.Thread(
            target=self.run_sync_custom,
            args=(venue_id, label, on_complete),
            daemon=True,
            name=f"crawl-custom-{label}",
        )
        thread.start()

    def load_from_raw(self, conference: str, year: int) -> int:
        """Index papers from a previously saved raw JSON file (re-index without re-crawling)."""
        raw_file = Path(settings.raw_data_dir) / f"{conference.lower()}_{year}" / "papers.json"
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_file}")
        with open(raw_file, encoding="utf-8") as f:
            data = json.load(f)
        papers = [Paper.model_validate(d) for d in data]
        self._index_papers(papers)
        self._index_reviews(papers)
        return len(papers)

    def refetch_reviews(self, conference: str, year: int) -> int:
        """Re-fetch and re-index reviews for an already-crawled conference/year.

        Reads paper IDs from the saved raw JSON (no re-crawl of papers) and
        calls the review pipeline with the current (fixed) fetcher logic.
        Returns the number of papers that received at least one review.
        """
        raw_file = Path(settings.raw_data_dir) / f"{conference.lower()}_{year}" / "papers.json"
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_file}")
        with open(raw_file, encoding="utf-8") as f:
            data = json.load(f)
        papers = [Paper.model_validate(d) for d in data]
        self._index_reviews(papers)
        return len(papers)

    def refetch_reviews_async(
        self,
        conference: str,
        year: int,
        on_complete: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Run refetch_reviews in a background thread."""
        def _run() -> None:
            try:
                n = self.refetch_reviews(conference, year)
                if on_complete:
                    on_complete(n)
            except Exception:
                pass

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name=f"refetch-reviews-{conference}-{year}",
        )
        thread.start()
