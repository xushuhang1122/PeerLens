from __future__ import annotations

import re
from typing import Optional

from ..config import settings
from ..schemas.tools import PaperResult, SearchPapersInput, SearchPapersOutput
from ..store.bm25_index import BM25Index
from ..store.chroma import ChromaManager
from .embedder import Embedder
from .filters import build_where


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _parse_abstract(doc: str) -> str:
    m = re.search(r"Abstract:\s*(.*?)(?:\nKeywords:|$)", doc, re.S)
    return m.group(1).strip() if m else ""


class HybridSearcher:
    def __init__(self) -> None:
        self._chroma = ChromaManager()
        self._bm25 = BM25Index()
        self._embedder = Embedder()

    def search(self, inp: SearchPapersInput) -> SearchPapersOutput:
        where = build_where(
            decision_filter=inp.decision_filter,
            conference_filter=inp.conference_filter,
            year_filter=inp.year_filter,
        )
        top_k = inp.top_k
        fetch_k = min(top_k * 5, 200)
        k = settings.agent.rrf_k

        query_emb = self._embedder.embed_query(inp.query)

        # --- BM25 ---
        bm25_hits = self._bm25.search(inp.query, top_k=fetch_k)
        bm25_rank: dict[str, int] = {pid: i + 1 for i, (pid, _) in enumerate(bm25_hits)}

        # --- Vector: content ---
        content_res = self._chroma.query_content(query_emb, n_results=fetch_k, where=where)
        content_ids: list[str] = content_res["ids"][0] if content_res["ids"] else []
        content_rank: dict[str, int] = {pid: i + 1 for i, pid in enumerate(content_ids)}
        content_meta: dict[str, dict] = {
            pid: meta
            for pid, meta in zip(content_ids, content_res["metadatas"][0])
        }
        content_docs: dict[str, str] = {
            pid: doc
            for pid, doc in zip(content_ids, content_res.get("documents", [[]])[0])
        }

        # --- Vector: reviews ---
        review_res = self._chroma.query_reviews(query_emb, n_results=fetch_k, where=where)
        review_ids: list[str] = review_res["ids"][0] if review_res["ids"] else []
        review_rank: dict[str, int] = {pid: i + 1 for i, pid in enumerate(review_ids)}

        # --- RRF fusion ---
        candidate_ids = set(bm25_rank) | set(content_rank) | set(review_rank)
        total = len(candidate_ids)
        default_rank = total + 1

        scored: list[tuple[str, float]] = []
        for pid in candidate_ids:
            score = (
                _rrf_score(bm25_rank.get(pid, default_rank), k)
                + _rrf_score(content_rank.get(pid, default_rank), k)
                + _rrf_score(review_rank.get(pid, default_rank), k)
            )
            scored.append((pid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [pid for pid, _ in scored[:top_k]]
        score_map = {pid: s for pid, s in scored}

        # --- Hydrate results ---
        # Collect metadata from content results; fetch missing ones if needed
        missing = [pid for pid in top_ids if pid not in content_meta]
        if missing:
            extra = self._chroma.get_content_by_ids(missing)
            for pid, meta, doc in zip(
                extra.get("ids", []),
                extra.get("metadatas", []),
                extra.get("documents", []),
            ):
                content_meta[pid] = meta
                content_docs[pid] = doc or ""

        results: list[PaperResult] = []
        for pid in top_ids:
            meta = content_meta.get(pid)
            if not meta:
                continue
            results.append(
                PaperResult(
                    paper_id=pid,
                    title=meta.get("title", ""),
                    authors=meta.get("authors", "").split(", ") if meta.get("authors") else [],
                    abstract=_parse_abstract(content_docs.get(pid, "")),
                    keywords=meta.get("keywords", "").split(", ") if meta.get("keywords") else [],
                    primary_area=meta.get("primary_area", ""),
                    venue=meta.get("venue", ""),
                    decision=meta.get("decision", "unknown"),
                    forum_url=meta.get("forum_url", ""),
                    year=int(meta.get("year", 0)),
                    conference=meta.get("conference", ""),
                    rrf_score=score_map[pid],
                    bm25_rank=bm25_rank.get(pid),
                    content_vec_rank=content_rank.get(pid),
                    review_vec_rank=review_rank.get(pid),
                )
            )

        return SearchPapersOutput(results=results, total_found=len(results), query=inp.query)
