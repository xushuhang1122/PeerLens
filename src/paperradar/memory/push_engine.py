from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..schemas.memory import EpisodicEvent
from ..schemas.tools import PaperResult
from ..store.chroma import ChromaManager
from ..store.sqlite_memory import EpisodicStore
from .semantic import SemanticMemory


class PushEngine:
    def __init__(self) -> None:
        self._chroma = ChromaManager()
        self._store = EpisodicStore()
        self._semantic = SemanticMemory()

    def check_new_papers(
        self,
        since_timestamp: str | None = None,
        top_k: int = 10,
    ) -> list[PaperResult]:
        pref_vec = self._semantic.get_preference_vector()
        if pref_vec is None:
            return []

        all_data = self._chroma.get_all_content_embeddings()
        embs = all_data.get("embeddings")
        embs = [] if embs is None else list(embs)
        metas = list(all_data.get("metadatas") or [])
        ids = list(all_data.get("ids") or [])

        if len(embs) == 0:
            return []

        # Filter by timestamp if provided
        if since_timestamp:
            filtered = [
                (i, emb, meta)
                for i, (emb, meta) in enumerate(zip(embs, metas))
                if meta.get("crawled_at", "") > since_timestamp
            ]
            if not filtered:
                return []
            indices, embs, metas = zip(*filtered)
            ids = [ids[i] for i in indices]

        arr = np.array(embs, dtype=float)
        pref = np.array(pref_vec, dtype=float)
        scores = arr @ pref
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: list[PaperResult] = []
        for i in top_idx:
            meta = metas[i]
            results.append(
                PaperResult(
                    paper_id=ids[i],
                    title=meta.get("title", ""),
                    authors=meta.get("authors", "").split(", ") if meta.get("authors") else [],
                    abstract="",
                    keywords=meta.get("keywords", "").split(", ") if meta.get("keywords") else [],
                    primary_area=meta.get("primary_area", ""),
                    venue=meta.get("venue", ""),
                    decision=meta.get("decision", "unknown"),
                    forum_url=meta.get("forum_url", ""),
                    year=int(meta.get("year", 0)),
                    conference=meta.get("conference", ""),
                    rrf_score=float(scores[i]),
                )
            )

        self._store.log_event(
            EpisodicEvent(
                event_type="push_sent",
                content=f"push:{len(results)} papers",
                session_id="push_engine",
            )
        )
        return results

    def run_push_check(self, top_k: int = 10) -> list[PaperResult]:
        last_ts = self._store.get_last_push_timestamp()
        return self.check_new_papers(since_timestamp=last_ts, top_k=top_k)
