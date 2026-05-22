from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..retrieval.embedder import Embedder
from ..schemas.memory import UserPreferenceEmbedding
from ..schemas.paper import Paper
from ..store.chroma import ChromaManager


class SemanticMemory:
    def __init__(self) -> None:
        self._chroma = ChromaManager()
        self._embedder = Embedder()

    def add_liked_paper(self, paper: Paper) -> None:
        text = f"Title: {paper.title}\nAbstract: {paper.abstract}"
        embedding = self._embedder.embed_query(text)
        pref = UserPreferenceEmbedding(
            doc_id=f"liked_{paper.id}",
            text=text,
            source="liked_paper",
            paper_id=paper.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._chroma.upsert_user_preference(pref, embedding)

    def add_liked_topic(self, topic: str) -> None:
        import hashlib
        topic_id = hashlib.md5(topic.encode()).hexdigest()[:12]
        embedding = self._embedder.embed_query(topic)
        pref = UserPreferenceEmbedding(
            doc_id=f"topic_{topic_id}",
            text=topic,
            source="liked_topic",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._chroma.upsert_user_preference(pref, embedding)

    def get_preference_vector(self) -> Optional[list[float]]:
        data = self._chroma.get_all_preference_embeddings()
        embs = data.get("embeddings")
        embs = [] if embs is None else list(embs)
        if len(embs) == 0:
            return None
        arr = np.array(embs, dtype=float)
        mean_vec = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.tolist()

    def get_preference_summary(self) -> str:
        data = self._chroma.get_all_preference_embeddings()
        metas = data.get("metadatas") or []
        docs = self._chroma.get_all_preference_embeddings().get("documents") or []
        sources = [m.get("source", "") for m in metas]
        n_liked = sources.count("liked_paper")
        n_topic = sources.count("liked_topic")
        return f"{n_liked} liked papers, {n_topic} liked topics stored in semantic memory."
