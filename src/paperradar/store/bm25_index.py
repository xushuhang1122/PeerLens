from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from ..config import settings
from ..schemas.paper import Paper

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can of in on at to for "
    "with by from as or and but not this that these those it its".split()
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


_bm25_instance: "BM25Index | None" = None


class BM25Index:
    """Process-level singleton: the BM25 pickle is loaded once and shared."""

    def __new__(cls, index_path: str | None = None) -> "BM25Index":
        global _bm25_instance
        if _bm25_instance is None:
            _bm25_instance = super().__new__(cls)
        return _bm25_instance

    def __init__(self, index_path: str | None = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self._path = Path(index_path or settings.bm25_index_path)
        self._bm25: Optional[BM25Okapi] = None
        self._paper_ids: list[str] = []
        self._initialized = True

    def build(self, papers: list[Paper]) -> None:
        corpus = [
            _tokenize(f"{p.title} {p.abstract} {' '.join(p.keywords)}")
            for p in papers
        ]
        self._paper_ids = [p.id for p in papers]
        self._bm25 = BM25Okapi(corpus)
        self._save()

    def load(self) -> bool:
        if not self._path.exists():
            return False
        with open(self._path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._paper_ids = data["paper_ids"]
        return True

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        if self._bm25 is None:
            if not self.load():
                return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._paper_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(pid, float(score)) for pid, score in ranked[:top_k] if score > 0]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "paper_ids": self._paper_ids}, f)
