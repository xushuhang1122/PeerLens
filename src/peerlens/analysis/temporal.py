from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import settings
from ..retrieval.embedder import Embedder
from ..retrieval.filters import build_where
from ..schemas.tools import AnalyzeTemporalInput, TemporalAnalysis, YearConferenceBucket
from ..store.chroma import ChromaManager


def _linear_trend(values: list[float]) -> str:
    if len(values) < 2:
        return "stable"
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])
    mean = float(np.mean(values)) or 1.0
    rel = slope / mean
    if rel > 0.1:
        return "rising"
    if rel < -0.1:
        return "declining"
    return "stable"


class TemporalAnalyzer:
    def __init__(self) -> None:
        self._chroma = ChromaManager()
        self._embedder = Embedder()

    def analyze(self, inp: AnalyzeTemporalInput) -> TemporalAnalysis:
        query_emb = self._embedder.embed_query(inp.topic)

        where = build_where(
            conference_filter=inp.conferences if inp.conferences else None,
            year_filter=inp.years if inp.years else None,
        )

        res = self._chroma.query_content(query_emb, n_results=500, where=where)
        ids: list[str] = res["ids"][0] if res["ids"] else []
        metas: list[dict] = res["metadatas"][0] if res["metadatas"] else []

        buckets: dict[tuple[str, int], dict] = defaultdict(
            lambda: {
                "paper_count": 0,
                "oral_count": 0,
                "spotlight_count": 0,
                "poster_count": 0,
                "titles": [],
            }
        )

        for meta in metas:
            conf = meta.get("conference", "")
            year = int(meta.get("year", 0))
            dec = meta.get("decision", "unknown")
            title = meta.get("title", "")
            key = (conf, year)
            buckets[key]["paper_count"] += 1
            if dec == "oral":
                buckets[key]["oral_count"] += 1
            elif dec == "spotlight":
                buckets[key]["spotlight_count"] += 1
            elif dec == "poster":
                buckets[key]["poster_count"] += 1
            if len(buckets[key]["titles"]) < 3:
                buckets[key]["titles"].append(title)

        distribution: list[YearConferenceBucket] = []
        for (conf, year), b in sorted(buckets.items(), key=lambda x: (x[0][1], x[0][0])):
            distribution.append(
                YearConferenceBucket(
                    conference=conf,
                    year=year,
                    paper_count=b["paper_count"],
                    oral_count=b["oral_count"],
                    spotlight_count=b["spotlight_count"],
                    poster_count=b["poster_count"],
                    top_papers=b["titles"],
                )
            )

        totals_by_year: dict[int, int] = defaultdict(int)
        for b in distribution:
            totals_by_year[b.year] += b.paper_count

        years_sorted = sorted(totals_by_year.keys())
        counts = [totals_by_year[y] for y in years_sorted]
        trend = _linear_trend(counts)

        peak_year = years_sorted[int(np.argmax(counts))] if counts else None
        peak_conf: str | None = None
        if peak_year:
            conf_counts = {
                b.conference: b.paper_count
                for b in distribution
                if b.year == peak_year
            }
            peak_conf = max(conf_counts, key=lambda c: conf_counts[c]) if conf_counts else None

        summary = (
            f"Found {len(ids)} papers related to '{inp.topic}' across "
            f"{len(distribution)} conference-year combinations. "
            f"Trend: {trend}."
            + (f" Peak year: {peak_year} ({peak_conf})." if peak_year else "")
        )

        return TemporalAnalysis(
            topic=inp.topic,
            distribution=distribution,
            trend=trend,
            peak_year=peak_year,
            peak_conference=peak_conf,
            summary=summary,
        )
