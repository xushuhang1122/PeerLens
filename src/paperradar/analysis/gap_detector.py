from __future__ import annotations

import json

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans

from ..config import settings
from ..retrieval.embedder import Embedder
from ..schemas.tools import (
    ClusterReviewsInput,
    GapReport,
    IdentifyGapsInput,
    ResearchGap,
)
from ..store.chroma import ChromaManager
from .clustering import ReviewClusterer


class GapDetector:
    def __init__(self) -> None:
        self._chroma = ChromaManager()
        self._embedder = Embedder()
        self._clusterer = ReviewClusterer()
        self._llm = OpenAI(
            api_key=settings.llm.openai_api_key,
            **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
        )

    def detect(self, inp: IdentifyGapsInput) -> GapReport:
        data = self._chroma.get_all_content_embeddings(
            where={"primary_area": inp.domain}
        )
        embeddings_raw = data.get("embeddings")
        embeddings_raw = [] if embeddings_raw is None else list(embeddings_raw)
        metadatas = list(data.get("metadatas") or [])

        if len(embeddings_raw) == 0:
            return GapReport(
                domain=inp.domain,
                gaps=[],
                covered_areas=[],
                sparse_areas=[],
                rejection_patterns=[],
                submission_advice="Not enough data to analyze this domain.",
            )

        emb_matrix = np.array(embeddings_raw, dtype=float)
        n_clusters = min(10, len(embeddings_raw))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(emb_matrix)

        total = len(embeddings_raw)
        covered_areas: list[str] = []
        sparse_areas: list[str] = []
        gap_candidates: list[tuple[int, float, list[str]]] = []

        for cid in range(n_clusters):
            mask = labels == cid
            density = float(mask.sum()) / total
            cluster_metas = [m for m, flag in zip(metadatas, mask) if flag]
            titles = [m.get("title", "") for m in cluster_metas[:3]]

            area_summary = f"cluster_{cid} (~{mask.sum()} papers)"
            if density >= inp.min_cluster_density:
                covered_areas.append(area_summary)
            else:
                sparse_areas.append(area_summary)
                gap_candidates.append((cid, density, titles))

        cluster_analysis = self._clusterer.cluster(
            ClusterReviewsInput(primary_area=inp.domain, n_clusters=5)
        )
        rejection_patterns = cluster_analysis.high_frequency_rejections

        gaps = self._synthesize_gaps(gap_candidates, inp.domain)

        advice = self._generate_advice(inp.domain, gaps, rejection_patterns)

        return GapReport(
            domain=inp.domain,
            gaps=gaps,
            covered_areas=covered_areas,
            sparse_areas=sparse_areas,
            rejection_patterns=rejection_patterns,
            submission_advice=advice,
        )

    def _synthesize_gaps(
        self,
        candidates: list[tuple[int, float, list[str]]],
        domain: str,
    ) -> list[ResearchGap]:
        if not candidates:
            return []

        descriptions: list[dict] = [
            {"cluster_id": cid, "density": round(d, 3), "sample_titles": titles}
            for cid, d, titles in candidates
        ]

        prompt = (
            f"You are analyzing research gaps in the domain: {domain}.\n"
            f"The following low-density paper clusters were identified:\n"
            f"{json.dumps(descriptions, indent=2)}\n\n"
            "For each cluster, provide a ResearchGap as JSON array:\n"
            '[{"gap_description": "...", "evidence": ["title1", "title2"], "suggested_angle": "..."}]\n'
            "Return only the JSON array."
        )
        resp = self._llm.chat.completions.create(
            model=settings.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        try:
            raw = json.loads(resp.choices[0].message.content or "[]")
            if isinstance(raw, list):
                items = raw
            else:
                items = raw.get("gaps", raw.get("items", []))
            return [
                ResearchGap(
                    gap_description=g.get("gap_description", ""),
                    evidence=g.get("evidence", []),
                    suggested_angle=g.get("suggested_angle", ""),
                )
                for g in items
                if g.get("gap_description")
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def _generate_advice(
        self,
        domain: str,
        gaps: list[ResearchGap],
        rejection_patterns: list[str],
    ) -> str:
        prompt = (
            f"Domain: {domain}\n"
            f"Identified gaps: {[g.gap_description for g in gaps]}\n"
            f"Common rejection reasons: {rejection_patterns}\n\n"
            "Write 2-3 sentences of actionable submission advice for a researcher in this domain."
        )
        resp = self._llm.chat.completions.create(
            model=settings.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content or ""
