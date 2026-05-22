from __future__ import annotations

import json
from collections import Counter

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import settings
from ..schemas.tools import ClusterAnalysis, ClusterInfo, ClusterReviewsInput
from ..store.chroma import ChromaManager


def _top_tfidf_terms(docs: list[str], top_n: int = 8) -> list[str]:
    if not docs:
        return []
    vec = TfidfVectorizer(max_features=500, stop_words="english")
    try:
        mat = vec.fit_transform(docs)
        scores = np.asarray(mat.sum(axis=0)).flatten()
        terms = vec.get_feature_names_out()
        top_idx = scores.argsort()[-top_n:][::-1]
        return [str(terms[i]) for i in top_idx]
    except ValueError:
        return []


def _gpt_label_cluster(
    client: OpenAI,
    top_terms: list[str],
    quotes: list[str],
    avg_rating: float | None,
) -> tuple[str, str]:
    prompt = (
        "You are analyzing a cluster of ML paper reviewer comments.\n"
        f"Top terms: {', '.join(top_terms)}\n"
        f"Representative excerpts:\n"
        + "\n".join(f"- {q[:300]}" for q in quotes[:3])
        + f"\nAverage reviewer rating: {avg_rating:.1f}" if avg_rating else ""
        + "\n\nRespond with JSON: {\"label\": \"<short label>\", \"criticism_pattern\": \"<one sentence>\"}"
    )
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(resp.choices[0].message.content or "{}")
        return parsed.get("label", ""), parsed.get("criticism_pattern", "")
    except (json.JSONDecodeError, KeyError):
        return "", ""


class ReviewClusterer:
    def __init__(self) -> None:
        self._chroma = ChromaManager()
        self._llm = OpenAI(
            api_key=settings.llm.openai_api_key,
            **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
        )

    def cluster(self, inp: ClusterReviewsInput) -> ClusterAnalysis:
        data = self._chroma.get_all_review_embeddings(
            where={"primary_area": inp.primary_area}
        )

        embeddings_raw = data.get("embeddings")
        embeddings_raw = [] if embeddings_raw is None else list(embeddings_raw)
        documents = list(data.get("documents") or [])
        metadatas = list(data.get("metadatas") or [])

        if len(embeddings_raw) < inp.n_clusters:
            return ClusterAnalysis(
                primary_area=inp.primary_area,
                n_clusters=0,
                clusters=[],
                high_frequency_rejections=[],
            )

        emb_matrix = np.array(embeddings_raw, dtype=float)
        n_clusters = min(inp.n_clusters, len(embeddings_raw))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(emb_matrix)

        clusters: list[ClusterInfo] = []
        all_rejection_terms: list[str] = []

        for cid in range(n_clusters):
            mask = labels == cid
            cluster_docs = [d for d, m in zip(documents, mask) if m]
            cluster_metas = [m for m, flag in zip(metadatas, mask) if flag]
            cluster_embs = emb_matrix[mask]

            centroid = km.cluster_centers_[cid]
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            closest_idx = np.argsort(dists)[:3]
            representative_quotes = [cluster_docs[i][:400] for i in closest_idx]

            ratings = [
                m.get("avg_rating")
                for m in cluster_metas
                if m.get("avg_rating") is not None
            ]
            avg_rating = float(np.mean(ratings)) if ratings else None

            top_terms = _top_tfidf_terms(cluster_docs)
            label, criticism = _gpt_label_cluster(
                self._llm, top_terms, representative_quotes, avg_rating
            )

            if avg_rating is not None and avg_rating < 5.0:
                all_rejection_terms.append(label or criticism)

            clusters.append(
                ClusterInfo(
                    cluster_id=cid,
                    label=label,
                    top_terms=top_terms,
                    representative_quotes=representative_quotes,
                    paper_count=int(mask.sum()),
                    avg_rating=avg_rating,
                    criticism_pattern=criticism,
                )
            )

        counter = Counter(all_rejection_terms)
        high_freq = [t for t, _ in counter.most_common(5) if t]

        return ClusterAnalysis(
            primary_area=inp.primary_area,
            n_clusters=n_clusters,
            clusters=clusters,
            high_frequency_rejections=high_freq,
        )
