from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings


class Embedder:
    """OpenAI-SDK-compatible embedding client. Works with any OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        cfg = settings.embedding
        self._model = cfg.model
        self._batch_size = cfg.batch_size
        self._client = OpenAI(
            api_key=cfg.api_key,
            **({"base_url": cfg.base_url} if cfg.base_url else {}),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._embed_batch(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
