from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class OpenReviewConfig:
    base_url: str = "https://api2.openreview.net/notes"
    limit: int = 1000
    batch_sleep: float = 1.0
    single_sleep: float = 0.5
    async_concurrency: int = 10
    max_retries: int = 3
    user_agent: str = "Mozilla/5.0 (compatible; PaperRadar/1.0)"


@dataclass
class ConferenceConfig:
    # Conferences confirmed to be on OpenReview with known venue_id patterns.
    # decisions: known venue-string patterns per decision type.
    # Empty decisions dict = auto-detect from venue field after fetching all papers.
    CONFERENCES: dict = field(default_factory=lambda: {
        # --- Top ML (CCF-A / de-facto top) ---
        "NeurIPS": {
            "venue_id": "NeurIPS.cc/{year}/Conference",
            "decisions": {
                "oral":      "NeurIPS {year} oral",
                "spotlight": "NeurIPS {year} spotlight",
                "poster":    "NeurIPS {year} poster",
            },
        },
        "ICML": {
            "venue_id": "ICML.cc/{year}/Conference",
            "decisions": {},
        },
        "ICLR": {
            "venue_id": "ICLR.cc/{year}/Conference",
            "decisions": {
                "oral":      "ICLR {year} oral",
                "spotlight": "ICLR {year} notable top 5%",
                "poster":    "ICLR {year} poster",
                "rejected":  "ICLR {year} rejected",
            },
        },
        # --- Other strong ML/AI venues on OpenReview ---
        "AISTATS": {
            "venue_id": "aistats.org/AISTATS/{year}/Conference",
            "decisions": {},
        },
        "UAI": {
            "venue_id": "auai.org/UAI/{year}/Conference",
            "decisions": {},
        },
        "CoRL": {
            "venue_id": "robot-learning.org/CoRL/{year}/Conference",
            "decisions": {},
        },
        "COLM": {
            "venue_id": "colmweb.org/COLM/{year}/Conference",
            "decisions": {},
        },
        "RLC": {
            "venue_id": "rl-conference.cc/RLC/{year}/Conference",
            "decisions": {},
        },
    })
    years: list = field(default_factory=lambda: [2022, 2023, 2024, 2025])


@dataclass
class EmbeddingConfig:
    api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "")
    )
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL") or None
    )
    model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    batch_size: int = 100
    review_max_tokens: int = 6000


@dataclass
class ChromaConfig:
    persist_dir: str = "data/chroma_db"
    col_papers_content: str = "papers_content"
    col_papers_reviews: str = "papers_reviews"
    col_user_preferences: str = "user_preferences"


@dataclass
class SQLiteConfig:
    db_path: str = "data/memory.db"


@dataclass
class LLMConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class AgentConfig:
    max_tool_retries: int = 3
    max_iterations: int = 20
    top_k_default: int = 20
    rrf_k: int = 60


@dataclass
class Settings:
    openreview: OpenReviewConfig = field(default_factory=OpenReviewConfig)
    conferences: ConferenceConfig = field(default_factory=ConferenceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    raw_data_dir: str = "data/raw"
    bm25_index_path: str = "data/bm25_index.pkl"


settings = Settings()
