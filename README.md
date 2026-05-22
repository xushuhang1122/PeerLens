# 🔬 PeerLens

> A local AI research assistant powered by real peer review data from OpenReview.
> Search ML papers by reviewer signal, generate literature surveys, and get venue-calibrated diagnosis for your own paper.

---

## ✨ Features

| | |
|---|---|
| 🔍 **Hybrid Search** | BM25 + dual-vector semantic search across paper content *and* reviewer comments, fused via RRF. Filter by decision, conference, and year. |
| 📝 **Research Agent** | Describe a topic in natural language → structured literature survey with citations, temporal trends, and submission advice. |
| 🩺 **Diagnosis Agent** | Upload your PDF → simulated peer review (overall + soundness/presentation/contribution scores) + prioritized suggestions each matched to a real reviewer comment. |
| 📚 **Library** | Crawl any OpenReview venue, re-fetch reviews, and browse database stats. |
| 🧠 **Memory** | Episodic query history + semantic preference vectors. Surface newly crawled papers that match your interests. |

---

## 🏛️ Supported Venues

**NeurIPS · ICML · ICLR · AISTATS · UAI · CoRL · COLM · RLC**

All publish submissions and open peer reviews on OpenReview — no authentication required.
Any other OpenReview venue can be discovered and indexed via the Library page.

---

## 🚀 Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure API keys**

```bash
cp .env.example .env
```

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=           # leave blank for OpenAI, or any compatible endpoint
LLM_MODEL=gpt-4o

EMBEDDING_API_KEY=      # falls back to LLM_API_KEY if blank
EMBEDDING_BASE_URL=     # e.g. https://cloud.infini-ai.com/maas/v1
EMBEDDING_MODEL=text-embedding-3-large
```

**3. Launch**

```bash
streamlit run app.py
```

On first launch, an onboarding wizard detects an empty database and offers a one-click crawl setup for top conferences + last 3 years.

**4. CLI (optional)**

```bash
python main.py crawl --conference NeurIPS --year 2024
python main.py search "efficient attention mechanisms" --decision oral spotlight --top-k 20
```

---

## 🏗️ Architecture

```
OpenReview API (api2.openreview.net)
      │
  CrawlPipeline ──► ChromaDB  (papers_content · papers_reviews · user_preferences)
                └─► BM25Index
                └─► SQLite    (crawl_log · episodic memory)

Query / PDF
  │
HybridSearcher
  ├── BM25 keyword score
  ├── papers_content vector score
  └── papers_reviews vector score   ← reviewer signal search
        └── RRF fusion ──► ranked results

📝 Research Agent (LangGraph)      🩺 Diagnosis Agent (LangGraph)
  clarify (multi-turn)               detect  (domain + keywords)
  retrieve (hybrid search)           search  (accepted + rejected)
  analyze  (temporal trends)         review_analysis (K-Means clusters)
  synthesize_survey (LLM)            diagnose (SimulatedReview + suggestions)
```

---

## 📁 Project Structure

```
src/paperradar/
├── config.py              # All settings, loaded from .env
├── schemas/               # Pydantic v2: papers, tools, agent states, survey, diagnosis
├── crawl/                 # OpenReview crawler, async review fetcher, crawl pipeline
├── store/                 # ChromaDB (singleton), BM25 (singleton), SQLite episodic store
├── retrieval/             # Embedder, hybrid search, RRF fusion
├── analysis/              # Temporal trends, K-Means clustering, gap detection
├── memory/                # Episodic memory, semantic preferences, push engine
└── agent/                 # LangGraph graphs + runners (research, diagnosis)

pages/
├── home.py                # Home + onboarding wizard
├── 2_Agent.py             # Research Agent
├── 6_Diagnose.py          # Diagnosis Agent
├── 1_Search.py            # Hybrid paper search
├── 3_Analysis.py          # Trends + clustering
├── 4_Library.py           # Crawl management + database stats
└── 5_Memory.py            # Query history + push recommendations
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | Yes | — | API key for LLM calls |
| `LLM_BASE_URL` | No | OpenAI | Compatible API base URL |
| `LLM_MODEL` | No | `gpt-4o` | Model name |
| `EMBEDDING_API_KEY` | No | = `LLM_API_KEY` | Embedding API key |
| `EMBEDDING_BASE_URL` | No | OpenAI | Embedding endpoint base URL |
| `EMBEDDING_MODEL` | No | `text-embedding-3-large` | Embedding model |

---

## 🗂️ Data Source

Paper and review data is fetched from [OpenReview](https://openreview.net) via the public API (`api2.openreview.net`). Data is stored locally for personal research use only — do not redistribute. Requests are rate-limited (batch sleep 1s, per-request 0.5s) to avoid overloading the service. See [OpenReview Terms of Use](https://openreview.net/legal/terms).

---

## 🗺️ Roadmap

- [ ] **Writing Agent** — improve your paper section by section guided by real reviewer patterns
- [ ] **Scheduled push notifications** — email / Slack digest of new papers matching your interests
- [ ] **Multi-user support** — isolated memory and preference vectors per user identity
- [ ] **Score normalization** — unify review scales across venues (NeurIPS 1–10, ICLR 1–10, COLM 1–5, …)

---

## 📄 License

MIT
