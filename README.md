# 🔬 PeerLens

> A local AI research assistant powered by real peer review data from OpenReview.
> Search ML papers by reviewer signal, generate literature surveys, and get venue-calibrated diagnosis for your own paper.

---

## ✨ Features

| | |
|---|---|
| 🔍 **Hybrid Search** | BM25 + dual-vector semantic search across paper content *and* reviewer comments, fused via RRF. Filter by decision, conference, and year. |
| 📝 **Research Agent** | Describe a topic in natural language → structured literature survey with citations, temporal trends, and submission advice. |
| 🩺 **Diagnosis Agent** | Upload your PDF → structured findings ranked by repair cost (revision / experiment / redesign), with concrete action steps for each quick fix, grounded in real reviewer patterns from the database. |
| 📖 **Reading Agent** *(experimental)* | Deep-read any paper via PDF, OpenReview URL, ArXiv URL, or topic search. Generates a structured report (TL;DR, contributions, methodology, reviewer perspectives) and supports multi-turn academic discussion. |
| 📚 **Library** | Crawl any OpenReview venue, re-fetch reviews, and browse database stats. |
| 🧠 **Memory** | Episodic query history + semantic preference vectors. Surface newly crawled papers that match your interests. |

---

## 🏛️ Supported Venues

**NeurIPS · ICML · ICLR**

The current shared database covers these three top ML conferences (2023–2025), including both accepted papers and public rejected submissions from ICLR, totalling ~25,000 papers with full reviewer comments.

Other OpenReview venues (AISTATS, UAI, CoRL, COLM, RLC, etc.) can be discovered and indexed locally via the Library page.

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

## 🔌 Remote MCP Mode *(experimental)*

PeerLens can connect to a shared remote database instead of building one locally. A public server is available for immediate use:

```env
REMOTE_MCP_URL=http://43.134.60.58:8765/mcp
```

Add this line to your `.env` (or toggle it in the sidebar), and all agent search calls are routed to the shared server — no local crawling or embedding required. The shared database currently covers NeurIPS / ICML / ICLR 2023–2025 (~25,000 papers).

**Self-hosting** (on a machine with pre-crawled data):

```bash
pip install -r requirements.txt
python server/mcp_server.py          # listens on 0.0.0.0:8765 by default
```

For a production deployment with systemd + optional nginx/HTTPS:

```bash
sudo EMBEDDING_API_KEY=sk-... bash server/deploy.sh
# With domain + HTTPS:
sudo EMBEDDING_API_KEY=sk-... DOMAIN=mcp.example.com ENABLE_HTTPS=1 bash server/deploy.sh
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

📖 Reading Agent (LangGraph)
  parse_input (PDF / URL / topic)
  fetch_openreview / fetch_arxiv
  inject_reviews (local DB lookup)
  deep_read (structured report + reviewer perspectives)
  discussion (multi-turn chat)
```

---

## 📁 Project Structure

```
src/peerlens/
├── config.py              # All settings, loaded from .env
├── schemas/               # Pydantic v2: papers, tools, agent states, survey, diagnosis, reading
├── crawl/                 # OpenReview crawler, async review fetcher, crawl pipeline
├── store/                 # ChromaDB (singleton), BM25 (singleton), SQLite episodic store
├── retrieval/             # Embedder, hybrid search, RRF fusion
├── analysis/              # Temporal trends, K-Means clustering, gap detection
├── memory/                # Episodic memory, semantic preferences, push engine
└── agent/                 # LangGraph graphs + runners (research, diagnosis, reading)

pages/
├── home.py                # Home + onboarding wizard
├── 1_Search.py            # Hybrid paper search
├── 2_Agent.py             # Research Agent
├── 3_Analysis.py          # Trends + clustering
├── 4_Library.py           # Crawl management + database stats
├── 5_Memory.py            # Query history + push recommendations
├── 6_Diagnose.py          # Diagnosis Agent
└── 7_Reading.py           # Reading Agent + multi-turn discussion
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
| `REMOTE_MCP_URL` | No | — | MCP server endpoint (e.g. `http://host:8765/mcp`). When set, agent search uses the remote database instead of local. |

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
