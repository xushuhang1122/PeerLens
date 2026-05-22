# PaperRadar

A local research assistant for ML/AI academic papers on OpenReview. Combines hybrid search, review signal analysis, a multi-turn Research Agent that generates literature surveys, and a Diagnosis Agent that simulates peer review for your own paper — all in a Streamlit interface.

## Changelog

**2026-05-22**
- **Diagnosis Agent** — upload your PDF, select a target venue (NeurIPS, ICML, ICLR, JMLR, …), and get a simulated peer review with venue-specific scoring scales, strengths/weaknesses/questions, and prioritized improvement suggestions with matched reviewer comments
- **Research Agent** — redesigned with multi-turn intent clarification (one focused question per turn, sensible defaults), force-extract refined query on early start, and full survey persistence across page reruns
- **Grouped navigation** — sidebar reorganized with an "Agents" section (Research Agent, Diagnose Paper) and a "Tools" section; adding future agents requires one line in `app.py`
- **Onboarding wizard** — first-run experience detects empty database and offers a one-click crawl setup for the top conferences + last 3 years
- **Removed gap analysis from survey** — the research agent no longer runs `identify_research_gaps`; it now reports temporal trends only, which are more relevant for a literature survey
- **Diagnosis search fix** — removed decision filter from similarity search; papers are now split accepted/rejected in Python after retrieval to maximize recall
- **Survey no longer disappears** — survey object persisted in session state and re-rendered after `st.rerun()`
- **Decision field fix** — papers now correctly show `oral/spotlight/poster/accepted/rejected` (previously all showed `unknown` due to wrong OpenReview field priority)
- **Performance** — ChromaDB and BM25 index use process-level singletons; agent tools initialize lazily to avoid blocking startup

**2026-05-xx** *(initial release)*
- Hybrid BM25 + dual-vector search with RRF fusion
- OpenReview crawler for NeurIPS, ICML, ICLR, AISTATS, UAI, CoRL, COLM, RLC
- Temporal trend analysis, K-Means review clustering, research gap detection
- Episodic + semantic long-term memory, personalized push recommendations

---

## Features

### Agents
- **Research Agent** — describe a research topic in natural language. The agent asks one focused question to clarify time range and venue, confirms its understanding, then searches the local paper database and writes a structured mini survey: background, thematic sections with in-text citations, a list of key papers with abstracts, open research questions, and submission advice.
- **Diagnosis Agent** — upload your paper as a PDF and optionally specify your target venue. The agent extracts your domain and keywords, finds similar accepted and rejected papers, analyzes reviewer concern patterns, and produces: a simulated peer review (overall score, soundness/presentation/contribution, strengths/weaknesses/questions to authors) calibrated to the target venue's actual review criteria, plus prioritized improvement suggestions each paired with an example reviewer comment.

### Search & Retrieval
- **Hybrid Search** — BM25 keyword matching + dual-vector semantic search (paper content and reviewer comments), fused via Reciprocal Rank Fusion. Filter by decision (oral/spotlight/poster/accepted/rejected), conference, and year.
- **Review Signal Index** — reviewer comments are embedded separately from abstracts, enabling queries like "papers reviewers found poorly motivated" to surface meaningfully different results.

### Analysis
- **Temporal Trend Analysis** — track how a research topic's presence evolves across conferences and years.
- **K-Means Review Clustering** — identify high-frequency criticism patterns in a research area from reviewer text.
- **Research Gap Detection** — find under-explored topic clusters from paper embeddings.

### Memory & Personalization
- **Long-term User Memory** — episodic layer (SQLite) records query history and paper feedback; semantic layer (ChromaDB) builds a preference vector from liked papers.
- **Push Recommendations** — surface newly crawled papers that match your research interests.

---

## Supported Venues

Preset: **NeurIPS, ICML, ICLR, AISTATS, UAI, CoRL, COLM, RLC**

These venues publish submissions and open peer reviews directly on OpenReview, enabling full crawling without authentication. ACL-family venues (ACL, EMNLP, NAACL) use ACL Rolling Review with access-controlled reviews — they are not included as crawl presets but the Diagnosis Agent's simulated review supports them as target venues.

Any OpenReview venue can be discovered and crawled via the Library page.

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure API keys**

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM (agents + report generation)
LLM_API_KEY=sk-...
LLM_BASE_URL=          # leave blank for OpenAI, or any compatible endpoint
LLM_MODEL=gpt-4o       # or any compatible model name

# Embeddings (any OpenAI-compatible endpoint)
EMBEDDING_API_KEY=     # falls back to LLM_API_KEY if blank
EMBEDDING_BASE_URL=    # e.g. https://cloud.infini-ai.com/maas/v1
EMBEDDING_MODEL=text-embedding-3-large
```

**3. Launch the web interface**

```bash
streamlit run app.py
```

On first launch, an onboarding wizard detects an empty database and offers a one-click crawl setup.

**4. Or crawl directly via CLI**

```bash
python main.py crawl --conference NeurIPS --year 2024
python main.py crawl --conference ICLR --year 2024

# Search via CLI
python main.py search "efficient attention mechanisms" --decision oral spotlight --top-k 20
```

---

## Project Structure

```
src/paperradar/
├── config.py              # All settings, loaded from .env
├── schemas/               # Pydantic v2 models: papers, tools, agent states, survey, diagnosis
├── crawl/                 # OpenReview crawler, async review fetcher, crawl pipeline
├── store/                 # ChromaDB manager (singleton), BM25 index (singleton), SQLite episodic store
├── retrieval/             # Embedder, hybrid search (RRF fusion), filter builders
├── analysis/              # Temporal trends, K-Means clustering, gap detection
├── memory/                # Episodic memory, semantic preferences, push engine
├── agent/                 # LangGraph graphs, tools, runners
│   ├── research_graph.py  # retrieve → analyze → synthesize_survey
│   ├── research_runner.py
│   ├── diagnosis_graph.py # detect → search → review_analysis → diagnose
│   └── diagnosis_runner.py
└── utils/                 # PDF text extraction

pages/
├── home.py                # Home page + onboarding wizard
├── 2_Agent.py             # Research Agent (multi-turn clarification + survey)
├── 6_Diagnose.py          # Diagnosis Agent (PDF upload + simulated review)
├── 1_Search.py            # Hybrid paper search
├── 3_Analysis.py          # Temporal trends, clustering, gap detection
├── 4_Library.py           # Crawl management, database stats
└── 5_Memory.py            # Query history, liked papers, push recommendations

app.py                     # Streamlit entry point + navigation (st.navigation with sections)
main.py                    # CLI entry point
```

---

## Architecture Overview

```
OpenReview API (api2.openreview.net)
      │
  CrawlPipeline ──► ChromaDB  (papers_content, papers_reviews, user_preferences)
                └─► BM25Index (pickle, process singleton)
                └─► SQLite    (crawl_log, episodic_events)

Query / PDF
  │
HybridSearcher
  ├── BM25 score
  ├── papers_content vector score
  └── papers_reviews vector score
        └── RRF fusion ──► ranked PaperResult list

Research Agent (LangGraph)           Diagnosis Agent (LangGraph)
  clarify (plain LLM, multi-turn)      detect_node  (LLM → domain + keywords)
  │                                    search_node  (hybrid search, split by decision)
  └─► retrieve_node  (search_papers)   review_analysis_node (cluster + reviews)
      analyze_node   (temporal trend)  diagnose_node (LLM → SimulatedReview + suggestions)
      synthesize_survey_node (LLM)
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | Yes | — | API key for LLM calls |
| `LLM_BASE_URL` | No | OpenAI | Compatible API base URL |
| `LLM_MODEL` | No | `gpt-4o` | Model name |
| `EMBEDDING_API_KEY` | No | = `LLM_API_KEY` | Embedding API key |
| `EMBEDDING_BASE_URL` | No | OpenAI | Embedding endpoint base URL |
| `EMBEDDING_MODEL` | No | `text-embedding-3-large` | Embedding model name |

---

## Data Source

All paper and review data is fetched from [OpenReview](https://openreview.net) via the public API (`api2.openreview.net`), the platform's official documented endpoint for programmatic access. OpenReview publishes submissions and peer reviews openly as part of its mission to make academic review transparent.

- Data is attributed to OpenReview and the respective authors and reviewers.
- PaperRadar stores data locally for personal research use only. Do not redistribute crawled data or use it for commercial purposes.
- Requests are rate-limited (batch sleep 1 s, per-request sleep 0.5 s) to avoid overloading the service.
- See [OpenReview Terms of Use](https://openreview.net/legal/terms) for full details.

---

## TODO

- [ ] **BibTeX / CSV export** — export selected search results or survey papers to `.bib` or `.csv`.
- [ ] **Incremental crawl** — detect newly added papers in an already-crawled venue/year and index only the delta.
- [ ] **Review score parsing** — extract numeric ratings and confidence scores from review text and expose them as filter/sort dimensions.
- [ ] **Multi-user support** — isolate episodic memory and preference vectors per user identity for lab sharing.
- [ ] **Scheduled push notifications** — run the push engine on a cron schedule and deliver alerts via email or Slack webhook.
- [ ] **More agents** — e.g. a Writing Agent (improve your paper section by section against reviewer patterns), a Trend Agent (weekly digest of new papers matching your interests).

---

## License

MIT
