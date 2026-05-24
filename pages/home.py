import datetime
import sys
sys.path.insert(0, ".")

import streamlit as st

from src.paperradar.config import settings
from src.paperradar.store.chroma import ChromaManager
from src.paperradar.crawl.pipeline import CrawlPipeline, is_crawl_running


@st.cache_resource
def get_chroma():
    return ChromaManager()


@st.cache_resource
def get_pipeline():
    return CrawlPipeline()


def _show_onboarding():
    st.title("Welcome to PeerLens")
    st.markdown(
        "Your local database is empty. "
        "Select the conferences and years you want to index first, then click **Start Setup**. "
        "Crawling runs in the background — you can start using the app while it loads."
    )
    st.divider()

    _current_year = datetime.datetime.now().year
    _default_years = [_current_year - 1, _current_year - 2, _current_year - 3]
    _supported = [c for c in settings.conferences.CONFERENCES]

    col_conf, col_year = st.columns(2)
    with col_conf:
        selected_confs = st.multiselect(
            "Conferences",
            options=_supported,
            default=["NeurIPS", "ICML", "ICLR"],
            help="Only venues that publish open peer reviews on OpenReview are listed.",
        )
    with col_year:
        selected_years = st.multiselect(
            "Years",
            options=list(range(_current_year - 1, 2021, -1)),
            default=_default_years,
        )

    if selected_confs and selected_years:
        total = len(selected_confs) * len(selected_years)
        st.caption(
            f"{total} crawl job(s) selected — "
            f"{', '.join(selected_confs)} x {', '.join(str(y) for y in sorted(selected_years, reverse=True))}"
        )
    st.info(
        "All submissions are indexed by default, including rejected papers. "
        "Rejected samples are used by the Diagnosis Agent to identify common reviewer concerns — "
        "removing them will reduce diagnostic accuracy.",
        icon=":material/info:",
    )

    col_btn, col_skip = st.columns([2, 8])
    with col_btn:
        start = st.button("Start Setup", type="primary", disabled=not (selected_confs and selected_years))
    with col_skip:
        skip = st.button("Skip for now", help="Go straight to the app — you can crawl data later via the Library page.")

    if start:
        pipeline = get_pipeline()
        launched = []
        for conf in selected_confs:
            for year in selected_years:
                if not is_crawl_running(conf, year):
                    pipeline.run_async(conf, year)
                    launched.append(f"{conf} {year}")
        if launched:
            st.success(
                f"Started background crawl for: {', '.join(launched)}. "
                "Check **Library > Database Stats** for progress. "
                "You can navigate to other pages while indexing runs."
            )
            st.session_state["onboarding_done"] = True
            st.rerun()

    if skip:
        st.session_state["onboarding_done"] = True
        st.rerun()


def _show_home():
    st.title("PeerLens")
    st.caption("Peer review-powered ML research assistant")

    if settings.remote_mcp.url:
        st.success(
            f"Connected to remote database at `{settings.remote_mcp.url}` — "
            "no local crawling required.",
            icon=":material/cloud_done:",
        )
    else:
        st.info(
            "Running in local mode. Use the **Library** page to crawl conference data.",
            icon=":material/storage:",
        )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.page_link("pages/1_Search.py", label="Search Papers", icon=":material/search:")
    with col2:
        st.page_link("pages/2_Agent.py", label="Research Agent", icon=":material/manage_search:")
    with col3:
        st.page_link("pages/3_Analysis.py", label="Analysis", icon=":material/bar_chart:")
    with col4:
        st.page_link("pages/4_Library.py", label="Library", icon=":material/library_books:")
    with col5:
        st.page_link("pages/5_Memory.py", label="My Profile", icon=":material/person:")

    st.divider()
    st.markdown("""
**Quick start**

1. Go to **Library** to crawl a conference (e.g. NeurIPS 2024) into the local database.
2. Use **Search Papers** to find relevant papers with hybrid BM25 + semantic retrieval.
3. Use **Research Agent** to conduct a deep literature survey on any topic.
4. Use **Diagnose Paper** to upload your PDF and get targeted reviewer feedback.
5. Like papers to train **My Profile** for personalized recommendations.
""")

    try:
        chroma = get_chroma()
        count = chroma._content.count()
        st.info(f"Database: **{count}** papers indexed. Add more via [Library](Library).")
    except Exception:
        pass


onboarding_done = st.session_state.get("onboarding_done", False)

if not onboarding_done:
    try:
        chroma = get_chroma()
        db_empty = chroma._content.count() == 0
    except Exception:
        db_empty = False

    if db_empty:
        _show_onboarding()
    else:
        _show_home()
else:
    _show_home()
