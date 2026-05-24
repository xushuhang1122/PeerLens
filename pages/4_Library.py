import datetime
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

import sys
sys.path.insert(0, ".")

from src.paperradar.config import settings
from src.paperradar.agent.tools_remote import is_remote_mode
from src.paperradar.crawl.discover import discover_venue
from src.paperradar.crawl.pipeline import CrawlPipeline, is_crawl_running, get_crawl_progress
from src.paperradar.store.chroma import ChromaManager


@st.cache_resource
def get_chroma():
    return ChromaManager()


@st.cache_resource
def get_pipeline():
    return CrawlPipeline()


_PHASE_LABELS = {
    "fetching":  "Fetching papers from OpenReview...",
    "embedding": "Embedding papers...",
    "reviews":   "Fetching and indexing reviews...",
    "done":      "Complete",
    "failed":    "Failed",
}


@st.fragment(run_every=2)
def _crawl_progress_panel() -> None:
    tracking: list[dict] = st.session_state.get("tracking_crawls", [])
    if not tracking:
        return

    items_with_prog = [(t, get_crawl_progress(t["key"])) for t in tracking]
    items_with_prog = [(t, p) for t, p in items_with_prog if p is not None]
    if not items_with_prog:
        return

    st.subheader("Crawl Progress")
    all_finished = True
    for item, prog in items_with_prog:
        phase = prog["phase"]
        step = prog["step"]
        total = prog["total"]
        count = prog["paper_count"]
        pct = step / total

        if phase == "embedding" and count:
            label = f"Embedding {count:,} papers..."
        elif phase == "reviews" and count:
            label = f"Fetching reviews for {count:,} papers..."
        elif phase == "done" and count:
            label = f"Complete — {count:,} papers indexed"
        elif phase == "done":
            label = "Complete — no papers found"
        elif phase == "failed":
            label = f"Failed: {prog.get('error', 'unknown error')}"
        else:
            label = _PHASE_LABELS.get(phase, phase)

        if phase == "failed":
            st.error(f"**{item['label']}** — {label}")
        else:
            st.progress(pct, text=f"**{item['label']}** — {label}")

        if phase not in ("done", "failed"):
            all_finished = False

    if all_finished:
        col_msg, col_btn = st.columns([6, 1])
        with col_msg:
            st.caption("All crawl jobs finished. Refresh Database Stats to see updated counts.")
        with col_btn:
            if st.button("Dismiss", key="dismiss_crawl_progress"):
                st.session_state["tracking_crawls"] = []
                st.rerun()

    st.divider()


def _track(key: str, label: str) -> None:
    tracking: list[dict] = st.session_state.setdefault("tracking_crawls", [])
    if not any(t["key"] == key for t in tracking):
        tracking.append({"key": key, "label": label})


st.title("Library")

_crawl_progress_panel()

tab_db, tab_crawl, tab_discover = st.tabs(
    ["Database Stats", "Crawl Conference", "Discover & Add Venue"]
)

# -----------------------------------------------------------------------
# Tab 1: Database Stats
# -----------------------------------------------------------------------
with tab_db:
    st.subheader("Local Database Overview")
    if st.button("Refresh"):
        st.cache_data.clear()

    try:
        chroma = get_chroma()
        all_data = chroma.get_all_content_embeddings()
        metas = list(all_data.get("metadatas") or [])

        if not metas:
            st.info("Database is empty. Go to **Crawl Conference** to add papers.")
        else:
            st.metric("Total papers indexed", len(metas))
            df = pd.DataFrame(metas)

            if {"conference", "year"}.issubset(df.columns):
                pivot = df.groupby(["conference", "year"]).size().reset_index(name="count")
                fig = px.bar(
                    pivot, x="year", y="count", color="conference",
                    barmode="group", title="Papers by conference and year",
                )
                st.plotly_chart(fig, use_container_width=True)

            if "decision" in df.columns:
                dec_counts = df["decision"].value_counts().reset_index()
                dec_counts.columns = ["decision", "count"]
                fig2 = px.pie(dec_counts, names="decision", values="count",
                              title="Decision distribution")
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Raw metadata sample (first 20 rows)"):
                st.dataframe(df.head(20))
    except Exception as e:
        st.error(f"Could not load stats: {e}")

# -----------------------------------------------------------------------
# Tab 2: Crawl Conference
# -----------------------------------------------------------------------
with tab_crawl:
    st.subheader("Crawl a Conference into the Database")
    if is_remote_mode():
        st.info(
            "The remote database already contains NeurIPS, ICML, and ICLR (2022-2025). "
            "Crawling here adds data to your **local** database only and does not affect the remote.",
            icon=":material/cloud_done:",
        )

    mode = st.radio(
        "Conference source",
        ["Preset conference", "Custom venue_id"],
        horizontal=True,
    )

    if mode == "Preset conference":
        col1, col2, col3 = st.columns(3)
        with col1:
            conf = st.selectbox("Conference", list(settings.conferences.CONFERENCES.keys()))
        with col2:
            _current_year = datetime.datetime.now().year
            _year_options = list(range(_current_year - 1, 2021, -1))
            year = st.selectbox("Year", _year_options)
        with col3:
            decision = st.selectbox(
                "Decision (optional)",
                ["All", "oral", "spotlight", "poster", "accepted", "rejected"],
            )
        force = st.checkbox("Force re-crawl if already exists")

        if decision != "All":
            st.warning(
                "Filtering by decision excludes rejected papers. "
                "The Diagnosis Agent uses rejected samples to identify common reviewer concerns — "
                "for best results, keep **All** selected.",
                icon=":material/warning:",
            )

        if st.button("Start Crawl", type="primary", key="crawl_preset"):
            pipeline = get_pipeline()
            if not force and pipeline.check_local(conf, year):
                st.success(f"{conf} {year} is already in the database.")
            elif is_crawl_running(conf, year):
                st.warning(f"Crawl for {conf} {year} is already running.")
            else:
                dec_arg = None if decision == "All" else decision
                pipeline.run_async(conf, year, decision=dec_arg)
                crawl_key = f"{conf}_{year}"
                _track(crawl_key, f"{conf} {year}")
                st.success(
                    f"Crawl started for **{conf} {year}**"
                    + (f" ({dec_arg})" if dec_arg else "")
                    + ". Progress shown above."
                )

    else:  # Custom venue_id
        st.caption(
            "Enter any OpenReview venue_id directly. "
            "Use the **Discover & Add Venue** tab first to confirm it exists."
        )
        custom_venue = st.text_input(
            "venue_id",
            placeholder="e.g. aclweb.org/ACL/2024/Conference",
            key="crawl_custom_venue",
        )
        custom_label = st.text_input(
            "Short label (used as conference name in the database)",
            placeholder="e.g. ACL-2024",
            key="crawl_custom_label",
        )

        if st.button("Start Crawl", type="primary", key="crawl_custom"):
            if not custom_venue.strip():
                st.warning("Enter a venue_id.")
            elif not custom_label.strip():
                st.warning("Enter a label for this venue.")
            else:
                pipeline = get_pipeline()
                pipeline.run_async_custom(
                    venue_id=custom_venue.strip(),
                    label=custom_label.strip(),
                )
                crawl_key = f"custom_{custom_venue.strip()}"
                _track(crawl_key, custom_label.strip())
                st.success(
                    f"Background crawl started for `{custom_venue}` "
                    f"(label: **{custom_label}**). Progress shown above."
                )

    st.divider()
    with st.expander("Re-fetch Reviews for an Already-Crawled Conference"):
        st.caption(
            "Use this if papers are already in the database but reviewer comments are missing. "
            "This re-pulls reviews from OpenReview using the current field-mapping logic "
            "without re-crawling papers."
        )
        import os
        from pathlib import Path
        from src.paperradar.config import settings as _settings

        raw_root = Path(_settings.raw_data_dir)
        available_pairs: list[tuple[str, int]] = []
        if raw_root.exists():
            for d in sorted(raw_root.iterdir()):
                if d.is_dir() and (d / "papers.json").exists():
                    parts = d.name.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        available_pairs.append((parts[0].upper(), int(parts[1])))

        if not available_pairs:
            st.info("No locally saved raw data found. Run a crawl first.")
        else:
            pair_labels = [f"{c} {y}" for c, y in available_pairs]
            selected = st.selectbox(
                "Select conference/year to re-fetch reviews",
                options=pair_labels,
                key="refetch_pair",
            )
            idx = pair_labels.index(selected)
            rf_conf, rf_year = available_pairs[idx]

            if st.button("Re-fetch Reviews", key="btn_refetch_reviews"):
                pipeline = get_pipeline()
                pipeline.refetch_reviews_async(
                    conference=rf_conf.lower(),
                    year=rf_year,
                )
                crawl_key = f"refetch_{rf_conf.lower()}_{rf_year}"
                _track(crawl_key, f"{rf_conf} {rf_year} (reviews)")
                st.success(
                    f"Re-fetching reviews for **{rf_conf} {rf_year}** in the background. "
                    "Progress shown above."
                )

    st.divider()
    st.subheader("Crawl Log")
    try:
        conn = sqlite3.connect(settings.sqlite.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT conference, year, decision, paper_count, crawled_at, status "
            "FROM crawl_log ORDER BY crawled_at DESC LIMIT 20"
        ).fetchall()
        conn.close()
        if rows:
            st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True)
        else:
            st.caption("No crawl history yet.")
    except Exception as e:
        st.caption(f"Could not load crawl log: {e}")

# -----------------------------------------------------------------------
# Tab 3: Discover & Add Venue
# -----------------------------------------------------------------------
with tab_discover:
    st.subheader("Discover Any OpenReview Venue")
    st.caption(
        "Probe any venue_id to see available fields and decision patterns, "
        "then crawl it directly from this page."
    )

    venue_input = st.text_input(
        "venue_id",
        placeholder="e.g. aclweb.org/ACL/2024/Conference",
        key="discover_venue_input",
    )

    if st.button("Probe Venue", type="primary"):
        if not venue_input.strip():
            st.warning("Enter a venue_id first.")
        else:
            with st.spinner("Probing OpenReview API..."):
                result = discover_venue(venue_input.strip())
            st.session_state["discover_result"] = result
            st.session_state["discover_venue_id"] = venue_input.strip()

    result = st.session_state.get("discover_result")
    if result is not None:
        if not result.found:
            st.error(f"Not found: {result.error}")
        else:
            info = result.info
            st.success(f"Found **{info.paper_count}** sample papers for `{info.venue_id}`.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Available content fields**")
                st.code(", ".join(info.content_fields))
            with col2:
                st.markdown("**Decision / venue patterns** (use these in search filters)")
                for p in info.decision_patterns:
                    st.markdown(f"- `{p}`")

            if info.review_invitation_pattern:
                st.markdown(f"**Review invitation pattern:** `{info.review_invitation_pattern}`")

            st.markdown("**Sample titles**")
            for t in info.sample_titles:
                st.markdown(f"- {t}")

            with st.expander("Full field summary"):
                st.text(info.filterable_summary)

            st.divider()
            st.subheader("Add to Library")

            disc_label = st.text_input(
                "Label for this venue in the database",
                value=st.session_state.get("discover_venue_id", "").split("/")[0],
                key="discover_label",
            )

            if st.button("Crawl & Index This Venue", type="primary"):
                if not disc_label.strip():
                    st.warning("Enter a label first.")
                else:
                    venue_id = st.session_state["discover_venue_id"]
                    get_pipeline().run_async_custom(
                        venue_id=venue_id,
                        label=disc_label.strip(),
                    )
                    crawl_key = f"custom_{venue_id}"
                    _track(crawl_key, disc_label.strip())
                    st.success(
                        f"Background crawl started for `{venue_id}` "
                        f"(label: **{disc_label}**). "
                        "Check the progress bar above."
                    )
