import datetime
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

import sys
sys.path.insert(0, ".")

from src.paperradar.config import settings
from src.paperradar.crawl.discover import discover_venue
from src.paperradar.crawl.pipeline import CrawlPipeline, is_crawl_running
from src.paperradar.store.chroma import ChromaManager


@st.cache_resource
def get_chroma():
    return ChromaManager()


@st.cache_resource
def get_pipeline():
    return CrawlPipeline()


st.title("📚 Library")

tab_db, tab_crawl, tab_discover = st.tabs(
    ["Database Stats", "Crawl Conference", "Discover & Add Venue"]
)

# -----------------------------------------------------------------------
# Tab 1: Database Stats
# -----------------------------------------------------------------------
with tab_db:
    st.subheader("Local Database Overview")
    if st.button("Refresh", icon="🔄"):
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

        if st.button("Start Crawl", type="primary", key="crawl_preset"):
            pipeline = get_pipeline()
            if not force and pipeline.check_local(conf, year):
                st.success(f"{conf} {year} is already in the database.")
            elif is_crawl_running(conf, year):
                st.warning(f"Crawl for {conf} {year} is already running.")
            else:
                dec_arg = None if decision == "All" else decision
                pipeline.run_async(conf, year, decision=dec_arg)
                st.success(
                    f"Crawl started for **{conf} {year}**"
                    + (f" ({dec_arg})" if dec_arg else "")
                    + ". Check stats in a few minutes."
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
                get_pipeline().run_async_custom(
                    venue_id=custom_venue.strip(),
                    label=custom_label.strip(),
                )
                st.success(
                    f"Background crawl started for `{custom_venue}` "
                    f"(label: **{custom_label}**). Check stats in a few minutes."
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

            if st.button("Crawl & Index This Venue", type="primary", icon="⬇️"):
                if not disc_label.strip():
                    st.warning("Enter a label first.")
                else:
                    get_pipeline().run_async_custom(
                        venue_id=st.session_state["discover_venue_id"],
                        label=disc_label.strip(),
                    )
                    st.success(
                        f"Background crawl started for `{st.session_state['discover_venue_id']}` "
                        f"(label: **{disc_label}**). "
                        "Check **Database Stats** in a few minutes."
                    )
