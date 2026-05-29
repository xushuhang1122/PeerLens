import streamlit as st

import sys
sys.path.insert(0, ".")

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.peerlens.analysis.temporal import TemporalAnalyzer
from src.peerlens.analysis.clustering import ReviewClusterer
from src.peerlens.analysis.gap_detector import GapDetector
from src.peerlens.schemas.tools import (
    AnalyzeTemporalInput, ClusterReviewsInput, IdentifyGapsInput
)
from src.peerlens.config import settings

@st.cache_resource
def get_temporal():
    return TemporalAnalyzer()

@st.cache_resource
def get_clusterer():
    return ReviewClusterer()

@st.cache_resource
def get_gap():
    return GapDetector()


st.title("📊 Analysis")

tab_temporal, tab_cluster, tab_gap = st.tabs(
    ["Temporal Trends", "Review Clustering", "Research Gaps"]
)

# -----------------------------------------------------------------------
# Tab 1: Temporal Trends
# -----------------------------------------------------------------------
with tab_temporal:
    st.subheader("Topic Trends Over Time")
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic", "reinforcement learning from human feedback", key="t_topic")
    with col2:
        conf_sel = st.multiselect(
            "Conferences",
            list(settings.conferences.CONFERENCES.keys()),
            default=["NeurIPS", "ICML", "ICLR"],
            key="t_conf",
        )

    year_sel = st.multiselect(
        "Years", list(range(2025, 2021, -1)), default=[2022, 2023, 2024, 2025], key="t_year"
    )

    if st.button("Analyze Trends", type="primary", key="btn_temporal"):
        with st.spinner("Analyzing..."):
            try:
                result = get_temporal().analyze(
                    AnalyzeTemporalInput(topic=topic, conferences=conf_sel, years=year_sel)
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.caption(result.summary)

        if result.distribution:
            df = pd.DataFrame([b.model_dump() for b in result.distribution])

            fig_line = px.line(
                df,
                x="year",
                y="paper_count",
                color="conference",
                markers=True,
                title=f'Papers per year: "{topic}"',
                labels={"paper_count": "Papers", "year": "Year"},
            )
            st.plotly_chart(fig_line, use_container_width=True)

            fig_bar = px.bar(
                df,
                x="year",
                y=["oral_count", "spotlight_count", "poster_count"],
                barmode="stack",
                facet_col="conference",
                title="Decision breakdown by year",
                labels={"value": "Papers", "variable": "Decision"},
                color_discrete_map={
                    "oral_count": "#e74c3c",
                    "spotlight_count": "#e67e22",
                    "poster_count": "#3498db",
                },
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.metric("Trend", result.trend.capitalize())
            if result.peak_year:
                st.metric("Peak", f"{result.peak_conference} {result.peak_year}")
        else:
            st.info("No data found for this topic/filter combination.")

# -----------------------------------------------------------------------
# Tab 2: Review Clustering
# -----------------------------------------------------------------------
with tab_cluster:
    st.subheader("Review Comment Clustering")
    st.caption("Groups reviewer comments by K-Means to surface common criticism patterns.")

    col1, col2 = st.columns([3, 1])
    with col1:
        area = st.text_input("Primary area", "reinforcement_learning", key="c_area")
    with col2:
        n_clust = st.slider("Clusters", 2, 10, 5, key="c_n")

    if st.button("Run Clustering", type="primary", key="btn_cluster"):
        with st.spinner("Clustering reviews..."):
            try:
                result = get_clusterer().cluster(
                    ClusterReviewsInput(primary_area=area, n_clusters=n_clust)
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

        if result.n_clusters == 0:
            st.warning("Not enough review data for this area.")
            st.stop()

        if result.high_frequency_rejections:
            st.error(
                "**High-frequency rejection patterns:** "
                + " · ".join(result.high_frequency_rejections)
            )

        for c in result.clusters:
            with st.expander(
                f"Cluster {c.cluster_id}: {c.label or 'Unlabeled'} "
                f"({c.paper_count} papers"
                + (f", avg rating {c.avg_rating:.1f}" if c.avg_rating else "")
                + ")"
            ):
                st.markdown(f"**Criticism pattern:** {c.criticism_pattern}")
                st.markdown(f"**Top terms:** {', '.join(c.top_terms)}")
                st.markdown("**Representative excerpts:**")
                for q in c.representative_quotes:
                    st.markdown(f"> {q[:300]}")

        # Rating distribution chart
        ratings_data = [
            {"cluster": f"C{c.cluster_id}: {c.label[:20] if c.label else ''}", "rating": c.avg_rating}
            for c in result.clusters if c.avg_rating is not None
        ]
        if ratings_data:
            fig = px.bar(
                ratings_data,
                x="cluster",
                y="rating",
                title="Average rating per cluster",
                color="rating",
                color_continuous_scale="RdYlGn",
                range_color=[1, 10],
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Tab 3: Research Gaps
# -----------------------------------------------------------------------
with tab_gap:
    st.subheader("Research Gap Detection")
    st.caption(
        "Identifies under-explored topic clusters and common reviewer rejection patterns."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        domain = st.text_input("Domain", "reinforcement_learning", key="g_domain")
    with col2:
        min_density = st.slider("Min coverage density", 0.05, 0.3, 0.1, step=0.05, key="g_dens")

    if st.button("Detect Gaps", type="primary", key="btn_gap"):
        with st.spinner("Detecting research gaps..."):
            try:
                result = get_gap().detect(
                    IdentifyGapsInput(domain=domain, min_cluster_density=min_density)
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

        if result.rejection_patterns:
            st.warning(
                "**Common rejection reasons:** " + " · ".join(result.rejection_patterns)
            )

        col_cov, col_sparse = st.columns(2)
        with col_cov:
            st.markdown("**Covered areas**")
            for a in result.covered_areas:
                st.markdown(f"- {a}")
        with col_sparse:
            st.markdown("**Sparse / under-explored**")
            for a in result.sparse_areas:
                st.markdown(f"- {a}")

        if result.gaps:
            st.subheader("Identified Gaps")
            for g in result.gaps:
                with st.container(border=True):
                    st.markdown(f"**{g.gap_description}**")
                    st.markdown(f"*Suggested angle:* {g.suggested_angle}")
                    if g.evidence:
                        st.caption("Evidence: " + " | ".join(g.evidence[:3]))

        if result.submission_advice:
            st.info(f"**Submission advice:** {result.submission_advice}")
