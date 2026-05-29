import streamlit as st

import sys
sys.path.insert(0, ".")

from src.peerlens.retrieval.hybrid_search import HybridSearcher
from src.peerlens.schemas.tools import SearchPapersInput
from src.peerlens.config import settings
from src.peerlens.memory.episodic import EpisodicMemory
from src.peerlens.memory.semantic import SemanticMemory
from src.peerlens.schemas.paper import Paper

_DECISION_COLORS = {
    "oral":      "#e74c3c",
    "spotlight": "#e67e22",
    "poster":    "#3498db",
    "accepted":  "#27ae60",
    "rejected":  "#7f8c8d",
    "unknown":   "#bdc3c7",
}

@st.cache_resource
def get_searcher():
    return HybridSearcher()

@st.cache_resource
def get_episodic():
    return EpisodicMemory()

@st.cache_resource
def get_semantic():
    return SemanticMemory()


def decision_badge(decision: str) -> str:
    color = _DECISION_COLORS.get(decision, "#bdc3c7")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.75em;font-weight:bold">{decision.upper()}</span>'


st.title("🔍 Search Papers")

query = st.text_input(
    "Search query",
    placeholder="e.g. efficient attention mechanisms for long sequences",
)

with st.expander("Filters", expanded=False):
    all_conferences = list(settings.conferences.CONFERENCES.keys())
    col1, col2, col3, col4 = st.columns([3, 2, 3, 1])
    with col1:
        conf_filter = st.multiselect("Conference", all_conferences)
    with col2:
        year_filter = st.multiselect("Year", list(range(2025, 2021, -1)))
    with col3:
        decision_filter = st.multiselect(
            "Decision", ["oral", "spotlight", "poster", "accepted", "rejected"]
        )
    with col4:
        top_k = st.number_input("Top K", min_value=5, max_value=200, value=20, step=5)

search_btn = st.button("Search", type="primary", use_container_width=True)

if search_btn and query.strip():
    searcher = get_searcher()
    episodic = get_episodic()
    episodic.record_query(query.strip())

    with st.spinner("Searching..."):
        inp = SearchPapersInput(
            query=query.strip(),
            decision_filter=decision_filter or None,
            conference_filter=conf_filter or None,
            year_filter=year_filter or None,
            top_k=top_k,
        )
        try:
            output = searcher.search(inp)
        except Exception as e:
            st.error(f"Search error: {e}")
            st.stop()

    st.caption(f"Found **{output.total_found}** results for: *{query}*")

    if not output.results:
        st.info("No papers found. Try broadening your query or removing filters.")
        st.stop()

    semantic = get_semantic()

    for r in output.results:
        with st.container(border=True):
            cols = st.columns([8, 2])
            with cols[0]:
                st.markdown(
                    f"**[{r.title}]({r.forum_url})**",
                    unsafe_allow_html=False,
                )
                authors_str = ", ".join(r.authors[:4]) + (" et al." if len(r.authors) > 4 else "")
                st.caption(f"{authors_str}")
                st.markdown(
                    decision_badge(r.decision)
                    + f"&nbsp;&nbsp;**{r.conference} {r.year}**"
                    + f"&nbsp;&nbsp;RRF: `{r.rrf_score:.4f}`",
                    unsafe_allow_html=True,
                )
                if r.primary_area:
                    st.caption(f"Area: {r.primary_area}")
            with cols[1]:
                like_key = f"like_{r.paper_id}"
                if st.button("👍 Like", key=like_key):
                    episodic = get_episodic()
                    episodic.record_feedback(r.paper_id, r.title, "up")
                    paper = Paper(
                        id=r.paper_id,
                        title=r.title,
                        abstract=r.abstract,
                        authors=r.authors,
                        keywords=r.keywords,
                        conference=r.conference,
                        year=r.year,
                        decision=r.decision,  # type: ignore[arg-type]
                        forum_url=r.forum_url,
                    )
                    semantic.add_liked_paper(paper)
                    st.toast(f"Liked: {r.title[:40]}...")
                if st.button("👎", key=f"dislike_{r.paper_id}"):
                    episodic.record_feedback(r.paper_id, r.title, "down")
                    st.toast("Noted.")
