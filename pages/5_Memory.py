import streamlit as st

import sys
sys.path.insert(0, ".")

from src.paperradar.memory.episodic import EpisodicMemory
from src.paperradar.memory.semantic import SemanticMemory
from src.paperradar.memory.push_engine import PushEngine

@st.cache_resource
def get_episodic():
    return EpisodicMemory()

@st.cache_resource
def get_semantic():
    return SemanticMemory()

@st.cache_resource
def get_push():
    return PushEngine()


st.title("👤 My Profile")

tab_history, tab_prefs, tab_push = st.tabs(
    ["Query History", "Liked Papers", "New Recommendations"]
)

# -----------------------------------------------------------------------
# Tab 1: Query History
# -----------------------------------------------------------------------
with tab_history:
    st.subheader("Recent Queries")
    try:
        queries = get_episodic().get_recent_queries(30)
        if queries:
            for i, q in enumerate(queries, 1):
                st.markdown(f"{i}. {q}")
        else:
            st.info("No queries yet. Start searching!")
    except Exception as e:
        st.error(str(e))

# -----------------------------------------------------------------------
# Tab 2: Liked Papers
# -----------------------------------------------------------------------
with tab_prefs:
    st.subheader("Liked Papers")

    try:
        liked_ids = get_episodic().get_liked_paper_ids(50)
        if liked_ids:
            from src.paperradar.store.chroma import ChromaManager
            chroma = ChromaManager()  # returns singleton
            data = chroma.get_content_by_ids(liked_ids)
            ids = data.get("ids") or []
            metas = data.get("metadatas") or []
            for pid, meta in zip(ids, metas):
                with st.container(border=True):
                    title = meta.get("title", pid)
                    forum_url = meta.get("forum_url", "")
                    conf = meta.get("conference", "")
                    year = meta.get("year", "")
                    dec = meta.get("decision", "")
                    st.markdown(f"**[{title}]({forum_url})**" if forum_url else f"**{title}**")
                    st.caption(f"{conf} {year} · {dec}")
        else:
            st.info("No liked papers yet. Use 👍 in Search to build your profile.")
    except Exception as e:
        st.error(str(e))

    st.divider()
    st.subheader("Add a Topic to Preferences")
    topic_input = st.text_input(
        "Topic description",
        placeholder="e.g. efficient transformers for long-context reasoning",
    )
    if st.button("Add Topic"):
        if topic_input.strip():
            get_semantic().add_liked_topic(topic_input.strip())
            st.success(f"Added topic: {topic_input}")
        else:
            st.warning("Enter a topic first.")

    st.caption(get_semantic().get_preference_summary())

# -----------------------------------------------------------------------
# Tab 3: New Recommendations
# -----------------------------------------------------------------------
with tab_push:
    st.subheader("New Paper Recommendations")
    st.caption(
        "Papers recently added to the database that match your research interests."
    )

    top_k = st.slider("How many recommendations", 5, 30, 10)

    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Matching against your preferences..."):
            try:
                results = get_push().run_push_check(top_k=top_k)
            except Exception as e:
                st.error(str(e))
                st.stop()

        if not results:
            st.info(
                "No recommendations yet. Like some papers first to train your profile, "
                "or crawl new conference data."
            )
        else:
            st.caption(f"Top {len(results)} papers matching your interests:")
            for r in results:
                with st.container(border=True):
                    st.markdown(
                        f"**[{r.title}]({r.forum_url})**" if r.forum_url else f"**{r.title}**"
                    )
                    authors_str = ", ".join(r.authors[:3]) + (" et al." if len(r.authors) > 3 else "")
                    st.caption(f"{authors_str} · {r.conference} {r.year} · {r.decision}")
                    st.caption(f"Similarity score: {r.rrf_score:.4f}")
