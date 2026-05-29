import sys
sys.path.insert(0, ".")

import uuid
import streamlit as st

from src.peerlens.agent.reading_runner import (
    get_reading_report,
    resume_discussion,
    start_reading_agent,
)
from src.peerlens.memory.agent_memory import AgentMemoryManager
from src.peerlens.schemas.reading import ReadingReport

st.set_page_config(page_title="Paper Reading Agent", page_icon="📖", layout="wide")
st.title("Paper Reading Agent")
st.caption("Deep-read any paper — with real reviewer perspectives and multi-turn discussion.")

# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
if "reading_thread_id" not in st.session_state:
    st.session_state.reading_thread_id = str(uuid.uuid4())
if "reading_report" not in st.session_state:
    st.session_state.reading_report = None
if "discussion_history" not in st.session_state:
    st.session_state.discussion_history = []
if "reading_started" not in st.session_state:
    st.session_state.reading_started = False
if "topic_candidates" not in st.session_state:
    st.session_state.topic_candidates = []
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None


def _reset_session():
    st.session_state.reading_thread_id = str(uuid.uuid4())
    st.session_state.reading_report = None
    st.session_state.discussion_history = []
    st.session_state.reading_started = False
    st.session_state.topic_candidates = []
    st.session_state.selected_paper = None


def _get_memory_context(query: str) -> str | None:
    try:
        mgr = AgentMemoryManager()
        sessions = mgr.retrieve_relevant(query[:800], n_results=3)
        return mgr.build_memory_context(sessions, "reading") if sessions else None
    except Exception:
        return None


def _save_reading_session(paper_text: str, report: ReadingReport):
    try:
        mgr = AgentMemoryManager()
        mgr.save_session("reading", paper_text, report)
    except Exception:
        pass


# ------------------------------------------------------------------
# Phase 1: Input (only shown before reading starts)
# ------------------------------------------------------------------
if not st.session_state.reading_started:
    tab_pdf, tab_or, tab_ax, tab_topic = st.tabs([
        "Upload PDF", "OpenReview URL", "ArXiv URL", "Topic / Question"
    ])

    input_mode = None
    pdf_bytes = None
    url = ""
    paper_title = ""
    paper_text = ""
    paper_id = ""
    paper_authors = []
    paper_venue = ""
    source_url = ""
    ready_to_run = False

    with tab_pdf:
        uploaded = st.file_uploader("Upload your paper PDF", type="pdf", key="pdf_upload")
        if uploaded:
            pdf_bytes = uploaded.read()
            from src.peerlens.utils.pdf_parser import extract_paper_text, extract_title_abstract
            preview_text = extract_paper_text(pdf_bytes, max_words=500)
            meta = extract_title_abstract(preview_text)
            paper_title = meta.get("title", "")
            st.info(f"Detected title: **{paper_title}**" if paper_title else "Title not detected.")
        if st.button("Read this PDF", disabled=not pdf_bytes, key="btn_pdf"):
            input_mode = "pdf"
            ready_to_run = True

    with tab_or:
        or_url = st.text_input(
            "OpenReview forum URL",
            placeholder="https://openreview.net/forum?id=xxxxx",
            key="or_url",
        )
        if st.button("Read this paper", disabled=not or_url.strip(), key="btn_or"):
            input_mode = "openreview_url"
            url = or_url.strip()
            ready_to_run = True

    with tab_ax:
        ax_url = st.text_input(
            "ArXiv URL or ID",
            placeholder="https://arxiv.org/abs/2301.xxxxx  or  2301.xxxxx",
            key="ax_url",
        )
        if st.button("Read this paper", disabled=not ax_url.strip(), key="btn_ax"):
            input_mode = "arxiv_url"
            url = ax_url.strip()
            ready_to_run = True

    with tab_topic:
        topic_query = st.text_input(
            "Describe a topic or question",
            placeholder="e.g. efficient attention for long sequences",
            key="topic_query",
        )
        find_btn = st.button("Find Papers", disabled=not topic_query.strip(), key="btn_find")

        if find_btn and topic_query.strip():
            with st.spinner("Searching local database..."):
                try:
                    from src.peerlens.agent.tools import search_papers as _local_search
                    from src.peerlens.agent.tools_remote import resolve_tool
                    from src.peerlens.schemas.tools import SearchPapersOutput
                    _search = resolve_tool("search_papers", _local_search)
                    raw = _search.invoke({"query": topic_query.strip(), "top_k": 5})
                    results = SearchPapersOutput(**raw) if isinstance(raw, dict) else raw
                    st.session_state.topic_candidates = results.results[:5]
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state.topic_candidates = []

        if st.session_state.topic_candidates:
            st.markdown("**Select a paper to read:**")
            for i, p in enumerate(st.session_state.topic_candidates):
                with st.container(border=True):
                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        decision_badge = f" `{p.decision}`" if p.decision and p.decision != "unknown" else ""
                        st.markdown(f"**{p.title}**{decision_badge}")
                        st.caption(f"{p.conference} {p.year}  ·  {', '.join(p.authors[:3])}")
                        st.write(p.abstract[:250] + "..." if len(p.abstract) > 250 else p.abstract)
                    with col_btn:
                        if st.button("Read", key=f"select_paper_{i}"):
                            st.session_state.selected_paper = p

        if st.session_state.selected_paper:
            p = st.session_state.selected_paper
            st.success(f"Selected: **{p.title}**")
            if st.button("Read this paper", key="btn_topic_go"):
                input_mode = "topic_query_selected"
                paper_title = p.title
                paper_text = p.abstract
                paper_id = p.paper_id
                paper_authors = p.authors
                paper_venue = f"{p.conference} {p.year}"
                source_url = p.forum_url
                ready_to_run = True

    # ------------------------------------------------------------------
    # Launch reading agent
    # ------------------------------------------------------------------
    if ready_to_run and input_mode:
        st.session_state.reading_started = True
        query_for_memory = (
            paper_title or paper_text[:200] or url or topic_query or ""
        )
        memory_context = _get_memory_context(query_for_memory)

        with st.status("Analyzing paper...", expanded=True) as status_box:
            for event in start_reading_agent(
                input_mode=input_mode,
                thread_id=st.session_state.reading_thread_id,
                pdf_bytes=pdf_bytes if input_mode == "pdf" else None,
                url=url,
                paper_title=paper_title,
                paper_text=paper_text,
                paper_id=paper_id,
                paper_authors=paper_authors,
                paper_venue=paper_venue,
                source_url=source_url,
                memory_context=memory_context,
            ):
                if isinstance(event, dict):
                    if "error" in event:
                        st.error(event["error"])
                        break
                    node = event.get("active_node", "")
                    if node == "parse_input":
                        st.write("Parsing input...")
                    elif node in ("fetch_openreview", "fetch_arxiv"):
                        st.write("Fetching paper metadata...")
                    elif node == "inject_reviews":
                        n_reviews = len(event.get("paper_reviews", []))
                        if n_reviews:
                            st.write(f"Loaded {n_reviews} reviewer comments.")
                        else:
                            st.write("No reviews found in local database.")
                    elif node == "deep_read":
                        st.write("Generating deep reading report...")
                    report = event.get("report")
                    if report:
                        st.session_state.reading_report = report
            status_box.update(label="Reading complete.", state="complete", expanded=False)

        # Save session async
        if st.session_state.reading_report:
            input_text = (
                paper_title or paper_text[:300] or url
                or (st.session_state.selected_paper.title if st.session_state.selected_paper else "")
            )
            _save_reading_session(input_text, st.session_state.reading_report)

        st.rerun()

# ------------------------------------------------------------------
# Phase 2: Display reading report
# ------------------------------------------------------------------
report: ReadingReport | None = st.session_state.reading_report

if report:
    # Header
    col_title, col_reset = st.columns([6, 1])
    with col_title:
        st.subheader(report.paper_title)
        venue_authors = " · ".join(filter(None, [
            report.venue,
            ", ".join(report.authors[:3]) + ("..." if len(report.authors) > 3 else ""),
        ]))
        if venue_authors:
            st.caption(venue_authors)
        if report.source_url:
            st.link_button("View Original Paper", url=report.source_url)
    with col_reset:
        if st.button("New Reading", key="btn_reset"):
            _reset_session()
            st.rerun()

    # TL;DR
    if report.tldr:
        st.info(f"**TL;DR:** {report.tldr}")

    st.divider()

    # Two-column layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        if report.problem_statement:
            st.markdown("### Problem")
            st.write(report.problem_statement)

        if report.core_contributions:
            st.markdown("### Core Contributions")
            for c in report.core_contributions:
                st.markdown(f"- {c}")

        if report.methodology_summary:
            st.markdown("### Methodology")
            st.write(report.methodology_summary)

        if report.key_innovations:
            st.markdown("**Key Innovations**")
            for inn in report.key_innovations:
                st.markdown(f"- {inn}")

    with col_right:
        if report.datasets_and_baselines:
            st.markdown("### Experiments")
            st.write(report.datasets_and_baselines)

        if report.main_results:
            st.markdown("**Main Results**")
            st.write(report.main_results)

        if report.ablations:
            with st.expander("Ablations"):
                st.write(report.ablations)

        if report.limitations:
            st.markdown("### Limitations")
            for lim in report.limitations:
                st.markdown(f"- {lim}")

        if report.open_questions:
            st.markdown("### Open Questions")
            for q in report.open_questions:
                st.markdown(f"- {q}")

    # Reviewer perspectives
    if report.reviewer_perspectives:
        st.divider()
        st.markdown("### Reviewer Perspectives")
        _STANCE_COLORS = {"positive": "green", "negative": "red", "mixed": "orange"}
        for rp in report.reviewer_perspectives:
            color = _STANCE_COLORS.get(rp.stance, "gray")
            with st.container(border=True):
                st.markdown(f"**{rp.reviewer_id}** · :{color}[{rp.stance.upper()}]")
                for pt in rp.key_points:
                    st.markdown(f"- {pt}")

    # Memory connections
    if report.memory_connections:
        st.divider()
        st.markdown("### Connections to Your Research History")
        for conn in report.memory_connections:
            _AGENT_ICONS = {"diagnosis": "🔬", "research": "📚", "reading": "📖"}
            icon = _AGENT_ICONS.get(conn.agent_type, "")
            ts = conn.timestamp.strftime("%Y-%m-%d") if conn.timestamp else ""
            with st.container(border=True):
                st.markdown(f"{icon} **{conn.agent_type.title()} Session** · {ts}")
                st.write(conn.connection_description)
                if conn.related_input_summary:
                    with st.expander("Session context"):
                        st.caption(conn.related_input_summary)

    # ------------------------------------------------------------------
    # Phase 3: Discussion
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Academic Discussion")
    st.caption(
        "Ask follow-up questions about this paper. "
        "The agent has access to the full paper, reviewer comments, and your research history. "
        "Type `exit` to end the discussion."
    )

    # Display conversation history
    for msg in st.session_state.discussion_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if user_input := st.chat_input("Your question about this paper..."):
        if user_input.strip().lower() in {"exit", "quit", "done", "bye", "结束", "退出"}:
            st.info("Discussion ended.")
        else:
            # Show user message immediately
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.discussion_history.append({"role": "user", "content": user_input})

            # Stream AI response
            reply_parts = []
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for event in resume_discussion(
                    st.session_state.reading_thread_id, user_input
                ):
                    if isinstance(event, dict):
                        if "error" in event:
                            placeholder.error(event["error"])
                            break
                        msgs = event.get("messages", [])
                        if msgs:
                            last = msgs[-1]
                            content = getattr(last, "content", "")
                            if content and not getattr(last, "type", "") == "human":
                                reply_parts = [content]
                                placeholder.write(content)
                if reply_parts:
                    st.session_state.discussion_history.append({
                        "role": "assistant", "content": reply_parts[-1]
                    })
            st.rerun()
