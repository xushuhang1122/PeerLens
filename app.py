import streamlit as st
from src.paperradar.config import settings

st.set_page_config(
    page_title="PeerLens",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data source selector (persists across pages via session_state) ─────────────
_default_url = settings.remote_mcp.url or ""
if "remote_mcp_url" not in st.session_state:
    st.session_state["remote_mcp_url"] = _default_url or None

with st.sidebar:
    st.caption("Data Source")
    _use_remote = st.toggle(
        "Remote MCP",
        value=bool(st.session_state["remote_mcp_url"]),
        help="Use the shared cloud database instead of a local one.",
    )
    if _use_remote:
        _url = st.text_input(
            "MCP URL",
            value=st.session_state["remote_mcp_url"] or _default_url,
            placeholder="http://your-server:8765/mcp",
            label_visibility="collapsed",
        )
        st.session_state["remote_mcp_url"] = _url.strip() or None
    else:
        st.session_state["remote_mcp_url"] = None
    st.divider()

pg = st.navigation(
    {
        "": [
            st.Page("pages/home.py", title="Home", icon=":material/home:", default=True),
        ],
        "Agents": [
            st.Page("pages/2_Agent.py", title="Research Agent", icon=":material/manage_search:"),
            st.Page("pages/6_Diagnose.py", title="Diagnose Paper", icon=":material/clinical_notes:"),
        ],
        "Tools": [
            st.Page("pages/1_Search.py", title="Search Papers", icon=":material/search:"),
            st.Page("pages/3_Analysis.py", title="Analysis", icon=":material/bar_chart:"),
            st.Page("pages/4_Library.py", title="Library", icon=":material/library_books:"),
            st.Page("pages/5_Memory.py", title="My Profile", icon=":material/person:"),
        ],
    }
)
pg.run()
