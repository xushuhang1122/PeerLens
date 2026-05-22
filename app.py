import streamlit as st

st.set_page_config(
    page_title="PeerLens",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
