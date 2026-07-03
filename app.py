import streamlit as st
from src.dashboard import render_ui

st.set_page_config(
    page_title="Neo4j GraphRAG Engine",
    page_icon=":cyclone:",
    layout="wide",
)

render_ui()
