from pathlib import Path

import streamlit as st

from src.pipeline import ingest, query


def render_ui():
    st.title("Neo4j GraphRAG Engine")
    st.markdown(
        "Ingest documents, extract entities and relationships into Neo4j, "
        "embed into Pinecone, and query via hybrid retrieval (vector + graph)."
    )

    tab1, tab2 = st.tabs(["Query", "Ingest Documents"])

    with tab1:
        q = st.text_input("Ask a question about your documents:", key="query_input")

        if st.button("Search", type="primary", disabled=not q):
            with st.spinner("Running hybrid retrieval..."):
                try:
                    result = query(q)
                    st.session_state["query_result"] = result
                except Exception as e:
                    st.error(f"Query failed: {e}")

        if "query_result" in st.session_state:
            r = st.session_state["query_result"]

            st.subheader("Answer")
            st.write(r.answer)

            with st.expander("Vector Context (top chunks)"):
                for i, ctx in enumerate(r.vector_context[:5]):
                    st.caption(f"Chunk {i + 1}")
                    st.text(ctx[:500])

            with st.expander("Graph Context (connected entities)"):
                for ctx in r.graph_context[:20]:
                    st.write(f"- {ctx}")

    with tab2:
        docs_path = st.text_input("Documents directory", value="data/sample_docs")

        if st.button("Ingest", type="secondary"):
            _run_ingestion(docs_path)

        _render_sample_docs()
            _run_ingestion(docs_path)


def _render_sample_docs():
    docs_dir = Path("data/sample_docs")
    if not docs_dir.exists():
        return

    md_files = sorted(docs_dir.glob("*.md"))
    txt_files = sorted(docs_dir.glob("*.txt"))
    files = md_files + txt_files

    if not files:
        return

    st.divider()
    st.subheader("Source Documents")

    for f in files:
        content = f.read_text(encoding="utf-8")
        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f"<div style='border:1px solid #e2e8f0;border-radius:8px;"
            f"padding:12px 16px;margin-bottom:8px;max-height:200px;"
            f"overflow-y:auto;font-size:13px;'>"
            f"<b>{f.name}</b><br>{escaped}</div>"
        )
        st.markdown(html, unsafe_allow_html=True)


def _run_ingestion(docs_path: str):
    with st.status("Preparing ingestion...", expanded=True) as status:
        def on_msg(msg: str):
            if msg.startswith("extracting:"):
                status.update(label=msg, state="running")
            else:
                status.update(label=msg, state="running")

        try:
            ingest(docs_path, status_callback=on_msg)
            status.update(label="Ingestion complete", state="complete")
            st.success("All data ingested successfully.")
        except Exception as e:
            status.update(label=f"Ingestion failed: {e}", state="error")
            st.error(f"Ingestion failed: {e}")
