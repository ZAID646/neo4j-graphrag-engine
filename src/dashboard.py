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
            with st.spinner("Ingesting documents..."):
                try:
                    ingest(docs_path)
                    st.success("Ingestion complete")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
