from pathlib import Path

from src.ingest import load_documents
from src.embedder import embed_chunks
from src.extractor import extract_knowledge as extract_knowledge_fn
from src.graph_store import store_graph, close as close_graph
from src.vector_store import upsert_embeddings
from src.retriever import hybrid_retrieve
from src.synthesizer import synthesize
from src.models import QueryResult


def ingest(docs_dir: str | Path, status_callback=None):
    def report(msg):
        if status_callback:
            status_callback(msg)

    report("Loading documents...")
    chunks = load_documents(docs_dir)
    report(f"Loaded {len(chunks)} chunks")

    report("Extracting entities and relationships...")
    entities, relationships = extract_knowledge_fn(chunks, status_callback=status_callback)
    report(f"Found {len(entities)} entities, {len(relationships)} relationships")

    report("Storing graph in Neo4j...")
    store_graph(entities, relationships)
    report("Graph stored")

    report("Generating embeddings (this loads the ML model)...")
    vectors = embed_chunks(chunks)
    report(f"Generated {len(vectors)} embeddings")

    report("Upserting to Pinecone...")
    upsert_embeddings(chunks, vectors)
    report("Embeddings stored")

    close_graph()


def query(query_str: str) -> QueryResult:
    retrieved = hybrid_retrieve(query_str)

    answer = synthesize(
        query=query_str,
        vector_context=retrieved["vector_context"],
        graph_context=retrieved["graph_context"],
    )

    result = QueryResult(
        query=query_str,
        answer=answer,
        vector_context=retrieved["vector_context"],
        graph_context=retrieved["graph_context"],
        entities=[],
        relationships=[],
    )

    close_graph()
    return result
