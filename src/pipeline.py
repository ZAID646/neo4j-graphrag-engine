from pathlib import Path

from src.ingest import load_documents
from src.embedder import embed_chunks
from src.extractor import extract_knowledge as extract_knowledge_fn
from src.graph_store import store_graph, close as close_graph
from src.vector_store import upsert_embeddings, close as close_vector
from src.retriever import hybrid_retrieve
from src.synthesizer import synthesize
from src.models import QueryResult


def ingest(docs_dir: str | Path):
    print(f"Ingesting documents from {docs_dir}...")

    chunks = load_documents(docs_dir)
    print(f"  Loaded {len(chunks)} chunks")

    print("  Extracting entities and relationships...")
    entities, relationships = extract_knowledge_fn(chunks)
    print(f"  Found {len(entities)} entities, {len(relationships)} relationships")

    print("  Storing graph in Neo4j...")
    store_graph(entities, relationships)
    print("  Graph stored")

    print("  Generating embeddings...")
    vectors = embed_chunks(chunks)
    print(f"  Generated {len(vectors)} embeddings")

    print("  Upserting to Pinecone...")
    upsert_embeddings(chunks, vectors)
    print("  Embeddings stored")

    close_graph()
    print("Ingestion complete.")


def query(query_str: str) -> QueryResult:
    print(f"Query: {query_str}")

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
