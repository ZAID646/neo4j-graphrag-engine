from src.embedder import embed_query
from src.vector_store import semantic_search
from src.graph_store import get_connected_entities, extract_entity_names


def hybrid_retrieve(query: str, top_k: int = 5) -> dict:
    query_vec = embed_query(query)

    vector_results = semantic_search(query_vec, top_k=top_k)

    all_entities = extract_entity_names()

    chunks_text = [r["text"] for r in vector_results]

    graph_context = []
    for entity_name in all_entities:
        if entity_name.lower() in query.lower():
            connected = get_connected_entities(entity_name, depth=2)
            for c in connected:
                if c.get("name"):
                    graph_context.append(f"{c['name']} ({c.get('type', '?')})")
            break

    return {
        "vector_context": chunks_text,
        "graph_context": list(set(graph_context))[:20],
    }
