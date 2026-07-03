from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from src.models import Chunk
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from src.embedder import embedding_dimension


_pc: Optional[Pinecone] = None
_index = None


def _get_pc() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def _ensure_index():
    global _index
    if _index is not None:
        return _index

    pc = _get_pc()
    dim = embedding_dimension()

    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def upsert_embeddings(chunks: list[Chunk], vectors: list[list[float]]):
    index = _ensure_index()
    records = []
    for chunk, vec in zip(chunks, vectors):
        records.append({
            "id": chunk.id,
            "values": vec,
            "metadata": {"text": chunk.text[:1000], "source": chunk.source},
        })
    batch_size = 100
    for i in range(0, len(records), batch_size):
        index.upsert(records[i:i + batch_size])


def semantic_search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    index = _ensure_index()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [{"id": m.id, "score": m.score, "text": m.metadata.get("text", "")}
            for m in result.matches]


def close():
    global _pc, _index
    _index = None
    _pc = None
