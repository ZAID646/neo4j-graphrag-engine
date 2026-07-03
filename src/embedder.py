from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import Chunk
from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    model = _load_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> list[float]:
    model = _load_model()
    return model.encode(query).tolist()


def embedding_dimension() -> int:
    model = _load_model()
    return model.get_sentence_embedding_dimension()
