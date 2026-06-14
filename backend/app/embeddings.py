from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def model_loaded() -> bool:
    return get_embedding_model.cache_info().currsize > 0


def encode_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.matmul(vectors, vectors.T)


def top_k(query: str, chunks: list[dict], k: int) -> list[dict]:
    if not chunks:
        return []
    chunk_vectors = np.vstack([chunk["embedding"] for chunk in chunks])
    query_vector = encode_texts([query])[0]
    scores = np.matmul(chunk_vectors, query_vector)
    order = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in order]
