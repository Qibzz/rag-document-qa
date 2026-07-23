"""Embedding package for the RAG project."""

from .embedder import DEFAULT_MODEL_NAME, Embedder, EmbeddingError, ModelLoadError

__all__ = [
    "DEFAULT_MODEL_NAME",
    "Embedder",
    "EmbeddingError",
    "ModelLoadError",
]
