"""Retrieval utilities for the lightweight RAG pipeline.

This module loads saved embeddings and chunk metadata from the embed step,
then ranks the most relevant chunks for a user query using cosine similarity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Base exception for retrieval problems."""


class RetrievalIndexError(RetrievalError):
    """Raised when an embedding index cannot be loaded or validated."""


class Retriever:
    """Rank stored document chunks using cosine similarity."""

    def __init__(self, embeddings_path: str | Path, metadata_path: str | Path | None = None):
        """Initialize the retriever from stored embedding artifacts.

        Args:
            embeddings_path: Path to the `.npy` embedding matrix.
            metadata_path: Path to the JSON metadata file. When omitted, it is
                inferred from the same directory as the embedding file.

        Raises:
            RetrievalIndexError: If the index artifacts are missing or invalid.
        """
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.embeddings_path.with_name("chunk_metadata.json")

        if not self.embeddings_path.exists():
            raise RetrievalIndexError(f"Embedding file not found: {self.embeddings_path}")
        if not self.metadata_path.exists():
            raise RetrievalIndexError(f"Metadata file not found: {self.metadata_path}")

        try:
            self.embeddings = np.load(self.embeddings_path)
        except Exception as exc:
            raise RetrievalIndexError(f"Unable to load embedding matrix: {self.embeddings_path}") from exc

        try:
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        except Exception as exc:
            raise RetrievalIndexError(f"Unable to load metadata file: {self.metadata_path}") from exc

        if self.embeddings.ndim != 2:
            raise RetrievalIndexError("The embedding matrix must be a 2D NumPy array.")
        if len(self.metadata) != len(self.embeddings):
            raise RetrievalIndexError(
                "Embedding rows and metadata entries must have the same length."
            )

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize one or more vectors to unit length for cosine similarity."""
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RetrievalError("Cannot compute cosine similarity for a zero vector.")
        return embedding / norms

    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity scores against all stored chunks.

        Args:
            query_embedding: 1D embedding vector.

        Returns:
            A 1D array of similarity scores aligned with the stored chunk list.
        """
        if query_embedding.ndim != 1:
            raise RetrievalError("Query embedding must be a 1D vector.")

        normalized_query = self._normalize_embedding(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
        normalized_corpus = self._normalize_embedding(np.asarray(self.embeddings, dtype=np.float32))
        scores = normalized_corpus @ normalized_query
        return np.asarray(scores, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the most relevant chunks for a user query.

        Args:
            query: User question or search phrase.
            top_k: Number of top-ranked chunks to return.

        Returns:
            A list of dictionaries with similarity score and chunk metadata.

        Raises:
            RetrievalError: If the query is empty or the top-k selection is invalid.
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Query text cannot be empty.")
        if top_k <= 0:
            raise RetrievalError("top_k must be greater than zero.")

        try:
            from embed.embedder import Embedder
        except ImportError as exc:
            raise RetrievalError(
                "The embed package is required to build a query embedding."
            ) from exc

        embedder = Embedder()
        query_embedding = embedder.encode_text(query)
        scores = self.cosine_similarity(query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []

        for idx in top_indices:
            chunk_meta = dict(self.metadata[int(idx)])
            results.append(
                {
                    "chunk_id": chunk_meta.get("chunk_id"),
                    "source": chunk_meta.get("source"),
                    "content": chunk_meta.get("content"),
                    "score": float(scores[int(idx)]),
                }
            )

        logger.info("Retrieved %s top chunks for query: %s", len(results), query)
        return results
