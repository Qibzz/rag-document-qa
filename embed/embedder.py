"""Embedding utilities for the lightweight RAG pipeline.

This module uses sentence-transformers with the local-friendly
all-MiniLM-L6-v2 model to turn chunk text into dense vectors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Base exception for embedding problems."""


class ModelLoadError(EmbeddingError):
    """Raised when the sentence-transformer model cannot be loaded."""


class Embedder:
    """Wrapper for generating embeddings with sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """Initialize the model used for text embedding.

        Args:
            model_name: Hugging Face model identifier.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelLoadError(
                "The 'sentence-transformers' package is required. "
                "Install it with 'pip install sentence-transformers'."
            ) from exc

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise ModelLoadError(f"Unable to load model '{model_name}'.") from exc

        self.model_name = model_name

    def encode_text(self, text: str) -> np.ndarray:
        """Convert a single string into a dense embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            A 1D NumPy array representing the embedding.

        Raises:
            EmbeddingError: If the input text is empty or encoding fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Embedding input text cannot be empty.")

        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
        except Exception as exc:
            raise EmbeddingError("Failed to encode input text.") from exc

        return np.asarray(embedding, dtype=np.float32)

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Convert many text strings into a matrix of embeddings.

        Args:
            texts: Iterable of text strings.

        Returns:
            A 2D NumPy array where each row is one embedding.

        Raises:
            EmbeddingError: If the text iterable is empty or encoding fails.
        """
        text_list = [text for text in texts if isinstance(text, str) and text.strip()]
        if not text_list:
            raise EmbeddingError("No non-empty text values were provided for embedding.")

        try:
            embeddings = self.model.encode(
                text_list,
                normalize_embeddings=True,
                batch_size=32,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to encode text batch.") from exc

        return np.asarray(embeddings, dtype=np.float32)

    def embed_chunks(
        self,
        chunks: Iterable[Dict[str, Any]],
        output_dir: str | Path,
        metadata_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Embed chunk records and save vectors + metadata to disk.

        Args:
            chunks: Iterable of chunk dictionaries containing at least `content`.
            output_dir: Directory where embeddings and metadata should be stored.
            metadata_path: Optional custom path for the JSON metadata output.

        Returns:
            A dictionary describing the stored outputs.
        """
        chunk_records = list(chunks)
        if not chunk_records:
            raise EmbeddingError("At least one chunk is required for embedding.")

        texts = [record.get("content", "") for record in chunk_records]
        embeddings = self.encode_texts(texts)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        vector_path = output_path / "embeddings.npy"
        np.save(vector_path, embeddings)

        metadata_records: List[Dict[str, Any]] = []
        for record, embedding in zip(chunk_records, embeddings):
            metadata_records.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "source": record.get("source"),
                    "content": record.get("content"),
                    "embedding_path": str(vector_path),
                    "embedding_dim": int(embedding.shape[0]),
                }
            )

        metadata_file = Path(metadata_path) if metadata_path else output_path / "chunk_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(metadata_records, handle, ensure_ascii=False, indent=2)

        logger.info("Saved %s embeddings to %s", len(metadata_records), vector_path)
        return {
            "vector_path": str(vector_path),
            "metadata_path": str(metadata_file),
            "embedding_count": len(metadata_records),
            "embedding_dim": int(embeddings.shape[1]),
        }
