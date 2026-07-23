"""Answer generation module for the lightweight RAG pipeline.

The QA engine combines the top-k retrieved chunks into a grounded context and
sends that context to the Google Gemini API, asking the model to answer only
from the provided context.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gemini-flash-lite-latest"
DEFAULT_MAX_TOKENS = 512


class QAGenerationError(Exception):
    """Base exception for answer-generation failures."""


class MissingAPIKeyError(QAGenerationError):
    """Raised when the Gemini API key is unavailable."""


class PromptBuildError(QAGenerationError):
    """Raised when the prompt cannot be assembled safely."""


class QAEngine:
    """Generate answers grounded in retrieved chunks using Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialize the QA engine.

        Args:
            api_key: Gemini API key. When omitted, the value is read from the
                `GEMINI_API_KEY` environment variable.
            model_name: Gemini model identifier.
            max_tokens: Maximum tokens the model may generate.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily create the Gemini client when it is needed."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise QAGenerationError(
                    "The 'google-genai' package is required. Install it with 'pip install google-genai'."
                ) from exc

            if not self.api_key:
                raise MissingAPIKeyError(
                    "Gemini API key is missing. Set GEMINI_API_KEY in your environment or pass api_key explicitly."
                )

            self._client = genai.Client(api_key=self.api_key)

        return self._client

    def build_context(self, retrieved_chunks: Iterable[Dict[str, Any]]) -> str:
        """Build a single grounded context string from retrieved chunks.

        Args:
            retrieved_chunks: Iterable of chunk dictionaries returned by the retriever.

        Returns:
            Concatenated context text with chunk source labels.

        Raises:
            PromptBuildError: If no valid chunks are provided.
        """
        chunk_list = list(retrieved_chunks)
        if not chunk_list:
            raise PromptBuildError("No retrieved chunks were provided for answer generation.")

        context_parts: List[str] = []
        for i, chunk in enumerate(chunk_list, start=1):
            content = chunk.get("content", "")
            source = chunk.get("source", f"chunk-{i}")
            if not isinstance(content, str) or not content.strip():
                continue
            context_parts.append(f"[Source {i}: {source}]\n{content.strip()}")

        if not context_parts:
            raise PromptBuildError("Retrieved chunks do not contain usable content.")

        return "\n\n".join(context_parts)

    def build_prompt(self, question: str, context: str) -> str:
        """Construct the final user prompt for Gemini.

        Args:
            question: User's question.
            context: Context text derived from retrieved chunks.

        Returns:
            A grounded prompt string that instructs the model to answer only from context.

        Raises:
            PromptBuildError: If either argument is empty or invalid.
        """
        if not isinstance(question, str) or not question.strip():
            raise PromptBuildError("Question text cannot be empty.")
        if not isinstance(context, str) or not context.strip():
            raise PromptBuildError("Answer context cannot be empty.")

        return (
            "Jawab pertanyaan berikut HANYA berdasarkan context yang diberikan. "
            "Jika informasi tidak ada di context, katakan bahwa jawaban tidak tersedia di context.\n\n"
            f"Context:\n{context}\n\n"
            f"Pertanyaan:\n{question}"
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate an answer grounded in retrieved context using Gemini.

        Args:
            question: User's question.
            retrieved_chunks: Top-k chunks returned by the retriever.

        Returns:
            A dictionary containing the answer, source chunks used, and context.

        Raises:
            QAGenerationError: If the answer cannot be generated.
        """
        context = self.build_context(retrieved_chunks)
        prompt = self.build_prompt(question, context)

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=self.max_tokens,
                ),
            )
        except MissingAPIKeyError:
            raise
        except Exception as exc:
            raise QAGenerationError("Failed to generate answer from Gemini API.") from exc

        answer_text = ""
        if hasattr(response, "text"):
            answer_text = response.text
        elif hasattr(response, "candidates"):
            for candidate in response.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            answer_text += part.text

        if not answer_text.strip():
            raise QAGenerationError("Gemini returned an empty answer.")

        return {
            "answer": answer_text.strip(),
            "context": context,
            "sources": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "score": chunk.get("score"),
                }
                for chunk in retrieved_chunks
            ],
        }
