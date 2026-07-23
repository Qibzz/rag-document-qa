"""Utilities for loading and chunking documents for a lightweight RAG workflow.

This module is intentionally simple and dependency-light so it can be used in a
local demo without a heavy vector database.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


class DocumentLoaderError(Exception):
    """Base exception for document-loading problems."""


class UnsupportedFileTypeError(DocumentLoaderError):
    """Raised when the requested file type is not supported."""


class EmptyDocumentError(DocumentLoaderError):
    """Raised when the source document does not contain usable text."""


class InvalidChunkConfigError(DocumentLoaderError):
    """Raised when the chunking configuration is invalid."""


def _normalize_text(text: str) -> str:
    """Normalize whitespace so the chunking logic remains consistent.

    Args:
        text: Raw extracted document text.

    Returns:
        Cleaned text with repeated whitespace collapsed into single spaces.

    Raises:
        EmptyDocumentError: If the text is empty after normalization.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        raise EmptyDocumentError("Document content is empty after normalization.")
    return cleaned


def load_text_file(file_path: str | Path) -> str:
    """Read a UTF-8 text document from disk.

    Args:
        file_path: Path to the `.txt` file.

    Returns:
        The file content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        DocumentLoaderError: If the file cannot be opened or read.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Falling back to UTF-8 with BOM support for %s", path)
        try:
            return path.read_text(encoding="utf-8-sig")
        except Exception as exc:  # pragma: no cover - defensive branch
            raise DocumentLoaderError(f"Unable to decode text file: {path}") from exc
    except OSError as exc:
        raise DocumentLoaderError(f"Unable to read text file: {path}") from exc


def load_pdf_file(file_path: str | Path) -> str:
    """Extract text from a PDF document.

    Args:
        file_path: Path to the `.pdf` file.

    Returns:
        Concatenated text from all PDF pages.

    Raises:
        DocumentLoaderError: If the PDF library is unavailable or no text can be extracted.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoaderError(
            "PDF support requires the 'pypdf' package. Install it with 'pip install pypdf'."
        ) from exc

    try:
        reader = PdfReader(str(path))
        pages: List[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())

        combined_text = "\n\n".join(pages)
        if not combined_text.strip():
            raise EmptyDocumentError("No extractable text was found in the PDF document.")

        return combined_text
    except EmptyDocumentError:
        raise
    except Exception as exc:
        raise DocumentLoaderError(f"Unable to read PDF file: {path}") from exc


def load_document(file_path: str | Path) -> str:
    """Load a supported document into plain text.

    Args:
        file_path: Path to the source document.

    Returns:
        A single normalized text string.

    Raises:
        UnsupportedFileTypeError: If the file extension is not supported.
        EmptyDocumentError: If the document contains no usable text.
        DocumentLoaderError: For other load failures.
    """
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported document type: {path.suffix or 'unknown extension'}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if path.suffix.lower() == ".txt":
        raw_text = load_text_file(path)
    else:
        raw_text = load_pdf_file(path)

    return _normalize_text(raw_text)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Normalized document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks ready for embedding.

    Raises:
        InvalidChunkConfigError: If the configuration is invalid.
        EmptyDocumentError: If the text is empty.
    """
    if chunk_size <= 0:
        raise InvalidChunkConfigError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise InvalidChunkConfigError("overlap must be zero or greater.")
    if overlap >= chunk_size:
        raise InvalidChunkConfigError("overlap must be smaller than chunk_size.")

    normalized_text = _normalize_text(text)
    step = chunk_size - overlap
    chunks: List[str] = []

    for start in range(0, len(normalized_text), step):
        chunk = normalized_text[start : start + chunk_size]
        if not chunk.strip():
            continue
        chunks.append(chunk)

        if len(chunk) < chunk_size:
            break

    if not chunks:
        raise EmptyDocumentError("No chunks could be created from the provided text.")

    return chunks


def build_chunks(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """Load a document and convert it into chunk metadata records.

    Each record contains the original source path, a sequential chunk ID, and the
    chunk content that can later be embedded and stored.

    Args:
        file_path: Path to the source document.
        chunk_size: Maximum number of characters in each chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A list of dictionaries with chunk metadata.
    """
    path = Path(file_path)
    clean_text = load_document(path)
    chunk_list = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)

    records: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunk_list, start=1):
        records.append(
            {
                "chunk_id": f"{path.stem}-chunk-{index:03d}",
                "source": str(path),
                "content": chunk,
            }
        )

    logger.info("Created %s chunks from %s", len(records), path)
    return records


def save_chunks_to_json(chunks: Iterable[Dict[str, Any]], output_path: str | Path) -> str:
    """Persist chunk records to a JSON file.

    Args:
        chunks: Iterable of chunk dictionaries.
        output_path: Destination path for the JSON file.

    Returns:
        The normalized output path as a string.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk_records = list(chunks)

    try:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(chunk_records, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise DocumentLoaderError(f"Unable to write chunks to JSON: {destination}") from exc

    logger.info("Saved %s chunk records to %s", len(chunk_records), destination)
    return str(destination)
