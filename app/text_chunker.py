"""
Text Chunking Module - Step 2 of RAG Pipeline

Splits documents into smaller chunks for better embedding and retrieval.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.document_loader import Document
from app.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    DEFAULT_SEPARATORS
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a chunk of text with metadata.

    Chunks are smaller pieces of documents that will be embedded
    and stored in the vector database.
    """
    content: str
    metadata: dict
    chunk_id: int  # Unique identifier for this chunk

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"Chunk(id={self.chunk_id}, content='{preview}')"


class TextChunker(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into chunks."""
        raise NotImplementedError


def make_doc_id(metadata: dict, doc_idx: int = None, content: str = "") -> str:
    """
    Stable identifier for a document, based on metadata fields.

    This helps dedupe and trace chunks back to their parent doc even if input order changes.
    Falls back to doc_idx or content hash if metadata is missing to prevent collisions.

    Args:
        metadata: Document metadata dictionary
        doc_idx: Document index as fallback
        content: Document content for content-based hashing as last resort

    Returns:
        SHA-1 hash as document identifier
    """
    key = "|".join([
        str(metadata.get("brand", "")).strip().lower(),
        str(metadata.get("product", "")).strip().lower(),
        str(metadata.get("size", "")).strip().lower(),
    ])

    # If key is empty or just separators, use fallback to prevent collisions
    if not key.strip("|"):
        if doc_idx is not None:
            key = f"doc_{doc_idx}"
        else:
            # Last resort: hash the content
            key = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]

    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class RecursiveTextChunker(TextChunker):
    """
    Chunks text using Langchain's RecursiveCharacterTextSplitter.

    This splitter tries to keep paragraphs, then sentences, then words together.
    It's smart about splitting on natural boundaries.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize recursive text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a valid chunk
            separators: List of separators to try
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        if separators is None:
            separators = DEFAULT_SEPARATORS

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def _is_valid_chunk(self, chunk_content: str) -> bool:
        """
        Check if chunk is valid (not too small or empty).

        Args:
            chunk_content: The chunk text to validate

        Returns:
            True if chunk meets minimum size requirements
        """
        cleaned = chunk_content.strip()
        return len(cleaned) >= self.min_chunk_size

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with preserved metadata
        """
        logger.info(f"Chunking {len(documents)} documents (size: {self.chunk_size}, overlap: {self.chunk_overlap})")

        chunks: List[Chunk] = []
        chunk_id = 0

        for doc_idx, document in enumerate(documents):
            base_meta = document.metadata if isinstance(document.metadata, dict) else {}
            doc_id = make_doc_id(base_meta)
            content = document.content or ""
            text_chunks = self.text_splitter.split_text(content)

            for chunk_idx, text_chunk in enumerate(text_chunks):
                chunk_metadata = base_meta.copy()
                chunk_metadata.update({
                    "doc_id": doc_id,
                    "source_doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                })

                chunks.append(Chunk(
                    content=text_chunk,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                ))
                chunk_id += 1

        avg_chunks = len(chunks) / len(documents) if documents else 0
        logger.info(f"Created {len(chunks)} chunks (avg: {avg_chunks:.2f} per document)")

        return chunks
