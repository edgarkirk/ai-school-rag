"""
Embedding Module - Step 3 of RAG Pipeline

Converts text chunks into vector embeddings using Azure OpenAI.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from openai import AzureOpenAI

from app.config import ChatConfig
from app.text_chunker import Chunk
from app.constants import EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract base class for text embedding strategies."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Embed multiple chunks efficiently.

        Args:
            chunks: List of chunks to embed

        Returns:
            List of embedding vectors
        """
        pass


class AzureEmbedder(Embedder):
    """
    Azure OpenAI embedding implementation.

    Uses Azure's embedding models to convert text to vectors.
    Supports batch processing for efficiency.
    """

    def __init__(self, config: ChatConfig, model: str = None):
        """
        Initialize Azure embedder.

        Args:
            config: Azure OpenAI configuration
            model: Embedding model name (uses config.embedding_model if not provided)
        """
        self.config = config
        self.model = model or config.embedding_model
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key
        )

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 dimensions for ada-002)
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Embed multiple chunks in batches.

        Args:
            chunks: List of chunks to embed

        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks")
        texts = [chunk.content for chunk in chunks]
        all_embeddings = []
        total_batches = (len(texts) - 1) // EMBEDDING_BATCH_SIZE + 1

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_num = i // EMBEDDING_BATCH_SIZE + 1

            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        logger.info(f"Successfully embedded {len(all_embeddings)} chunks")
        return all_embeddings
