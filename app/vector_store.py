"""
Vector Store Module - Step 4 of RAG Pipeline

Manages storage and retrieval of embeddings using ChromaDB.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import chromadb

from app.text_chunker import Chunk
from app.constants import (
    CHROMA_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR
)

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Store chunks with their embeddings.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding embedding vectors
        """
        pass

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of matching chunks with metadata and scores
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        pass


class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector store implementation.

    Provides persistent storage and efficient similarity search
    for document chunks and their embeddings.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str = DEFAULT_PERSIST_DIR
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Nutrition RAG knowledge base"}
        )

        logger.info(f"ChromaDB initialized: collection='{collection_name}', path='{persist_directory}'")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Store chunks with their embeddings in ChromaDB.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding embedding vectors

        Raises:
            ValueError: If chunks and embeddings lengths don't match
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")

        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")

        ids = [str(chunk.chunk_id) for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        total_added = 0
        total_batches = (len(chunks) - 1) // CHROMA_BATCH_SIZE + 1

        for i in range(0, len(chunks), CHROMA_BATCH_SIZE):
            batch_end = min(i + CHROMA_BATCH_SIZE, len(chunks))
            batch_num = i // CHROMA_BATCH_SIZE + 1

            logger.debug(f"Adding batch {batch_num}/{total_batches} ({batch_end - i} items)")

            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

            total_added += (batch_end - i)

        logger.info(f"Successfully added {total_added} chunks. Total in collection: {self.collection.count()}")

    def query(self, query_embedding: List[float], top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"brand": "Pizza Hut"})

        Returns:
            List of results with content, metadata, and similarity scores
        """
        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }

        # Add metadata filter if provided
        if filter_metadata:
            query_params["where"] = filter_metadata

        # Query collection
        results = self.collection.query(**query_params)

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "chunk_id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })

        return formatted_results

    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Nutrition RAG knowledge base"}
        )
        logger.info(f"Collection '{self.collection_name}' cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "persist_directory": self.persist_directory
        }
