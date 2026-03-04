"""
Document Loading Module - Step 1 of RAG Pipeline

This module handles loading documents from various sources.
Currently supports JSONL format for nutrition data.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a document with content and metadata.

    This will later be split into chunks for embedding in the vector database.
    """
    content: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        """String representation showing content preview."""
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    Following Open/Closed Principle - extend for new file formats
    without modifying existing code.
    """

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load documents from source.

        Returns:
            List of Document objects
        """
        pass


class JSONLDocumentLoader(DocumentLoader):
    """
    Loads documents from JSONL (JSON Lines) files.

    Expected format per line:
    {
        "text": "document content here...",
        "metadata": {"brand": "Pizza Hut", "product": "Cheese Pizza", ...}
    }
    """

    def __init__(self, file_path: str):
        """
        Initialize JSONL document loader.

        Args:
            file_path: Path to the JSONL file
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        Load documents from JSONL file.

        Returns:
            List of Document objects

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        logger.info(f"Loading documents from {self.file_path}")
        documents = []

        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                    doc = Document(
                        content=data.get('text', ''),
                        metadata=data.get('metadata', {})
                    )
                    documents.append(doc)

                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(documents)} documents successfully")
        return documents
