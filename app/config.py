"""
Configuration Module

Handles loading and managing application configuration.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """Configuration data class for chat application."""
    api_key: str
    endpoint: str
    api_version: str
    model: str
    embedding_model: str = "text-embedding-3-small-1"


class ConfigurationLoader:
    """Loads and validates configuration from environment variables."""

    def load(self) -> ChatConfig:
        """
        Load configuration from environment.

        Returns:
            ChatConfig instance

        Raises:
            ValueError: If required configuration is missing
        """
        load_dotenv()

        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("DIAL_API_KEY or MY_KEY not found in .env file")

        config = ChatConfig(
            api_key=api_key,
            endpoint=os.getenv("AZURE_ENDPOINT", "https://ai-proxy.lab.epam.com"),
            api_version=os.getenv("API_VERSION", "2023-08-01-preview"),
            model=os.getenv("MODEL", "gpt-4o"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small-1")
        )

        logger.info("Configuration loaded successfully")
        return config

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        return os.getenv("DIAL_API_KEY") or os.getenv("MY_KEY")
