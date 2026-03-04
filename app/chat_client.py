"""
Chat Client Module

Abstractions and implementations for chat clients.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict

from openai import AzureOpenAI

from app.config import ChatConfig
from app.constants import DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)


class ChatClient(ABC):
    """Abstract base class for chat clients."""

    @abstractmethod
    def send_message(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages and get response.

        Args:
            messages: List of conversation messages

        Returns:
            Assistant's response
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection to the service.

        Returns:
            True if successful, False otherwise
        """
        pass


class AzureChatClient(ChatClient):
    """Azure OpenAI chat client implementation."""

    def __init__(self, config: ChatConfig):
        """
        Initialize Azure chat client.

        Args:
            config: Chat configuration
        """
        self.config = config
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key
        )

    def send_message(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages to Azure OpenAI.

        Args:
            messages: Conversation messages

        Returns:
            Assistant's response

        Raises:
            Exception: If API call fails
        """
        logger.debug(f"Sending message to {self.config.model}")
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=DEFAULT_TEMPERATURE,
            messages=messages
        )
        return response.choices[0].message.content

    def test_connection(self) -> bool:
        """
        Test connection to Azure OpenAI.

        Returns:
            True if connection successful
        """
        try:
            test_messages = [{
                "role": "user",
                "content": "Hello, please confirm you are ready."
            }]
            response = self.send_message(test_messages)
            return bool(response)
        except Exception:
            return False
