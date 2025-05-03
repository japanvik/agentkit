"""
Common test fixtures and configuration.

This module provides fixtures and configuration that are shared across
multiple test modules in the AgentKit test suite.
"""
# Standard library imports
import asyncio
from typing import Dict, Any, List, Optional

# Third-party imports
import pytest
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.common.interfaces import MessageSender, ComponentConfig


# We're removing the custom event_loop fixture to avoid the warning
# pytest-asyncio will provide its own event_loop fixture


class MockMessageSender:
    """Mock implementation of MessageSender for testing."""
    
    def __init__(self):
        self._attention = None
        self.sent_messages = []
    
    @property
    def attention(self) -> Optional[str]:
        """Get current attention target."""
        return self._attention
    
    @attention.setter
    def attention(self, value: str) -> None:
        """Set current attention target."""
        self._attention = value
    
    async def send_message(self, message: Message) -> None:
        """Send a message."""
        self.sent_messages.append(message)


@pytest.fixture
def mock_message_sender():
    """Create a MockMessageSender instance for testing."""
    return MockMessageSender()


@pytest.fixture
def component_config(mock_message_sender):
    """Create a ComponentConfig instance for testing."""
    return ComponentConfig(
        agent_name="test_agent",
        config={"test_key": "test_value"},
        message_sender=mock_message_sender
    )


@pytest.fixture
def chat_message():
    """Create a sample chat message for testing."""
    return Message(
        source="user",
        to="agent",
        content="Test message",
        message_type=MessageType.CHAT
    )
