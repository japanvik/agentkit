"""
Tests for the BaseMemory class.

This module contains tests for the BaseMemory class functionality, including
message storage, history retrieval, and context formatting.
"""
# Standard library imports
import unittest
from typing import Dict, Any, List, Optional

# Third-party imports
import pytest
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.memory.base_memory import BaseMemory
from agentkit.common.interfaces import ComponentConfig


class TestMemory(BaseMemory):
    """Concrete implementation of BaseMemory for testing."""
    
    def __init__(self, max_history_length: int = 10):
        """Initialize the memory with a maximum history length."""
        super().__init__(max_history_length)
        self.messages = []
    
    def remember(self, message: Message) -> None:
        """Store a message in the conversation history."""
        self.messages.append(message)
        if self.max_history_length > 0 and len(self.messages) > self.max_history_length:
            self.messages = self.messages[-self.max_history_length:]
    
    def get_history(self) -> List[Message]:
        """Retrieve the complete conversation history from memory."""
        return self.messages
    
    def get_chat_context(self, target: str, prefix: str = "", 
                        user_role_name: str = "", 
                        assistant_role_name: str = "") -> str:
        """Retrieve chat conversation history with a specific target and format it."""
        chat_log = self.chat_log_for(target)
        
        if not user_role_name:
            user_role_name = "User"
        if not assistant_role_name:
            assistant_role_name = "Assistant"
        
        formatted_messages = []
        for message in chat_log:
            role = assistant_role_name if message.source == target else user_role_name
            formatted_messages.append(f"{prefix}{role}: {message.content}")
        
        return "\n".join(formatted_messages)
    
    def chat_log_for(self, target: str) -> List[Message]:
        """Retrieve the chat log for the specified target."""
        return [m for m in self.messages if m.source == target or m.to == target]


class TestBaseMemory:
    """Test suite for BaseMemory class."""
    
    @pytest.fixture
    def memory(self):
        """Create a BaseMemory instance for testing."""
        return TestMemory(max_history_length=5)
    
    def test_initialization(self, memory):
        """Test that the memory initializes correctly."""
        assert memory.max_history_length == 5
        assert len(memory.messages) == 0
    
    def test_remember(self, memory):
        """Test that remember stores messages correctly."""
        message = Message(
            source="user",
            to="agent",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        memory.remember(message)
        
        assert len(memory.messages) == 1
        assert memory.messages[0] == message
    
    def test_max_history_length(self, memory):
        """Test that max_history_length limits the number of stored messages."""
        # Add 6 messages (max is 5)
        for i in range(6):
            message = Message(
                source="user",
                to="agent",
                content=f"Test message {i}",
                message_type=MessageType.CHAT
            )
            memory.remember(message)
        
        # Check that only the last 5 messages are stored
        assert len(memory.messages) == 5
        assert memory.messages[0].content == "Test message 1"
        assert memory.messages[4].content == "Test message 5"
    
    def test_unlimited_history(self):
        """Test that setting max_history_length to 0 allows unlimited messages."""
        memory = TestMemory(max_history_length=0)
        
        # Add 20 messages
        for i in range(20):
            message = Message(
                source="user",
                to="agent",
                content=f"Test message {i}",
                message_type=MessageType.CHAT
            )
            memory.remember(message)
        
        # Check that all messages are stored
        assert len(memory.messages) == 20
    
    def test_get_history(self, memory):
        """Test that get_history returns all stored messages."""
        # Add some messages
        messages = []
        for i in range(3):
            message = Message(
                source="user",
                to="agent",
                content=f"Test message {i}",
                message_type=MessageType.CHAT
            )
            memory.remember(message)
            messages.append(message)
        
        history = memory.get_history()
        
        assert len(history) == 3
        assert history == messages
    
    def test_chat_log_for(self, memory):
        """Test that chat_log_for filters messages correctly."""
        # Add messages with different sources and targets
        message1 = Message(
            source="user1",
            to="agent",
            content="Message from user1",
            message_type=MessageType.CHAT
        )
        message2 = Message(
            source="agent",
            to="user1",
            content="Response to user1",
            message_type=MessageType.CHAT
        )
        message3 = Message(
            source="user2",
            to="agent",
            content="Message from user2",
            message_type=MessageType.CHAT
        )
        message4 = Message(
            source="agent",
            to="user2",
            content="Response to user2",
            message_type=MessageType.CHAT
        )
        
        memory.remember(message1)
        memory.remember(message2)
        memory.remember(message3)
        memory.remember(message4)
        
        # Get chat log for user1
        user1_log = memory.chat_log_for("user1")
        
        assert len(user1_log) == 2
        assert user1_log[0] == message1
        assert user1_log[1] == message2
    
    def test_get_chat_context(self, memory):
        """Test that get_chat_context formats messages correctly."""
        # Add messages with different sources and targets
        message1 = Message(
            source="user1",
            to="agent",
            content="Message from user1",
            message_type=MessageType.CHAT
        )
        message2 = Message(
            source="agent",
            to="user1",
            content="Response to user1",
            message_type=MessageType.CHAT
        )
        
        memory.remember(message1)
        memory.remember(message2)
        
        # Get formatted chat context
        context = memory.get_chat_context("agent", prefix="> ", 
                                         user_role_name="Human", 
                                         assistant_role_name="AI")
        
        assert "> Human: Message from user1" in context
        assert "> AI: Response to user1" in context
