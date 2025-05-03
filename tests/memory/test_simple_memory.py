"""
Tests for the SimpleMemory class.

This module contains tests for the SimpleMemory class functionality, including
message storage, history retrieval, and context formatting.
"""
# Standard library imports
from typing import Dict, Any, List, Optional

# Third-party imports
import pytest
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.memory.simple_memory import SimpleMemory
from agentkit.common.interfaces import ComponentConfig


class TestSimpleMemory:
    """Test suite for SimpleMemory class."""
    
    @pytest.fixture
    def memory(self):
        """Create a SimpleMemory instance for testing."""
        return SimpleMemory(max_history_length=5)
    
    def test_initialization(self, memory):
        """Test that the memory initializes correctly."""
        assert memory.max_history_length == 5
        assert len(memory.history) == 0
    
    def test_remember(self, memory):
        """Test that remember stores messages correctly."""
        message = Message(
            source="user",
            to="agent",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        memory.remember(message)
        
        assert len(memory.history) == 1
        assert memory.history[0] == message
    
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
        assert len(memory.history) == 5
        assert memory.history[0].content == "Test message 1"
        assert memory.history[4].content == "Test message 5"
    
    def test_unlimited_history(self):
        """Test that setting max_history_length to 0 allows unlimited messages."""
        memory = SimpleMemory(max_history_length=0)
        
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
        assert len(memory.history) == 20
    
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
        # Add a non-chat message that should be filtered out
        message5 = Message(
            source="user1",
            to="agent",
            content="System message",
            message_type=MessageType.HELO
        )
        
        memory.remember(message1)
        memory.remember(message2)
        memory.remember(message3)
        memory.remember(message4)
        memory.remember(message5)
        
        # Get chat log for user1
        user1_log = memory.chat_log_for("user1")
        
        assert len(user1_log) == 2
        assert user1_log[0] == message1
        assert user1_log[1] == message2
        
        # Verify that non-chat messages are filtered out
        assert message5 not in user1_log
    
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
        
        # Get formatted chat context with default parameters
        context = memory.get_chat_context("user1")
        
        assert "user1: Message from user1" in context
        assert "agent: Response to user1" in context
        
        # Get formatted chat context with custom parameters
        context = memory.get_chat_context("user1", prefix="> ", 
                                         user_role_name="Human", 
                                         assistant_role_name="AI")
        
        assert "> Human: Message from user1" in context
        assert "> AI: Response to user1" in context
    
    def test_get_context(self, memory):
        """Test that get_context returns an empty string."""
        # The SimpleMemory implementation of get_context is intentionally empty
        context = memory.get_context()
        assert context == ""
