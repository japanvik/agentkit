"""
Tests for the ConversationContext class in the conversation management module.

These tests verify that the ConversationContext class correctly manages messages,
participants, and tasks within a conversation.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from networkkit.messages import Message, MessageType

from agentkit.memory.conversation.context import ConversationContext
from agentkit.memory.conversation.task import Task


class TestConversationContext:
    """Tests for the ConversationContext class."""
    
    def test_conversation_initialization(self):
        """Test that a conversation context is initialized correctly."""
        # Create a conversation context
        conversation = ConversationContext(conversation_id="conv-123")
        
        # Verify conversation attributes
        assert conversation.conversation_id == "conv-123"
        assert conversation.participants == set()
        assert conversation.history == []
        assert conversation.tasks == []
        assert isinstance(conversation.last_activity, datetime)
        assert conversation.state == {}
        assert not conversation.is_broadcast
    
    def test_conversation_initialization_with_message(self):
        """Test that a conversation context is initialized correctly with an initial message."""
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Create a conversation context with an initial message
        conversation = ConversationContext(conversation_id="conv-123", initial_message=message)
        
        # Verify conversation attributes
        assert conversation.conversation_id == "conv-123"
        assert conversation.participants == {"Sender", "Receiver"}
        assert len(conversation.history) == 1
        assert conversation.history[0] == message
        assert conversation.tasks == []
        assert isinstance(conversation.last_activity, datetime)
        assert conversation.state == {}
        assert not conversation.is_broadcast
    
    def test_add_message(self):
        """Test adding a message to a conversation."""
        # Create a conversation context
        conversation = ConversationContext(conversation_id="conv-123")
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Add the message to the conversation
        conversation.add_message(message)
        
        # Verify conversation state
        assert len(conversation.history) == 1
        assert conversation.history[0] == message
        assert conversation.participants == {"Sender", "Receiver"}
        assert isinstance(conversation.last_activity, datetime)
    
    def test_add_broadcast_message(self):
        """Test adding a broadcast message to a conversation."""
        # Create a conversation context
        conversation = ConversationContext(conversation_id="broadcast:test")
        
        # Create a mock broadcast message
        message = Message(
            source="Sender",
            to="ALL",
            content="Test broadcast message",
            message_type=MessageType.CHAT
        )
        
        # Add the message to the conversation
        conversation.add_message(message)
        
        # Verify conversation state
        assert len(conversation.history) == 1
        assert conversation.history[0] == message
        assert conversation.participants == {"Sender"}  # Only sender is added for broadcast
        assert isinstance(conversation.last_activity, datetime)
        assert conversation.is_broadcast
    
    def test_add_task(self):
        """Test adding a task to a conversation."""
        # Create a conversation context
        conversation = ConversationContext(conversation_id="conv-123")
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Create a task
        task = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task"
        )
        
        # Add the task to the conversation
        conversation.add_task(task)
        
        # Verify conversation state
        assert len(conversation.tasks) == 1
        assert conversation.tasks[0] == task
