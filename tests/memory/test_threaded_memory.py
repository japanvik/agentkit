"""
Tests for the ThreadedMemory class.

These tests verify that the ThreadedMemory correctly manages conversation-specific memory.
"""

import pytest
from unittest.mock import MagicMock, patch

from networkkit.messages import Message, MessageType

from agentkit.memory.threaded_memory import ThreadedMemory
from agentkit.memory.conversation.context import ConversationContext
from agentkit.memory.conversation.manager import ConversationManager


class TestThreadedMemory:
    """Tests for the ThreadedMemory class."""
    
    def test_memory_initialization(self):
        """Test that a threaded memory is initialized correctly."""
        # Create a threaded memory
        memory = ThreadedMemory(max_history_length=50)
        
        # Verify memory attributes
        assert memory.max_history_length == 50
        assert isinstance(memory.conversation_manager, ConversationManager)
    
    def test_remember(self):
        """Test remembering a message."""
        # Create a threaded memory
        memory = ThreadedMemory()
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Remember the message
        memory.remember(message)
        
        # Verify the message was added to a conversation
        conversation_id = memory.get_conversation_id(message)
        conversation = memory.get_conversation_by_id(conversation_id)
        
        assert conversation is not None
        assert len(conversation.history) == 1
        assert conversation.history[0] == message
    
    def test_get_history(self):
        """Test getting history from all conversations."""
        # Create a threaded memory
        memory = ThreadedMemory()
        
        # Create mock messages
        message1 = Message(
            source="Sender",
            to="Receiver1",
            content="Test message 1",
            message_type=MessageType.CHAT
        )
        
        message2 = Message(
            source="Sender",
            to="Receiver2",
            content="Test message 2",
            message_type=MessageType.CHAT
        )
        
        # Remember the messages
        memory.remember(message1)
        memory.remember(message2)
        
        # Get history
        history = memory.get_history()
        
        # Verify history
        assert len(history) == 2
        assert message1 in history
        assert message2 in history
    
    def test_get_chat_context(self):
        """Test getting formatted chat context for a specific target."""
        # Create a threaded memory
        memory = ThreadedMemory()
        
        # Create mock messages
        message1 = Message(
            source="Sender",
            to="Receiver",
            content="Hello Receiver",
            message_type=MessageType.CHAT
        )
        
        message2 = Message(
            source="Receiver",
            to="Sender",
            content="Hello Sender",
            message_type=MessageType.CHAT
        )
        
        message3 = Message(
            source="Sender",
            to="Receiver",
            content="How are you?",
            message_type=MessageType.CHAT
        )
        
        # Remember the messages
        memory.remember(message1)
        memory.remember(message2)
        memory.remember(message3)
        
        # Get chat context for Receiver
        context = memory.get_chat_context(
            target="Receiver",
            prefix="> ",
            user_role_name="User",
            assistant_role_name="Assistant"
        )
        
        # Verify context
        assert "> User: Hello Receiver" in context
        assert "> Assistant: Hello Sender" in context
        assert "> User: How are you?" in context
    
    def test_get_conversation_by_id(self):
        """Test getting a conversation by ID."""
        # Create a threaded memory
        memory = ThreadedMemory()
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Remember the message
        memory.remember(message)
        
        # Get the conversation ID
        conversation_id = memory.get_conversation_id(message)
        
        # Get the conversation by ID
        conversation = memory.get_conversation_by_id(conversation_id)
        
        # Verify conversation
        assert conversation is not None
        assert conversation.conversation_id == conversation_id
        assert len(conversation.history) == 1
        assert conversation.history[0] == message
        
        # Try to get a non-existent conversation
        non_existent = memory.get_conversation_by_id("non-existent")
        
        # Verify it's None
        assert non_existent is None
    
    def test_get_conversations_for_participant(self):
        """Test getting conversations for a participant."""
        # Create a threaded memory
        memory = ThreadedMemory()
        
        # Create mock messages
        message1 = Message(
            source="Sender",
            to="Receiver1",
            content="Test message 1",
            message_type=MessageType.CHAT
        )
        
        message2 = Message(
            source="Sender",
            to="Receiver2",
            content="Test message 2",
            message_type=MessageType.CHAT
        )
        
        # Remember the messages
        memory.remember(message1)
        memory.remember(message2)
        
        # Get conversations for Sender
        conversations = memory.get_conversations_for_participant("Sender")
        
        # Verify conversations
        assert len(conversations) == 2
        assert any(conv.conversation_id.startswith("Receiver1:Sender") for conv in conversations)
        assert any(conv.conversation_id.startswith("Receiver2:Sender") for conv in conversations)
    
    def test_get_active_conversations(self):
        """Test getting active conversations."""
        # Create a threaded memory
        memory = ThreadedMemory()
        
        # Create mock messages
        message1 = Message(
            source="Sender",
            to="Receiver1",
            content="Test message 1",
            message_type=MessageType.CHAT
        )
        
        message2 = Message(
            source="Sender",
            to="Receiver2",
            content="Test message 2",
            message_type=MessageType.CHAT
        )
        
        # Remember the messages
        memory.remember(message1)
        memory.remember(message2)
        
        # Get active conversations
        active_conversations = memory.get_active_conversations()
        
        # Verify active conversations
        assert len(active_conversations) == 2
    
    def test_clear(self):
        """Test clearing all conversations and tasks."""
        # Create a threaded memory
        memory = ThreadedMemory()
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Remember the message
        memory.remember(message)
        
        # Verify the message was added
        assert len(memory.get_history()) == 1
        
        # Clear the memory
        memory.clear()
        
        # Verify the memory was cleared
        assert len(memory.get_history()) == 0
        assert len(memory.conversation_manager.conversations) == 0
