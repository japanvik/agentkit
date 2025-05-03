"""
Tests for the SimpleBrain class.

This module contains tests for the SimpleBrain class functionality, including
message handling, LLM interaction, and response generation.
"""
# Standard library imports
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

# Third-party imports
import pytest
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.memory_protocol import Memory
from agentkit.common.interfaces import MessageSender, ComponentConfig


class MockMemory:
    """Mock implementation of Memory for testing."""
    
    def __init__(self):
        self.messages = []
    
    def remember(self, message: Message) -> None:
        """Store a message in memory."""
        self.messages.append(message)
    
    def get_history(self) -> List[Message]:
        """Get all stored messages."""
        return self.messages
    
    def get_context(self) -> str:
        """Get formatted context."""
        return "\n".join([f"{m.source}: {m.content}" for m in self.messages])
    
    def get_chat_context(self, target: str, prefix: str = "", 
                        user_role_name: str = "", 
                        assistant_role_name: str = "") -> str:
        """Get formatted chat context."""
        return "\n".join([f"{m.source}: {m.content}" for m in self.messages])
    
    def chat_log_for(self, target: str) -> List[Message]:
        """Get chat log for a specific target."""
        # For testing purposes, return all messages regardless of target
        return self.messages


class MockMessageSender:
    """Mock implementation of MessageSender for testing."""
    
    def __init__(self):
        self._attention = "test_target"
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


class TestSimpleBrain:
    """Test suite for SimpleBrain class."""
    
    @pytest.fixture
    def brain(self):
        """Create a SimpleBrain instance for testing."""
        memory = MockMemory()
        brain = SimpleBrain(
            name="test_brain",
            description="Test brain for testing",
            model="test_model",
            memory_manager=memory,
            system_prompt="You are {name}, {description}. Context: {context}. Target: {target}",
            user_prompt="This is a test",
            api_config={"temperature": 0.7}
        )
        
        message_sender = MockMessageSender()
        config = ComponentConfig(
            agent_name="test_agent",
            config={"test_key": "test_value"},
            message_sender=message_sender
        )
        brain.set_config(config)
        
        return brain
    
    def test_initialization(self, brain):
        """Test that the brain initializes correctly."""
        assert brain.name == "test_brain"
        assert brain.description == "Test brain for testing"
        assert brain.model == "test_model"
        assert brain.system_prompt == "You are {name}, {description}. Context: {context}. Target: {target}"
        assert brain.user_prompt == "This is a test"
        assert brain.api_config == {"temperature": 0.7}
        assert brain.component_config is not None
    
    @pytest.mark.asyncio
    @patch('agentkit.brains.simple_brain.llm_chat')
    async def test_handle_chat_message(self, mock_llm_chat, brain):
        """Test that handle_chat_message processes messages correctly."""
        # Configure the mock to return a specific response
        mock_llm_chat.return_value = "Test LLM response"
        
        # Create a test message
        message = Message(
            source="user",
            to="test_brain",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Handle the message
        await brain.handle_chat_message(message)
        
        # Check that the message was stored in memory
        assert len(brain.memory_manager.messages) == 1
        assert brain.memory_manager.messages[0] == message
        
        # Check that attention was set to the message source
        assert brain.component_config.message_sender.attention == "user"
        
        # Check that llm_chat was called with the correct parameters
        mock_llm_chat.assert_called_once()
        args, kwargs = mock_llm_chat.call_args
        assert kwargs["llm_model"] == "test_model"
        
        # Check that a response was sent
        message_sender = brain.component_config.message_sender
        assert len(message_sender.sent_messages) == 1
        sent_message = message_sender.sent_messages[0]
        assert sent_message.source == "test_brain"
        assert sent_message.to == "user"
        assert sent_message.content == "Test LLM response"
    
    @pytest.mark.asyncio
    @patch('agentkit.brains.simple_brain.llm_chat')
    async def test_generate_chat_response(self, mock_llm_chat, brain):
        """Test that generate_chat_response creates a response correctly."""
        # Configure the mock to return a specific response
        mock_llm_chat.return_value = "Test LLM response"
        
        # Add a message to memory
        message = Message(
            source="user",
            to="test_brain",
            content="Test message",
            message_type=MessageType.CHAT
        )
        brain.memory_manager.remember(message)
        
        # Generate a response
        response = await brain.generate_chat_response()
        
        # Check that llm_chat was called with the correct parameters
        mock_llm_chat.assert_called_once()
        args, kwargs = mock_llm_chat.call_args
        assert kwargs["llm_model"] == "test_model"
        
        # Check that the system prompt was formatted correctly
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "You are test_brain, Test brain for testing" in messages[0]["content"]
        
        # Check that the response was formatted correctly
        assert response.source == "test_brain"
        assert response.to == "test_target"
        assert response.content == "Test LLM response"
        assert response.message_type == MessageType.CHAT
    
    @pytest.mark.asyncio
    @patch('agentkit.brains.simple_brain.llm_chat')
    async def test_handle_chat_message_from_self(self, mock_llm_chat, brain):
        """Test that handle_chat_message ignores messages from self."""
        # Create a test message from the brain itself
        message = Message(
            source="test_brain",
            to="user",
            content="Test message from self",
            message_type=MessageType.CHAT
        )
        
        # Handle the message
        await brain.handle_chat_message(message)
        
        # Check that the message was stored in memory
        assert len(brain.memory_manager.messages) == 1
        assert brain.memory_manager.messages[0] == message
        
        # Check that no response was generated (llm_chat not called)
        mock_llm_chat.assert_not_called()
