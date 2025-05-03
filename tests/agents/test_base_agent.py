"""
Tests for the BaseAgent class.

This module contains tests for the BaseAgent class functionality, including
message handling, attention management, and component configuration.
"""
# Standard library imports
import asyncio
import unittest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

# Third-party imports
import pytest
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.agents.base_agent import BaseAgent
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.memory_protocol import Memory
from agentkit.common.interfaces import MessageSender, ComponentConfig


class MockMemory:
    """Mock implementation of Memory for testing."""
    
    def __init__(self):
        self.messages = []
        self.config = None
    
    def remember(self, message: Message) -> None:
        """Store a message in memory."""
        self.messages.append(message)
    
    def get_history(self) -> List[Message]:
        """Get all stored messages."""
        return self.messages
    
    def get_chat_context(self, target: str, prefix: str = "", 
                        user_role_name: str = "", 
                        assistant_role_name: str = "") -> str:
        """Get formatted chat context."""
        return "\n".join([f"{m.source}: {m.content}" for m in self.messages])
    
    def chat_log_for(self, target: str) -> List[Message]:
        """Get chat log for a specific target."""
        return [m for m in self.messages if m.source == target or m.to == target]
    
    def set_config(self, config: ComponentConfig) -> None:
        """Set component configuration."""
        self.config = config


class MockBrain:
    """Mock implementation of SimpleBrain for testing."""
    
    def __init__(self):
        self.messages = []
        self.config = None
    
    async def handle_chat_message(self, message: Message) -> None:
        """Handle a chat message."""
        self.messages.append(message)
    
    async def generate_chat_response(self) -> Message:
        """Generate a chat response."""
        return Message(
            source="mock_brain",
            to="test",
            content="Mock response",
            message_type=MessageType.CHAT
        )
    
    def set_config(self, config: ComponentConfig) -> None:
        """Set component configuration."""
        self.config = config


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


class TestBaseAgent:
    """Test suite for BaseAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create a BaseAgent instance for testing."""
        config = {"test_key": "test_value"}
        brain = MockBrain()
        memory = MockMemory()
        message_sender = MockMessageSender()
        
        agent = BaseAgent(
            name="test_agent",
            config=config,
            brain=brain,
            memory=memory,
            message_sender=message_sender
        )
        
        return agent
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test that the agent initializes correctly."""
        assert agent.name == "test_agent"
        assert agent.config == {"test_key": "test_value"}
        assert agent.brain is not None
        assert agent.memory is not None
        assert agent._message_sender is not None
        assert agent._running is False
        assert agent._attention is None
    
    @pytest.mark.asyncio
    async def test_component_config(self, agent):
        """Test that component config is created and passed to components."""
        assert agent.component_config is not None
        assert agent.component_config.agent_name == "test_agent"
        assert agent.component_config.config == {"test_key": "test_value"}
        assert agent.component_config.message_sender == agent
        
        # Check that config was passed to components
        assert agent.brain.config == agent.component_config
        assert agent.memory.config == agent.component_config
    
    @pytest.mark.asyncio
    async def test_brain_setter(self, agent):
        """Test that setting a new brain configures it correctly."""
        new_brain = MockBrain()
        agent.brain = new_brain
        
        assert agent._brain == new_brain
        assert new_brain.config == agent.component_config
    
    @pytest.mark.asyncio
    async def test_memory_setter(self, agent):
        """Test that setting a new memory configures it correctly."""
        new_memory = MockMemory()
        agent.memory = new_memory
        
        assert agent._memory == new_memory
        assert new_memory.config == agent.component_config
    
    @pytest.mark.asyncio
    async def test_attention_property(self, agent):
        """Test that attention property works correctly."""
        assert agent.attention is None
        
        agent.attention = "test_target"
        assert agent.attention == "test_target"
    
    @pytest.mark.asyncio
    async def test_send_message(self, agent):
        """Test that send_message delegates to message_sender."""
        message = Message(
            source="test_agent",
            to="test_target",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        await agent.send_message(message)
        
        # Check that message was delegated to message_sender
        assert len(agent._message_sender.sent_messages) == 1
        assert agent._message_sender.sent_messages[0] == message
    
    @pytest.mark.asyncio
    async def test_is_intended_for_me(self, agent):
        """Test that is_intended_for_me correctly identifies messages for the agent."""
        # Message addressed to agent
        message1 = Message(
            source="other",
            to="test_agent",
            content="Test message 1",
            message_type=MessageType.CHAT
        )
        
        # Broadcast message
        message2 = Message(
            source="other",
            to="ALL",
            content="Test message 2",
            message_type=MessageType.CHAT
        )
        
        # Message from agent
        message3 = Message(
            source="test_agent",
            to="other",
            content="Test message 3",
            message_type=MessageType.CHAT
        )
        
        # Message from attention target
        agent.attention = "attention_target"
        message4 = Message(
            source="attention_target",
            to="other",
            content="Test message 4",
            message_type=MessageType.CHAT
        )
        
        # Message not for agent
        message5 = Message(
            source="other1",
            to="other2",
            content="Test message 5",
            message_type=MessageType.CHAT
        )
        
        assert agent.is_intended_for_me(message1) is True
        assert agent.is_intended_for_me(message2) is True
        assert agent.is_intended_for_me(message3) is True
        assert agent.is_intended_for_me(message4) is True
        assert agent.is_intended_for_me(message5) is False
    
    @pytest.mark.asyncio
    async def test_handle_message_chat(self, agent):
        """Test that handle_message correctly processes chat messages."""
        # Create a chat message
        message = Message(
            source="other",
            to="test_agent",
            content="Test chat message",
            message_type=MessageType.CHAT
        )
        
        # Handle the message
        await agent.handle_message(message)
        
        # Check that the brain handled the message
        assert len(agent.brain.messages) == 1
        assert agent.brain.messages[0] == message
    
    @pytest.mark.asyncio
    async def test_handle_message_helo(self, agent):
        """Test that handle_message correctly processes HELO messages."""
        # Create a HELO message
        message = Message(
            source="other",
            to="test_agent",
            content="",
            message_type=MessageType.HELO
        )
        
        # Handle the message
        await agent.handle_message(message)
        
        # Check that an ACK response was sent
        assert len(agent._message_sender.sent_messages) == 1
        sent_message = agent._message_sender.sent_messages[0]
        assert sent_message.source == "test_agent"
        assert sent_message.to == "other"
        assert sent_message.message_type == MessageType.ACK
    
    @pytest.mark.asyncio
    async def test_register_message_handler(self, agent):
        """Test that message handlers can be registered and called."""
        # Create a mock handler
        handler_called = False
        handler_message = None
        
        async def mock_handler(message):
            nonlocal handler_called, handler_message
            handler_called = True
            handler_message = message
        
        # Register the handler
        agent.register_message_handler(MessageType.CHAT, mock_handler)
        
        # Create a message
        message = Message(
            source="other",
            to="test_agent",
            content="Test handler message",
            message_type=MessageType.CHAT
        )
        
        # Handle the message
        await agent.handle_message(message)
        
        # Check that the handler was called
        assert handler_called is True
        assert handler_message == message
        
        # Check that the brain did not handle the message (handler took precedence)
        assert len(agent.brain.messages) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop(self, agent):
        """Test that start and stop methods work correctly."""
        # Start the agent
        await agent.start()
        assert agent._running is True
        
        # Stop the agent
        await agent.stop()
        assert agent._running is False
