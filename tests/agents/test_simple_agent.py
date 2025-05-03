"""
Tests for the SimpleAgent class.

This module contains tests for the SimpleAgent class functionality, including
component initialization, configuration, and message handling.
"""
# Standard library imports
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

# Third-party imports
import pytest
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.agents.simple_agent import SimpleAgent, BUILTIN_BRAINS, BUILTIN_MEMORIES
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.simple_memory import SimpleMemory
from agentkit.common.interfaces import MessageSender, ComponentConfig


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


class TestSimpleAgent:
    """Test suite for SimpleAgent class."""
    
    @pytest.fixture
    def config(self):
        """Create a configuration dictionary for testing."""
        return {
            "description": "Test agent description",
            "model": "test_model",
            "system_prompt": "You are a test agent",
            "user_prompt": "This is a test",
            "api_config": {"temperature": 0.7},
            "brain_type": "SimpleBrain",
            "memory_type": "SimpleMemory"
        }
    
    @pytest.fixture
    def message_sender(self):
        """Create a MockMessageSender instance for testing."""
        return MockMessageSender()
    
    def test_initialization_with_defaults(self, config, message_sender):
        """Test that the agent initializes correctly with default components."""
        agent = SimpleAgent(
            name="test_agent",
            config=config,
            message_sender=message_sender
        )
        
        # Check basic properties
        assert agent.name == "test_agent"
        assert agent.config == config
        
        # Check that components were created
        assert isinstance(agent.brain, SimpleBrain)
        assert isinstance(agent.memory, SimpleMemory)
        
        # Check that brain was configured correctly
        assert agent.brain.name == "test_agent"
        assert agent.brain.description == "Test agent description"
        assert agent.brain.model == "test_model"
        assert agent.brain.system_prompt == "You are a test agent"
        assert agent.brain.user_prompt == "This is a test"
        assert agent.brain.api_config == {"temperature": 0.7}
        
        # Check that components were configured with ComponentConfig
        assert agent.brain.component_config == agent.component_config
        
        # Check that message_sender was set
        assert agent._message_sender == message_sender
    
    def test_initialization_with_custom_components(self, config, message_sender):
        """Test that the agent initializes correctly with custom components."""
        # Create custom components
        brain = SimpleBrain(
            name="custom_brain",
            description="Custom brain",
            model="custom_model",
            memory_manager=SimpleMemory(),
            system_prompt="Custom system prompt",
            user_prompt="Custom user prompt"
        )
        memory = SimpleMemory(max_history_length=20)
        
        # Create agent with custom components
        agent = SimpleAgent(
            name="test_agent",
            config=config,
            brain=brain,
            memory=memory,
            message_sender=message_sender
        )
        
        # Check that custom components were used
        assert agent.brain == brain
        assert agent.memory == memory
        
        # Check that components were configured with ComponentConfig
        assert agent.brain.component_config == agent.component_config
    
    def test_fallback_to_default_components(self, config, message_sender):
        """Test that the agent falls back to default components if specified ones are not available."""
        # Create a brain and memory to pass directly
        brain = SimpleBrain(
            name="test_brain",
            description="Test brain",
            model="test_model",
            memory_manager=SimpleMemory(),
            system_prompt="Test system prompt",
            user_prompt="Test user prompt"
        )
        memory = SimpleMemory()
        
        # Create agent with non-existent component types but provide the components directly
        config["brain_type"] = "NonExistentBrain"
        config["memory_type"] = "NonExistentMemory"
        
        agent = SimpleAgent(
            name="test_agent",
            config=config,
            brain=brain,
            memory=memory,
            message_sender=message_sender
        )
        
        # Check that the provided components were used
        assert agent.brain == brain
        assert agent.memory == memory
    
    @pytest.mark.asyncio
    async def test_message_handling(self, config, message_sender):
        """Test that the agent handles messages correctly."""
        # Create agent
        agent = SimpleAgent(
            name="test_agent",
            config=config,
            message_sender=message_sender
        )
        
        # Create a mock brain with an async handle_chat_message method
        mock_brain = MagicMock()
        mock_brain.handle_chat_message = MagicMock()
        mock_brain.handle_chat_message.return_value = None
        agent.brain = mock_brain
        
        # Create a test message
        message = Message(
            source="user",
            to="test_agent",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Register a custom handler for CHAT messages
        async def custom_handler(msg):
            # This will be called instead of the brain's handle_chat_message
            pass
        
        agent.register_message_handler(MessageType.CHAT, custom_handler)
        
        # Handle the message
        await agent.handle_message(message)
        
        # The brain's handle_chat_message should not be called because we registered a custom handler
        mock_brain.handle_chat_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_start_stop(self, config, message_sender):
        """Test that start and stop methods work correctly."""
        # Create agent
        agent = SimpleAgent(
            name="test_agent",
            config=config,
            message_sender=message_sender
        )
        
        # Start the agent
        await agent.start()
        assert agent._running is True
        
        # Stop the agent
        await agent.stop()
        assert agent._running is False
