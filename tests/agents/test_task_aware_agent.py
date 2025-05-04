"""
Tests for the TaskAwareAgent class.

These tests verify that the TaskAwareAgent correctly manages tasks and conversations.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from networkkit.messages import Message, MessageType

from agentkit.agents.task_aware_agent import TaskAwareAgent
from agentkit.memory.threaded_memory import ThreadedMemory
from agentkit.memory.conversation.task import Task


@pytest.fixture
def task_aware_agent():
    """Create a TaskAwareAgent for testing."""
    agent = TaskAwareAgent(
        name="TestAgent",
        config={"name": "TestAgent"}
    )
    return agent


@pytest.mark.asyncio
async def test_agent_initialization(task_aware_agent):
    """Test that a TaskAwareAgent is initialized correctly."""
    # Verify agent attributes
    assert task_aware_agent.name == "TestAgent"
    assert isinstance(task_aware_agent.memory, ThreadedMemory)
    assert isinstance(task_aware_agent.task_queue, asyncio.PriorityQueue)
    assert task_aware_agent._current_task is None
    assert task_aware_agent._task_processor_task is None


@pytest.mark.asyncio
async def test_agent_start_stop(task_aware_agent):
    """Test starting and stopping a TaskAwareAgent."""
    # Start the agent
    await task_aware_agent.start()
    
    # Verify agent is running
    assert task_aware_agent._running
    assert task_aware_agent._task_processor_task is not None
    
    # Stop the agent
    await task_aware_agent.stop()
    
    # Verify agent is stopped
    assert not task_aware_agent._running
    assert not task_aware_agent._tasks


@pytest.mark.asyncio
async def test_handle_message(task_aware_agent):
    """Test handling a message with a TaskAwareAgent."""
    # Start the agent
    await task_aware_agent.start()
    
    try:
        # Create a mock message
        message = Message(
            source="Sender",
            to="TestAgent",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Mock the brain's handle_chat_message method
        task_aware_agent.brain = MagicMock()
        task_aware_agent.brain.handle_chat_message = MagicMock(
            return_value=asyncio.Future()
        )
        task_aware_agent.brain.handle_chat_message.return_value.set_result(None)
        
        # Handle the message
        await task_aware_agent.handle_message(message)
        
        # Wait for the task to be processed
        await asyncio.sleep(0.1)
        
        # Verify the brain's handle_chat_message method was called
        task_aware_agent.brain.handle_chat_message.assert_called_once_with(message)
        
    finally:
        # Stop the agent
        await task_aware_agent.stop()


@pytest.mark.asyncio
async def test_add_task(task_aware_agent):
    """Test adding a task to a TaskAwareAgent."""
    # Start the agent
    await task_aware_agent.start()
    
    try:
        # Add a task
        task = await task_aware_agent.add_task(
            description="Test task",
            priority=5,
            due_time=datetime.now() + timedelta(minutes=5)
        )
        
        # Verify task attributes
        assert task.description == "Test task"
        assert task.priority == 5
        assert task.status == "pending"
        
        # Wait for a moment to allow the task to be processed
        await asyncio.sleep(0.1)
        
        # Get all tasks from the conversation manager
        tasks = list(task_aware_agent.memory.conversation_manager.tasks.values())
        
        # Verify the task was added
        assert len(tasks) == 1
        assert tasks[0].task_id == task.task_id
        
    finally:
        # Stop the agent
        await task_aware_agent.stop()


@pytest.mark.asyncio
async def test_get_active_conversations(task_aware_agent):
    """Test getting active conversations from a TaskAwareAgent."""
    # Start the agent
    await task_aware_agent.start()
    
    try:
        # Create a mock message
        message = Message(
            source="Sender",
            to="TestAgent",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Handle the message to create a conversation
        await task_aware_agent.handle_message(message)
        
        # Wait for a moment
        await asyncio.sleep(0.1)
        
        # Get active conversations
        active_conversations = await task_aware_agent.get_active_conversations()
        
        # Verify active conversations
        assert len(active_conversations) == 1
        
    finally:
        # Stop the agent
        await task_aware_agent.stop()
