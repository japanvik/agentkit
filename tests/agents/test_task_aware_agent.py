"""
Tests for the TaskAwareAgent class.

These tests verify that the TaskAwareAgent correctly manages tasks and conversations.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

from networkkit.messages import Message, MessageType

from agentkit.agents.task_aware_agent import TaskAwareAgent
from agentkit.agents.simple_agent import SimpleAgent
from agentkit.memory.threaded_memory import ThreadedMemory
from agentkit.memory.conversation.task import Task
from agentkit.functions.functions_registry import ToolExecutionContext


class InMemoryMessageBus:
    def __init__(self):
        self.agents = {}
        self.messages = []

    def register(self, agent):
        self.agents[agent.name] = agent

    async def send_message(self, message):
        self.messages.append(message)
        recipient = self.agents.get(message.to)
        if recipient:
            await recipient.handle_message(message)


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
async def test_agent_initialization_uses_planner_model_config():
    agent = TaskAwareAgent(
        name="PlannerConfigAgent",
        config={"name": "PlannerConfigAgent", "planner_model": "ollama/qwen3-planner"},
    )
    assert agent.planner._task_generation_model == "ollama/qwen3-planner"


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

        # Mock the task_queue.put method to capture the task
        original_put = task_aware_agent.task_queue.put
        task_aware_agent.task_queue.put = MagicMock(wraps=original_put)

        # Handle the message
        await task_aware_agent.handle_message(message)

        # Wait for the task to be processed
        await asyncio.sleep(0.1)

        # Verify that task_queue.put was called (a task was added to the queue)
        task_aware_agent.task_queue.put.assert_called_once()
        
        # Get the task from the call arguments
        call_args = task_aware_agent.task_queue.put.call_args
        queued = call_args[0][0]
        if len(queued) == 3:
            _, _, task = queued
        else:
            _, task = queued
        
        # Verify the task has the expected properties
        assert task.conversation_id is not None
        assert "Process CHAT message from Sender" in task.description

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


@pytest.mark.asyncio
async def test_delegate_reply_shortcut_forwards_to_requester(task_aware_agent):
    task_aware_agent.register_delegate_wait(
        delegate="echo-agent",
        requester="Human",
        intent="delegate_and_wait_for_reply",
    )
    task_aware_agent.functions_registry.execute = AsyncMock(return_value={"status": "sent"})

    message = Message(
        source="echo-agent",
        to="TestAgent",
        content="ping-test",
        message_type=MessageType.CHAT,
    )
    handled = await task_aware_agent._handle_delegate_reply_shortcut(message)
    assert handled is True
    task_aware_agent.functions_registry.execute.assert_awaited_once()
    _, kwargs = task_aware_agent.functions_registry.execute.call_args
    assert kwargs["function"] == "send_message"
    assert kwargs["parameters"]["recipient"] == "Human"
    assert kwargs["parameters"]["content"] == "ping-test"


@pytest.mark.asyncio
async def test_delegate_reply_shortcut_two_agent_flow():
    bus = InMemoryMessageBus()

    sophia = TaskAwareAgent(
        name="Sophia",
        config={"name": "Sophia", "planner_state_dir": "agent_state/test_sophia"},
        message_sender=bus,
    )
    worker = SimpleAgent(
        name="Worker",
        config={"name": "Worker"},
        message_sender=bus,
    )

    async def worker_chat_handler(message: Message):
        await worker.send_message(
            Message(
                source="Worker",
                to=message.source,
                content=message.content,
                message_type=MessageType.CHAT,
            )
        )

    worker.register_message_handler(MessageType.CHAT, worker_chat_handler)

    bus.register(sophia)
    bus.register(worker)

    sophia.register_delegate_wait(
        delegate="Worker",
        requester="Human",
        intent="delegate_and_wait_for_reply",
    )

    await sophia.functions_registry.execute(
        function="send_message",
        parameters={
            "recipient": "Worker",
            "content": "ping-two-agent",
            "message_type": "CHAT",
        },
        context=ToolExecutionContext(agent=sophia),
    )

    routed = [
        (msg.source, msg.to, msg.content, msg.message_type)
        for msg in bus.messages
        if msg.message_type == MessageType.CHAT
    ]
    assert ("Sophia", "Worker", "ping-two-agent", MessageType.CHAT) in routed
    assert ("Worker", "Sophia", "ping-two-agent", MessageType.CHAT) in routed
    assert ("Sophia", "Human", "ping-two-agent", MessageType.CHAT) in routed
