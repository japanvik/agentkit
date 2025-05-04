"""
Tests for the ConversationManager class in the conversation management module.

These tests verify that the ConversationManager correctly manages conversations and tasks.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from networkkit.messages import Message, MessageType

from agentkit.memory.conversation.manager import ConversationManager
from agentkit.memory.conversation.context import ConversationContext
from agentkit.memory.conversation.task import Task


class TestConversationManager:
    """Tests for the ConversationManager class."""
    
    def test_manager_initialization(self):
        """Test that a conversation manager is initialized correctly."""
        # Create a conversation manager
        manager = ConversationManager()
        
        # Verify manager attributes
        assert manager.conversations == {}
        assert manager.tasks == {}
    
    def test_get_or_create_conversation_direct(self):
        """Test getting or creating a direct conversation."""
        # Create a conversation manager
        manager = ConversationManager()
        
        # Create a mock direct message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Get or create a conversation
        conversation = manager.get_or_create_conversation(message)
        
        # Verify conversation
        assert conversation.conversation_id == "Receiver:Sender"  # Alphabetically sorted
        assert conversation in manager.conversations.values()
        assert manager.conversations[conversation.conversation_id] == conversation
        
        # Get the same conversation again
        conversation2 = manager.get_or_create_conversation(message)
        
        # Verify it's the same conversation
        assert conversation2 == conversation
    
    def test_get_or_create_conversation_broadcast(self):
        """Test getting or creating a broadcast conversation."""
        # Create a conversation manager
        manager = ConversationManager()
        
        # Create a mock broadcast message
        message = Message(
            source="Sender",
            to="ALL",
            content="Test broadcast message",
            message_type=MessageType.CHAT
        )
        
        # Get or create a conversation
        conversation = manager.get_or_create_conversation(message)
        
        # Verify conversation
        assert conversation.conversation_id.startswith("broadcast:Sender:")
        assert conversation in manager.conversations.values()
        assert manager.conversations[conversation.conversation_id] == conversation
    
    def test_get_conversation_by_id(self):
        """Test getting a conversation by ID."""
        # Create a conversation manager
        manager = ConversationManager()
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Get or create a conversation
        conversation = manager.get_or_create_conversation(message)
        
        # Get the conversation by ID
        retrieved_conversation = manager.get_conversation_by_id(conversation.conversation_id)
        
        # Verify it's the same conversation
        assert retrieved_conversation == conversation
        
        # Try to get a non-existent conversation
        non_existent = manager.get_conversation_by_id("non-existent")
        
        # Verify it's None
        assert non_existent is None
    
    def test_get_conversations_for_participant(self):
        """Test getting conversations for a participant."""
        # Create a conversation manager
        manager = ConversationManager()
        
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
        
        message3 = Message(
            source="OtherSender",
            to="OtherReceiver",
            content="Test message 3",
            message_type=MessageType.CHAT
        )
        
        # Get or create conversations
        conversation1 = manager.get_or_create_conversation(message1)
        conversation2 = manager.get_or_create_conversation(message2)
        conversation3 = manager.get_or_create_conversation(message3)
        
        # Get conversations for Sender
        sender_conversations = manager.get_conversations_for_participant("Sender")
        
        # Verify conversations
        assert len(sender_conversations) == 2
        assert conversation1 in sender_conversations
        assert conversation2 in sender_conversations
        assert conversation3 not in sender_conversations
        
        # Get conversations for OtherSender
        other_sender_conversations = manager.get_conversations_for_participant("OtherSender")
        
        # Verify conversations
        assert len(other_sender_conversations) == 1
        assert conversation3 in other_sender_conversations
    
    def test_get_active_conversations(self):
        """Test getting active conversations."""
        # Create a conversation manager
        manager = ConversationManager()
        
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
        
        # Get or create conversations
        conversation1 = manager.get_or_create_conversation(message1)
        conversation2 = manager.get_or_create_conversation(message2)
        
        # Set last_activity for conversation1 to be older
        conversation1.last_activity = datetime.now() - timedelta(minutes=30)
        
        # Get active conversations with default max_age (60 minutes)
        active_conversations = manager.get_active_conversations()
        
        # Verify both conversations are active
        assert len(active_conversations) == 2
        assert conversation1 in active_conversations
        assert conversation2 in active_conversations
        
        # Get active conversations with max_age of 15 minutes
        active_conversations = manager.get_active_conversations(max_age_minutes=15)
        
        # Verify only conversation2 is active
        assert len(active_conversations) == 1
        assert conversation1 not in active_conversations
        assert conversation2 in active_conversations
    
    def test_add_task(self):
        """Test adding a task."""
        # Create a conversation manager
        manager = ConversationManager()
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Get or create a conversation
        conversation = manager.get_or_create_conversation(message)
        
        # Create a task
        task = Task.create(
            conversation_id=conversation.conversation_id,
            message=message,
            description="Test task"
        )
        
        # Add the task
        manager.add_task(task)
        
        # Verify task was added to manager
        assert task.task_id in manager.tasks
        assert manager.tasks[task.task_id] == task
        
        # Verify task was added to conversation
        assert task in conversation.tasks
    
    def test_get_task(self):
        """Test getting a task by ID."""
        # Create a conversation manager
        manager = ConversationManager()
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Get or create a conversation
        conversation = manager.get_or_create_conversation(message)
        
        # Create a task
        task = Task.create(
            conversation_id=conversation.conversation_id,
            message=message,
            description="Test task"
        )
        
        # Add the task
        manager.add_task(task)
        
        # Get the task by ID
        retrieved_task = manager.get_task(task.task_id)
        
        # Verify it's the same task
        assert retrieved_task == task
        
        # Try to get a non-existent task
        non_existent = manager.get_task("non-existent")
        
        # Verify it's None
        assert non_existent is None
    
    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        # Create a conversation manager
        manager = ConversationManager()
        
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Get or create a conversation
        conversation = manager.get_or_create_conversation(message)
        
        # Create tasks with different statuses
        task1 = Task.create(
            conversation_id=conversation.conversation_id,
            message=message,
            description="Pending task 1"
        )
        
        task2 = Task.create(
            conversation_id=conversation.conversation_id,
            message=message,
            description="In progress task"
        )
        task2.mark_in_progress()
        
        task3 = Task.create(
            conversation_id=conversation.conversation_id,
            message=message,
            description="Completed task"
        )
        task3.mark_completed()
        
        task4 = Task.create(
            conversation_id=conversation.conversation_id,
            message=message,
            description="Failed task"
        )
        task4.mark_failed("Test failure")
        
        task5 = Task.create(
            conversation_id=conversation.conversation_id,
            message=message,
            description="Pending task 2"
        )
        
        # Add tasks to the manager
        manager.add_task(task1)
        manager.add_task(task2)
        manager.add_task(task3)
        manager.add_task(task4)
        manager.add_task(task5)
        
        # Get pending tasks
        pending_tasks = manager.get_pending_tasks()
        
        # Verify pending tasks
        assert len(pending_tasks) == 2
        assert task1 in pending_tasks
        assert task5 in pending_tasks
        assert task2 not in pending_tasks
        assert task3 not in pending_tasks
        assert task4 not in pending_tasks
