"""
Tests for the Task class in the conversation management module.

These tests verify that the Task class correctly manages task state, priority
calculation, and status transitions.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from networkkit.messages import Message, MessageType

from agentkit.memory.conversation.task import Task


class TestTask:
    """Tests for the Task class."""
    
    def test_task_initialization(self):
        """Test that a task is initialized correctly."""
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Create a task
        task = Task(
            task_id="task-123",
            conversation_id="conv-123",
            message=message,
            description="Test task",
            priority=5
        )
        
        # Verify task attributes
        assert task.task_id == "task-123"
        assert task.conversation_id == "conv-123"
        assert task.message == message
        assert task.description == "Test task"
        assert task.priority == 5
        assert task.status == "pending"
        assert task.failure_reason is None
        assert isinstance(task.created_at, datetime)
    
    def test_task_create_factory_method(self):
        """Test the create factory method."""
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Create a task using the factory method
        task = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task",
            priority=5
        )
        
        # Verify task attributes
        assert task.conversation_id == "conv-123"
        assert task.message == message
        assert task.description == "Test task"
        assert task.priority == 5
        assert task.status == "pending"
        assert task.failure_reason is None
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.task_id, str)
        assert len(task.task_id) > 0
    
    def test_is_overdue(self):
        """Test the is_overdue property."""
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Create a task with no due time
        task1 = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task 1"
        )
        
        # Create a task with a future due time
        future_time = datetime.now() + timedelta(hours=1)
        task2 = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task 2",
            due_time=future_time
        )
        
        # Create a task with a past due time
        past_time = datetime.now() - timedelta(hours=1)
        task3 = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task 3",
            due_time=past_time
        )
        
        # Verify overdue status
        assert not task1.is_overdue
        assert not task2.is_overdue
        assert task3.is_overdue
    
    def test_calculate_effective_priority(self):
        """Test the calculate_effective_priority method."""
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Create a task with base priority
        task1 = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task 1",
            priority=5
        )
        
        # Create a task with an overdue due time
        past_time = datetime.now() - timedelta(minutes=30)
        task2 = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task 2",
            priority=5,
            due_time=past_time
        )
        
        # Create a task with a broadcast conversation ID
        task3 = Task.create(
            conversation_id="broadcast:test",
            message=message,
            description="Test task 3",
            priority=5
        )
        
        # Verify effective priorities
        # Direct message task should get +5 priority
        assert task1.calculate_effective_priority() == 10
        
        # Overdue task should get additional priority based on how overdue it is
        # 30 minutes overdue = +6 priority (30/5 = 6)
        # Plus +5 for direct message
        assert task2.calculate_effective_priority() >= 16
        
        # Broadcast message task should not get the direct message bonus
        assert task3.calculate_effective_priority() == 5
    
    def test_status_transitions(self):
        """Test task status transitions."""
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
        
        # Verify initial status
        assert task.status == "pending"
        
        # Test status transitions
        task.mark_in_progress()
        assert task.status == "in_progress"
        
        task.mark_completed()
        assert task.status == "completed"
        
        # Create a new task for failure test
        task = Task.create(
            conversation_id="conv-123",
            message=message,
            description="Test task"
        )
        
        # Test failure with reason
        task.mark_failed("Test failure reason")
        assert task.status == "failed"
        assert task.failure_reason == "Test failure reason"
    
    def test_repr(self):
        """Test the __repr__ method."""
        # Create a mock message
        message = Message(
            source="Sender",
            to="Receiver",
            content="Test message",
            message_type=MessageType.CHAT
        )
        
        # Create a task
        task = Task(
            task_id="task-123",
            conversation_id="conv-123",
            message=message,
            description="Test task",
            priority=5
        )
        
        # Verify repr
        repr_str = repr(task)
        assert "Task" in repr_str
        assert "task-123" in repr_str
        assert "Test task" in repr_str
        assert "5" in repr_str
        assert "pending" in repr_str
