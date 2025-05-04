"""
Task module for conversation management.

This module provides the Task class, which represents a pending task or action
within a conversation. Tasks are used to track and prioritize actions that an agent
needs to perform in response to messages or other events.
"""

import uuid
from datetime import datetime
from typing import Optional, Any, Dict

from networkkit.messages import Message


class Task:
    """
    Represents a pending task or action within a conversation.
    
    Tasks are used to track and prioritize actions that an agent needs to perform
    in response to messages or other events. Each task has a priority, which can
    be used to determine the order in which tasks are processed.
    
    Attributes:
        task_id: A unique identifier for the task.
        conversation_id: The ID of the conversation this task belongs to.
        message: The message that triggered this task.
        description: A human-readable description of the task.
        priority: The base priority of the task (higher values = higher priority).
        due_time: An optional deadline for the task.
        status: The current status of the task (pending, in_progress, completed, failed).
        created_at: The time when the task was created.
    """
    
    def __init__(self, 
                 task_id: str,
                 conversation_id: str,
                 message: Message,
                 description: str,
                 priority: int = 0,
                 due_time: Optional[datetime] = None):
        """
        Initialize a new Task.
        
        Args:
            task_id: A unique identifier for the task.
            conversation_id: The ID of the conversation this task belongs to.
            message: The message that triggered this task.
            description: A human-readable description of the task.
            priority: The base priority of the task (higher values = higher priority).
            due_time: An optional deadline for the task.
        """
        self.task_id = task_id
        self.conversation_id = conversation_id
        self.message = message
        self.description = description
        self.priority = priority
        self.due_time = due_time
        self.status = "pending"  # pending, in_progress, completed, failed
        self.created_at = datetime.now()
        self.failure_reason: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    @classmethod
    def create(cls, conversation_id: str, message: Message, description: str, 
               priority: int = 0, due_time: Optional[datetime] = None) -> 'Task':
        """
        Create a new Task with a generated UUID.
        
        Args:
            conversation_id: The ID of the conversation this task belongs to.
            message: The message that triggered this task.
            description: A human-readable description of the task.
            priority: The base priority of the task (higher values = higher priority).
            due_time: An optional deadline for the task.
            
        Returns:
            A new Task instance with a generated UUID.
        """
        task_id = str(uuid.uuid4())
        return cls(task_id, conversation_id, message, description, priority, due_time)
    
    @property
    def is_overdue(self) -> bool:
        """
        Check if the task is overdue based on due_time.
        
        Returns:
            True if the task has a due_time and it has passed, False otherwise.
        """
        if not self.due_time:
            return False
        return datetime.now() > self.due_time
    
    def calculate_effective_priority(self) -> int:
        """
        Calculate effective priority based on base priority and other factors.
        
        The effective priority is calculated based on:
        - Base priority
        - Whether the task is overdue
        - Whether the conversation is direct or broadcast
        
        Returns:
            The effective priority as an integer.
        """
        effective_priority = self.priority
        
        # Increase priority for overdue tasks
        if self.is_overdue:
            time_overdue = datetime.now() - self.due_time
            overdue_minutes = time_overdue.total_seconds() / 60
            effective_priority += min(int(overdue_minutes / 5), 10)  # Max +10 for overdue
        
        # Direct messages get higher priority than broadcast
        if not self.conversation_id.startswith("broadcast:"):
            effective_priority += 5
        
        return effective_priority
    
    def mark_in_progress(self) -> None:
        """Mark this task as in progress."""
        self.status = "in_progress"
    
    def mark_completed(self) -> None:
        """Mark this task as completed."""
        self.status = "completed"
    
    def mark_failed(self, reason: str = None) -> None:
        """
        Mark this task as failed.
        
        Args:
            reason: An optional reason for the failure.
        """
        self.status = "failed"
        if reason:
            self.failure_reason = reason
    
    def __repr__(self) -> str:
        """
        Get a string representation of the task.
        
        Returns:
            A string representation of the task.
        """
        return f"Task(id={self.task_id}, desc={self.description}, priority={self.priority}, status={self.status})"
