"""
Conversation context module for conversation management.

This module provides the ConversationContext class, which encapsulates all information
related to a specific conversation. It maintains a list of messages, participants,
and tasks associated with the conversation.
"""

from datetime import datetime
from typing import List, Set, Dict, Any, Optional

from networkkit.messages import Message

from .task import Task


class ConversationContext:
    """
    Encapsulates all information related to a specific conversation.
    
    A conversation context maintains a list of messages, participants, and tasks
    associated with a conversation. It provides methods for adding messages and tasks,
    retrieving recent messages, and managing conversation state.
    
    Attributes:
        conversation_id: A unique identifier for the conversation.
        participants: A set of participant names involved in the conversation.
        history: A list of messages in the conversation.
        tasks: A list of tasks associated with the conversation.
        last_activity: The time of the last activity in the conversation.
        state: A dictionary for storing conversation-specific state.
        is_broadcast: Whether this is a broadcast conversation.
    """
    
    def __init__(self, conversation_id: str, initial_message: Optional[Message] = None):
        """
        Initialize a new ConversationContext.
        
        Args:
            conversation_id: A unique identifier for the conversation.
            initial_message: An optional initial message to add to the conversation.
        """
        self.conversation_id = conversation_id
        self.participants: Set[str] = set()
        self.history: List[Message] = []
        self.tasks: List[Task] = []
        self.last_activity: datetime = datetime.now()
        self.state: Dict[str, Any] = {}
        self.is_broadcast = conversation_id.startswith("broadcast:")
        
        # Add initial message if provided
        if initial_message:
            self.add_message(initial_message)
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to this conversation and update metadata.
        
        Args:
            message: The message to add to the conversation.
        """
        self.history.append(message)
        self.participants.add(message.source)
        if message.to != 'ALL':
            self.participants.add(message.to)
        self.last_activity = datetime.now()
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to this conversation.
        
        Args:
            task: The task to add to the conversation.
        """
        self.tasks.append(task)
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """
        Get the most recent messages in this conversation.
        
        Args:
            count: The maximum number of messages to return.
            
        Returns:
            A list of the most recent messages, up to the specified count.
        """
        return self.history[-count:] if len(self.history) > count else self.history
    
    def get_pending_tasks(self) -> List[Task]:
        """
        Get all pending tasks in this conversation.
        
        Returns:
            A list of pending tasks in this conversation.
        """
        return [task for task in self.tasks if task.status == "pending"]
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Get a task by its ID.
        
        Args:
            task_id: The ID of the task to retrieve.
            
        Returns:
            The task with the specified ID, or None if no such task exists.
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get a message by its ID.
        
        Args:
            message_id: The ID of the message to retrieve.
            
        Returns:
            The message with the specified ID, or None if no such message exists.
        """
        for message in self.history:
            if message.id == message_id:
                return message
        return None
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set a value in the conversation state.
        
        Args:
            key: The key to set.
            value: The value to set.
        """
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the conversation state.
        
        Args:
            key: The key to get.
            default: The default value to return if the key is not found.
            
        Returns:
            The value associated with the key, or the default value if the key is not found.
        """
        return self.state.get(key, default)
    
    def __repr__(self) -> str:
        """
        Get a string representation of the conversation context.
        
        Returns:
            A string representation of the conversation context.
        """
        return (f"ConversationContext(id={self.conversation_id}, "
                f"participants={self.participants}, "
                f"messages={len(self.history)}, "
                f"tasks={len(self.tasks)})")
