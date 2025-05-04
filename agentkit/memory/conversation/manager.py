"""
Conversation manager module for conversation management.

This module provides the ConversationManager class, which manages multiple conversations
and their associated tasks. It provides methods for creating, retrieving, and managing
conversations and tasks.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from networkkit.messages import Message

from .context import ConversationContext
from .task import Task


class ConversationManager:
    """
    Manages multiple conversations and their associated tasks.
    
    The ConversationManager is responsible for creating, retrieving, and managing
    conversations and tasks. It provides methods for getting or creating conversations
    based on messages, retrieving conversations by ID or participant, and managing tasks.
    
    Attributes:
        conversations: A dictionary mapping conversation IDs to ConversationContext objects.
        tasks: A dictionary mapping task IDs to Task objects.
    """
    
    def __init__(self):
        """Initialize a new ConversationManager."""
        self.conversations: Dict[str, ConversationContext] = {}
        self.tasks: Dict[str, Task] = {}
    
    def get_or_create_conversation(self, message: Message) -> ConversationContext:
        """
        Get an existing conversation or create a new one based on the message.
        
        Args:
            message: The message to get or create a conversation for.
            
        Returns:
            The existing or newly created ConversationContext.
        """
        conversation_id = self._get_conversation_id(message)
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(conversation_id, message)
        return self.conversations[conversation_id]
    
    def _get_conversation_id(self, message: Message) -> str:
        """
        Generate or extract a conversation ID from a message.
        
        For direct messages: "{smaller_name}:{larger_name}" (alphabetically sorted)
        For broadcast messages: Use a combination of participants who have engaged
        
        Args:
            message: The message to generate or extract a conversation ID from.
            
        Returns:
            A conversation ID.
        """
        if message.to != 'ALL':
            # Direct conversation between two agents
            participants = sorted([message.source, message.to])
            return f"{participants[0]}:{participants[1]}"
        else:
            # Broadcast message - need to determine conversation from context
            # Options:
            # 1. Use message thread_id if available
            if hasattr(message, 'thread_id') and message.thread_id:
                return f"thread:{message.thread_id}"
            
            # 2. Use reply_to to link to previous conversation
            if hasattr(message, 'reply_to') and message.reply_to:
                # Find the conversation containing the replied message
                replied_msg_conv = self._find_conversation_by_message_id(message.reply_to)
                if replied_msg_conv:
                    return replied_msg_conv
            
            # 3. If no thread info, create a new broadcast conversation
            return f"broadcast:{message.source}:{uuid.uuid4()}"
    
    def _find_conversation_by_message_id(self, message_id: str) -> Optional[str]:
        """
        Find the conversation containing a message with the given ID.
        
        Args:
            message_id: The ID of the message to find.
            
        Returns:
            The ID of the conversation containing the message, or None if no such conversation exists.
        """
        for conv_id, conversation in self.conversations.items():
            for msg in conversation.history:
                if hasattr(msg, 'id') and msg.id == message_id:
                    return conv_id
        return None
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        Get a conversation by its ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve.
            
        Returns:
            The conversation with the specified ID, or None if no such conversation exists.
        """
        return self.conversations.get(conversation_id)
    
    def get_conversations_for_participant(self, participant_name: str) -> List[ConversationContext]:
        """
        Get all conversations involving a specific participant.
        
        Args:
            participant_name: The name of the participant to get conversations for.
            
        Returns:
            A list of conversations involving the specified participant.
        """
        return [
            conv for conv in self.conversations.values()
            if participant_name in conv.participants
        ]
    
    def get_active_conversations(self, max_age_minutes: int = 60) -> List[ConversationContext]:
        """
        Get all conversations with activity within the specified time window.
        
        Args:
            max_age_minutes: The maximum age of conversations to consider active, in minutes.
            
        Returns:
            A list of conversations with activity within the specified time window.
        """
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        return [
            conv for conv in self.conversations.values()
            if conv.last_activity >= cutoff_time
        ]
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to the appropriate conversation and the tasks dictionary.
        
        Args:
            task: The task to add.
        """
        self.tasks[task.task_id] = task
        if task.conversation_id in self.conversations:
            self.conversations[task.conversation_id].add_task(task)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by its ID.
        
        Args:
            task_id: The ID of the task to retrieve.
            
        Returns:
            The task with the specified ID, or None if no such task exists.
        """
        return self.tasks.get(task_id)
    
    def get_pending_tasks(self) -> List[Task]:
        """
        Get all pending tasks across all conversations.
        
        Returns:
            A list of all pending tasks.
        """
        return [task for task in self.tasks.values() if task.status == "pending"]
    
    def get_pending_tasks_for_conversation(self, conversation_id: str) -> List[Task]:
        """
        Get all pending tasks for a specific conversation.
        
        Args:
            conversation_id: The ID of the conversation to get pending tasks for.
            
        Returns:
            A list of pending tasks for the specified conversation.
        """
        conversation = self.get_conversation_by_id(conversation_id)
        if conversation:
            return conversation.get_pending_tasks()
        return []
    
    def get_pending_tasks_for_participant(self, participant_name: str) -> List[Task]:
        """
        Get all pending tasks for conversations involving a specific participant.
        
        Args:
            participant_name: The name of the participant to get pending tasks for.
            
        Returns:
            A list of pending tasks for conversations involving the specified participant.
        """
        conversations = self.get_conversations_for_participant(participant_name)
        pending_tasks = []
        for conversation in conversations:
            pending_tasks.extend(conversation.get_pending_tasks())
        return pending_tasks
    
    def __repr__(self) -> str:
        """
        Get a string representation of the conversation manager.
        
        Returns:
            A string representation of the conversation manager.
        """
        return f"ConversationManager(conversations={len(self.conversations)}, tasks={len(self.tasks)})"
