"""
Threaded memory module for conversation-aware memory management.

This module provides the ThreadedMemory class, which extends BaseMemory to support
conversation-specific memory. It uses the ConversationManager to store messages in
the appropriate conversation context.
"""

from typing import List, Optional, Dict, Any

from networkkit.messages import Message, MessageType

from agentkit.memory.base_memory import BaseMemory
from agentkit.memory.conversation.manager import ConversationManager
from agentkit.memory.conversation.context import ConversationContext


class ThreadedMemory(BaseMemory):
    """
    Extends BaseMemory to support conversation-specific memory.
    
    ThreadedMemory uses a ConversationManager to store messages in the appropriate
    conversation context. It provides methods for retrieving conversation-specific
    history and formatting chat context for specific participants.
    
    Attributes:
        max_history_length: The maximum number of messages to keep in history.
        conversation_manager: The ConversationManager used to manage conversations.
    """
    
    def __init__(self, max_history_length: int = 100):
        """
        Initialize a new ThreadedMemory.
        
        Args:
            max_history_length: The maximum number of messages to keep in history.
        """
        super().__init__(max_history_length)
        self.conversation_manager = ConversationManager()
    
    def remember(self, message: Message) -> None:
        """
        Store a message in the appropriate conversation.
        
        Args:
            message: The message to store.
        """
        conversation = self.conversation_manager.get_or_create_conversation(message)
        
        # Check if the message is already in the conversation
        # This prevents duplicate messages when the conversation is created with the message
        if message not in conversation.history:
            conversation.add_message(message)
    
    def get_history(self) -> List[Message]:
        """
        Get combined history from all conversations.
        
        Returns:
            A list of messages from all conversations, sorted by timestamp and limited
            by max_history_length.
        """
        all_messages = []
        for conversation in self.conversation_manager.conversations.values():
            all_messages.extend(conversation.history)
        
        # Sort by timestamp and limit by max_history_length
        all_messages.sort(key=lambda m: getattr(m, 'timestamp', 0) if hasattr(m, 'timestamp') else 0)
        if self.max_history_length > 0 and len(all_messages) > self.max_history_length:
            all_messages = all_messages[-self.max_history_length:]
        
        return all_messages
    
    def get_chat_context(self, target: str, prefix: str = "", 
                         user_role_name: str = "", 
                         assistant_role_name: str = "") -> str:
        """
        Get formatted chat context for a specific target.
        
        Args:
            target: The target to get chat context for.
            prefix: A prefix to add to each line of the chat context.
            user_role_name: The name to use for the user role.
            assistant_role_name: The name to use for the assistant role.
            
        Returns:
            A formatted chat context string.
        """
        # Get conversations involving target
        conversations = self.conversation_manager.get_conversations_for_participant(target)
        if not conversations:
            return ""
        
        # Sort conversations by last activity
        conversations.sort(key=lambda c: c.last_activity, reverse=True)
        
        # Format chat context from most recent conversation
        context = ""
        for msg in conversations[0].history:
            if msg.message_type != MessageType.CHAT:
                continue
            
            # In the test, "Sender" is the user and "Receiver" is the assistant
            # So if the message is from "Sender", use the user_role_name
            # If the message is from "Receiver", use the assistant_role_name
            if msg.source == target:
                speaker = assistant_role_name if assistant_role_name else msg.source
            else:  # msg.to == target
                speaker = user_role_name if user_role_name else msg.source
            
            context += f"{prefix}{speaker}: {msg.content.strip()}\n"
        
        return context
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        Get a conversation by its ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve.
            
        Returns:
            The conversation with the specified ID, or None if no such conversation exists.
        """
        return self.conversation_manager.get_conversation_by_id(conversation_id)
    
    def get_conversation_id(self, message: Message) -> str:
        """
        Get the conversation ID for a message.
        
        Args:
            message: The message to get the conversation ID for.
            
        Returns:
            The conversation ID for the message.
        """
        return self.conversation_manager._get_conversation_id(message)
    
    def get_conversations_for_participant(self, participant_name: str) -> List[ConversationContext]:
        """
        Get all conversations involving a specific participant.
        
        Args:
            participant_name: The name of the participant to get conversations for.
            
        Returns:
            A list of conversations involving the specified participant.
        """
        return self.conversation_manager.get_conversations_for_participant(participant_name)
    
    def get_active_conversations(self, max_age_minutes: int = 60) -> List[ConversationContext]:
        """
        Get all conversations with activity within the specified time window.
        
        Args:
            max_age_minutes: The maximum age of conversations to consider active, in minutes.
            
        Returns:
            A list of conversations with activity within the specified time window.
        """
        return self.conversation_manager.get_active_conversations(max_age_minutes)
    
    def clear(self) -> None:
        """Clear all conversations and tasks."""
        self.conversation_manager = ConversationManager()
        
    def chat_log_for(self, target: str) -> List[Message]:
        """
        Retrieve the chat log for the specified target.
        
        This method filters the conversation history for messages involving the
        specified target and returns them as a list of Message objects.
        
        Args:
            target: The target user or entity for which to retrieve the chat log
            
        Returns:
            List[Message]: List of Message objects representing the chat log,
                          typically ordered from oldest to newest
        """
        # Get conversations involving target
        conversations = self.conversation_manager.get_conversations_for_participant(target)
        if not conversations:
            return []
            
        # Sort conversations by last activity
        conversations.sort(key=lambda c: c.last_activity, reverse=True)
        
        # Get messages from the most recent conversation
        messages = []
        for conversation in conversations:
            for msg in conversation.history:
                if msg.message_type == MessageType.CHAT and (msg.source == target or msg.to == target):
                    messages.append(msg)
                    
        # Sort messages by timestamp
        messages.sort(key=lambda m: getattr(m, 'timestamp', 0) if hasattr(m, 'timestamp') else 0)
        
        return messages
