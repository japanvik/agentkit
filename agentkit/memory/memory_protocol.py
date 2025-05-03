"""
Memory protocol definition.

This module defines the Memory protocol, which specifies the interface that all
memory implementations must adhere to. The protocol ensures that memory components
can be used interchangeably by agents and brains.

Memory components are responsible for storing conversation history, retrieving relevant
context, and managing the agent's memory of past interactions.
"""
# Standard library imports
from typing import List, Protocol

# Third-party imports
from networkkit.messages import Message


class Memory(Protocol):
    """
    Protocol defining the interface for memory storage components.
    
    This protocol outlines the expected behavior of memory components for agents.
    Concrete implementations should provide methods for storing messages (`remember`),
    retrieving the conversation history (`get_history`), and formatting chat context
    for specific targets.
    
    The Memory protocol ensures that different memory implementations can be used
    interchangeably by agents and brains, allowing for flexible memory strategies.
    """

    def remember(self, message: Message) -> None:
        """
        Store a message in the conversation history.
        
        This method is responsible for storing the provided message object
        in the agent's memory. Implementations should handle any necessary
        processing, such as truncating history if it exceeds a maximum length.
        
        Args:
            message: The message object to be stored, containing source,
                   target, content, and other metadata
        """
        ...

    def get_history(self) -> List[Message]:
        """
        Retrieve the complete conversation history from memory.
        
        This method should return all stored messages, potentially limited
        by a configured maximum history length.
        
        Returns:
            List[Message]: List of message objects representing the conversation history,
                          typically ordered from oldest to newest
        """
        ...
        
    def get_chat_context(self, target: str, prefix: str = "", 
                        user_role_name: str = "", 
                        assistant_role_name: str = "") -> str:
        """
        Retrieve chat conversation history with a specific target and format it.
        
        This method should filter the conversation history for messages involving the
        specified target and format them as a string for use in prompts or display.
        
        Args:
            target: The target name to filter the chat history for (e.g., user ID)
            prefix: A prefix to add before each message in the formatted output
            user_role_name: Optional custom name for user messages (e.g., "User")
            assistant_role_name: Optional custom name for assistant messages (e.g., "Assistant")
        
        Returns:
            str: Formatted string containing the chat context, typically with
                alternating user and assistant messages
        """
        ...

    def chat_log_for(self, target: str) -> List[Message]:
        """
        Retrieve the chat log for the specified target.
        
        This method should filter the conversation history for messages involving the
        specified target and return them as a list of Message objects.
        
        Args:
            target: The target user or entity for which to retrieve the chat log
        
        Returns:
            List[Message]: List of Message objects representing the chat log,
                          typically ordered from oldest to newest
        """
        ...
