"""
Base class for memory implementations.

This module provides the BaseMemory abstract base class, which defines the interface
and common functionality for all memory implementations in the AgentKit framework.
Memory components are responsible for storing conversation history, retrieving relevant
context, and managing the agent's memory of past interactions.

The BaseMemory class works in conjunction with Brain components to provide conversation
context for generating contextually relevant responses.
"""
# Standard library imports
from abc import ABC, abstractmethod
from typing import List, Optional

# Third-party imports
from networkkit.messages import Message, MessageType

class BaseMemory(ABC):
    """
    Abstract base class for memory implementations.
    
    The BaseMemory class defines the interface and common functionality for all memory
    implementations in the AgentKit framework. Memory components are responsible for
    storing conversation history, retrieving relevant context, and managing the
    agent's memory of past interactions.
    
    This class provides a foundation for implementing different memory strategies,
    such as simple in-memory storage, vector databases, or other persistence mechanisms.
    Concrete implementations must provide methods for storing and retrieving messages.
    
    Attributes:
        max_history_length (int): The maximum number of messages to store in the history
    """
    
    def __init__(self, max_history_length: int = 10):
        """
        Initialize the memory with a maximum history length.
        
        This constructor sets up the memory with its basic configuration.
        The max_history_length parameter controls how many messages are
        retained in the conversation history.
        
        Args:
            max_history_length: The maximum number of messages to store in the history.
                               Defaults to 10. Set to 0 or negative for unlimited history.
        """
        self.max_history_length = max_history_length
    
    @abstractmethod
    def remember(self, message: Message) -> None:
        """
        Store a message in the conversation history.
        
        This abstract method must be implemented by concrete memory classes.
        It should add the provided message to the conversation history,
        respecting the max_history_length limit if applicable.
        
        Args:
            message: The message object to be stored, containing source,
                   target, content, and other metadata
        """
        pass
    
    @abstractmethod
    def get_history(self) -> List[Message]:
        """
        Retrieve the complete conversation history from memory.
        
        This abstract method must be implemented by concrete memory classes.
        It should return all stored messages, potentially limited by
        max_history_length.
        
        Returns:
            List[Message]: List of message objects representing the conversation history,
                          typically ordered from oldest to newest
        """
        pass
    
    @abstractmethod
    def get_chat_context(self, target: str, prefix: str = "", 
                        user_role_name: str = "", 
                        assistant_role_name: str = "") -> str:
        """
        Retrieve chat conversation history with a specific target and format it.
        
        This abstract method must be implemented by concrete memory classes.
        It should filter the conversation history for messages involving the
        specified target and format them as a string for use in prompts or
        display.
        
        Args:
            target: The target name to filter the chat history for (e.g., user ID)
            prefix: A prefix to add before each message in the formatted output
            user_role_name: Optional custom name for user messages (e.g., "User")
            assistant_role_name: Optional custom name for assistant messages (e.g., "Assistant")
        
        Returns:
            str: Formatted string containing the chat context, typically with
                alternating user and assistant messages
        """
        pass

    @abstractmethod
    def chat_log_for(self, target: str) -> List[Message]:
        """
        Retrieve the chat log for the specified target.
        
        This abstract method must be implemented by concrete memory classes.
        It should filter the conversation history for messages involving the
        specified target and return them as a list of Message objects.
        
        Args:
            target: The target user or entity for which to retrieve the chat log
        
        Returns:
            List[Message]: List of Message objects representing the chat log,
                          typically ordered from oldest to newest
        """
        pass
