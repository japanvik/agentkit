"""Base class for memory implementations."""

from abc import ABC, abstractmethod
from typing import List
from networkkit.messages import Message, MessageType

class BaseMemory(ABC):
    """Abstract base class for memory implementations."""
    
    def __init__(self, max_history_length: int = 10):
        """
        Constructor for the BaseMemory class.

        Args:
            max_history_length (int, optional): The maximum number of messages to store in the history. Defaults to 10.
        """
        self.max_history_length = max_history_length
    
    @abstractmethod
    def remember(self, message: Message) -> None:
        """
        Store a message in the conversation history.

        Args:
            message: The message object to be stored
        """
        pass
    
    @abstractmethod
    def get_history(self) -> List[Message]:
        """
        Retrieve the complete conversation history from memory.

        Returns:
            List of message objects representing the conversation history
        """
        pass
    
    @abstractmethod
    def get_chat_context(self, target: str, prefix: str = "", 
                        user_role_name: str = "", 
                        assistant_role_name: str = "") -> str:
        """
        Retrieve chat conversation history with a specific target and format it.

        Args:
            target: The target name to filter the chat history for
            prefix: A prefix to add before each message
            user_role_name: Optional custom name for user messages
            assistant_role_name: Optional custom name for assistant messages

        Returns:
            Formatted string containing the chat context
        """
        pass

    @abstractmethod
    def chat_log_for(self, target: str) -> List[Message]:
        """
        Retrieves the chat log for the specified target.

        Args:
            target: The target user or entity for which to retrieve the chat log

        Returns:
            List of Message objects representing the chat log
        """
        pass
