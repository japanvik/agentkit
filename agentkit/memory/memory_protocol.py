from typing import List, Protocol
from networkkit.messages import Message


class Memory(Protocol):
    """
    Interface defining a memory storage for agent conversations.

    This protocol outlines the expected behavior of memory components for agents. Concrete implementations should provide methods
    for storing messages (`remember`) and retrieving the conversation history (`get_history`).
    """

    def remember(self, message: Message) -> None:
        """
        Method to store a message in the memory.

        This method is responsible for storing the provided message object (containing details like source, recipient, content, and type)
        in the agent's memory.

        Args:
            message: The message object to be stored (type: agentkit.messages.Message)
        """

        raise NotImplementedError

    def get_history(self) -> List[Message]:
        """
        Method to retrieve the conversation history from the memory.

        This method should return a list of message objects representing the stored conversation history.

        Returns:
            List[Message]: A list of message objects (type: agentkit.messages.Message) representing the conversation history.
        """

        raise NotImplementedError
