from agentkit.messages import Message


class SimpleMemory:
    """
    Simple in-memory implementation of the Memory protocol for agent conversation history.

    This class provides basic functionality for storing and retrieving conversation history in memory.
    It adheres to the `Memory` protocol defined in `memory_protocol.py`.
    """

    def __init__(self, max_history_length: int = 10) -> None:
        """
        Constructor for the SimpleMemory class.

        Args:
            max_history_length (int, optional): The maximum number of messages to store in the history. Defaults to 10.
        """

        self.history = []
        self.max_history_length = max_history_length

    def remember(self, message: Message) -> None:
        """
        Store a message in the conversation history.

        This method implements the `remember` method from the `Memory` protocol.
        It adds the provided message object to the internal history list, maintaining the maximum history length.
        If the history reaches its limit, the oldest message is removed before adding the new one.

        Args:
            message: The message object to be stored (type: agentkit.messages.Message)
        """

        if len(self.history) >= self.max_history_length:
            self.history.pop(0)  # Remove oldest message to maintain limit
        self.history.append(message)

    def get_history(self) -> list[Message]:
        """
        Retrieve the complete conversation history from memory.

        This method implements the `get_history` method from the `Memory` protocol.
        It returns a list containing all the message objects stored in the history.

        Returns:
            list[Message]: A list of message objects (type: agentkit.messages.Message) representing the conversation history.
        """

        return self.history

    def get_chat_context(self, target: str, prefix: str = "") -> str:
        """
        Retrieve chat conversation history with a specific target (source or recipient) and format it with a prefix.

        This method filters the complete history retrieved by `get_history` and returns only messages
        where the source or recipient matches the provided `target` and the message type is `CHAT`.
        It then formats the filtered messages with the specified `prefix` before joining them into a string.

        Args:
            target (str): The target name (source or recipient) to filter the chat history for.
            prefix (str, optional): A prefix to add before each message in the returned context string. Defaults to "".

        Returns:
            str: A formatted string containing the chat context for the specified target, including prefixes.
        """

        chat_log = [x for x in self.get_history() if (target in [x.to, x.source]) and x.message_type == MessageType.CHAT]
        context = "\n".join(f"{prefix}{x.source}: {x.content.strip()}" for x in chat_log)
        return context
