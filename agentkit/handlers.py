from networkkit.messages import Message, MessageType


async def default_handle_helo_message(agent, message: Message) -> None:
    """
    Default handler for HELO messages.

    1. Updates the agent's attention to the source of the HELO message.
    2. Creates an acknowledgement (ACK) message with the agent's description as content.
    3. Sets the target of the response message to the original HELO message source.
    4. Sends the acknowledgement message back to the sender using the agent's `send_message` method.

    Args:
        agent: The agent instance that received the message.
        message (Message): The received message object.
    """

    agent.attention = message.source
    response = Message(source=agent.name, to=message.source, content=agent.description, message_type=MessageType.ACK)
    agent.send_message(response)


async def print_chat_message(agent, message: Message) -> None:
    """
    Simple handler for CHAT messages that prints the message content.

    This function handles messages of type `MessageType.CHAT` by printing the message content
    in a formatted way, including the message source, target, message type, and the actual content.

    Args:
        agent: The agent instance that received the message.
        message (Message): The received message object.
    """

    agent.attention = message.source
    print(f"\n## {message.source} to {message.to}[{message.message_type}]: {message.content}")
