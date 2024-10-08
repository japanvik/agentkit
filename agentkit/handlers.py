# agentkit/handlers.py

from agentkit.agents.base_agent import BaseAgent
from networkkit.messages import Message, MessageType


async def default_handle_helo_message(agent: BaseAgent, message: Message) -> None:
    """
    Default handler for HELO messages.
    """
    agent.attention = message.source
    response = Message(
        source=agent.name,
        to=message.source,
        content=agent.description,
        message_type=MessageType.ACK
    )
    agent.send_message(response)


async def print_chat_message(agent, message: Message) -> None:
    """
    Simple handler for CHAT messages that prints the message content.
    """
    agent.attention = message.source
    print(f"\n## {message.source} to {message.to}[{message.message_type}]: {message.content}")
