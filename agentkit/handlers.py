# agentkit/handlers.py

from datetime import datetime
import logging

from agentkit.io.console import async_print
from networkkit.messages import Message, MessageType


async def handle_helo_message(agent, message: Message):
    """
    Handles incoming HELO messages by recording the sender and responding with an ACK.
    """
    try:
        sender = message.source
        description = message.content
        agent.available_agents[sender] = {
            "description": description,
            "last_seen": datetime.now()
        }

        # Assign a color if not already assigned
        if sender not in agent.agent_colors:
            agent.agent_colors[sender] = next(agent.color_cycle)

        logging.info(f"Received HELO from {sender}. Sending ACK.")

        # Respond with ACK
        ack_message = Message(
            source=agent.name,
            to=sender,
            content=agent.description,
            message_type=MessageType.ACK
        )
        await agent.send_message(ack_message)
    except Exception as e:
        logging.error(f"Error handling HELO message from {message.source}: {e}")

async def handle_ack_message(agent, message: Message):
    """
    Handles incoming ACK messages by recording the sender.
    """
    try:
        sender = message.source
        description = message.content
        agent.available_agents[sender] = {
            "description": description,
            "last_seen": datetime.now()
        }

        # Assign a color if not already assigned
        if sender not in agent.agent_colors:
            agent.agent_colors[sender] = next(agent.color_cycle)

        logging.info(f"Received ACK from {sender}.")
        logging.info(f"Available agents: {agent.available_agents}")
    except Exception as e:
        logging.error(f"Error handling ACK message from {message.source}: {e}")

async def print_chat_message(agent, message: Message) -> None:
    """
    Simple handler for CHAT messages that prints the message content.
    """
    agent.attention = message.source
    color = agent.agent_colors[message.source]
    await async_print(f"[bold {color}]{message.source}[/bold {color}]: @{message.to} {message.content}")
