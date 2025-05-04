"""
Built-in tools for agents.

This module provides built-in tools that can be registered with the functions registry.
"""

import uuid
from typing import Dict, Any

from networkkit.messages import Message, MessageType


async def send_message_tool(agent, recipient: str, content: str, message_type: str = "CHAT") -> Dict[str, Any]:
    """
    Send a message to a specified recipient.
    
    Args:
        agent: The agent sending the message.
        recipient: The name of the recipient.
        content: The message content.
        message_type: The type of message (default: "CHAT").
        
    Returns:
        A dictionary with information about the sent message.
    """
    # Create a message
    message = Message(
        source=agent.name,
        to=recipient,
        content=content,
        message_type=MessageType(message_type)
    )
    
    # Use the agent's internal send_message implementation
    await agent._internal_send_message(message)
    
    # Return information about the sent message
    return {
        "status": "sent",
        "message_id": str(uuid.uuid4()),
        "recipient": recipient,
        "message_type": message_type
    }
