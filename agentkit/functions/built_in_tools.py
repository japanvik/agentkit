"""
Built-in tools for agents.

This module provides built-in tools that can be registered with the functions registry.
"""

import uuid
from typing import Dict, Any

from networkkit.messages import Message, MessageType
from networkkit.network import HTTPMessageSender


from agentkit.functions.functions_registry import ToolExecutionContext


async def send_message_tool(
    context: ToolExecutionContext,
    recipient: str,
    content: str,
    message_type: str = "CHAT",
) -> Dict[str, Any]:
    """
    Send a message to a specified recipient.
    
    Args:
        context: Execution context providing access to the calling agent.
        recipient: The name of the recipient.
        content: The message content.
        message_type: The type of message (default: "CHAT").
        
    Returns:
        A dictionary with information about the sent message.
    """
    agent = context.agent
    message_sender = getattr(agent, "_message_sender", None)

    if message_sender is None:
        try:
            from agentkit.agents.base_agent import BaseAgent  # local import to avoid circular
        except ImportError:  # pragma: no cover
            BaseAgent = None  # type: ignore

        if BaseAgent is not None and not isinstance(agent, BaseAgent):
            BaseAgent = None

        agent_config = getattr(agent, "config", {})  # type: ignore[attr-defined]
        if not isinstance(agent_config, dict):
            agent_config = {}

        bus_ip = agent_config.get("bus_ip", "127.0.0.1")
        publish_address = agent_config.get("bus_publish_address", f"http://{bus_ip}:8000")
        if BaseAgent is not None:
            message_sender = HTTPMessageSender(publish_address=publish_address)
            setattr(agent, "_message_sender", message_sender)
            # ensure component config reflects updated sender if present
            if getattr(agent, "component_config", None):
                agent.component_config.message_sender = agent
        else:
            message_sender = None

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
