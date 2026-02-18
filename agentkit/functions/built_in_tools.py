"""Built-in tools for agents."""

import logging
import uuid
from typing import Any, Dict, Optional

from networkkit.messages import Message, MessageType
from networkkit.network import HTTPMessageSender

from agentkit.functions.functions_registry import ToolExecutionContext

logger = logging.getLogger(__name__)


def _prefer_mcp_send_message(agent_config: Dict[str, Any]) -> bool:
    backend = str(agent_config.get("send_message_backend", "")).strip().lower()
    if backend:
        return backend == "mcp"
    return bool(agent_config.get("use_mcp_send_message"))


def _select_mcp_send_function(agent_config: Dict[str, Any], registry: Any) -> Optional[str]:
    explicit_name = agent_config.get("mcp_send_message_function")
    if isinstance(explicit_name, str) and explicit_name.strip():
        return explicit_name.strip()

    namespace = str(agent_config.get("mcp_send_message_namespace", "")).strip()
    if namespace:
        candidate = f"{namespace}::send_message"
        if registry.has_function(candidate):
            return candidate

    function_map = getattr(registry, "function_map", {})
    names = list(function_map.keys()) if isinstance(function_map, dict) else []
    if "networkkit::send_message" in names:
        return "networkkit::send_message"

    send_candidates = sorted(name for name in names if name.endswith("::send_message"))
    if len(send_candidates) == 1:
        return send_candidates[0]
    if len(send_candidates) > 1:
        logger.warning(
            "Multiple MCP send_message tools discovered (%s); "
            "set config['mcp_send_message_function'] to select one.",
            ", ".join(send_candidates),
        )
    return None


def _normalize_mcp_send_response(
    raw_result: Any,
    *,
    recipient: str,
    message_type: str,
) -> Dict[str, Any]:
    if isinstance(raw_result, dict):
        if {"status", "recipient", "message_type"}.issubset(raw_result.keys()):
            return raw_result
        raw_payload = raw_result.get("raw")
        if isinstance(raw_payload, dict):
            payload = raw_payload.copy()
            payload.setdefault("recipient", recipient)
            payload.setdefault("message_type", message_type)
            payload.setdefault("message_id", str(uuid.uuid4()))
            payload.setdefault("status", "sent")
            return payload

    return {
        "status": "sent",
        "message_id": str(uuid.uuid4()),
        "recipient": recipient,
        "message_type": message_type,
    }


async def _send_message_via_mcp(
    context: ToolExecutionContext,
    *,
    recipient: str,
    content: str,
    message_type: str,
) -> Optional[Dict[str, Any]]:
    agent = context.agent
    registry = getattr(agent, "functions_registry", None)
    if registry is None or not hasattr(registry, "has_function"):
        return None

    agent_config = getattr(agent, "config", {})
    if not isinstance(agent_config, dict) or not _prefer_mcp_send_message(agent_config):
        return None

    mcp_function = _select_mcp_send_function(agent_config, registry)
    if not mcp_function or mcp_function == "send_message":
        return None
    if not registry.has_function(mcp_function):
        logger.warning("Configured MCP send tool '%s' is not registered", mcp_function)
        return None

    try:
        result = await registry.execute(
            function=mcp_function,
            parameters={
                "recipient": recipient,
                "content": content,
                "message_type": message_type,
            },
            context=context,
        )
    except Exception:
        logger.exception("MCP send_message via '%s' failed; falling back to direct send", mcp_function)
        return None

    return _normalize_mcp_send_response(
        result,
        recipient=recipient,
        message_type=message_type,
    )


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

    mcp_result = await _send_message_via_mcp(
        context,
        recipient=recipient,
        content=content,
        message_type=message_type,
    )
    if mcp_result is not None:
        return mcp_result

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
        "message_type": message_type,
    }
