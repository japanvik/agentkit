"""
Common interfaces and protocols used across the agent system.
"""
from typing import Protocol, Optional, Dict, Any
from dataclasses import dataclass
from networkkit.messages import Message

class MessageSender(Protocol):
    """Interface for components that can send messages and manage attention."""
    
    @property
    def attention(self) -> Optional[str]:
        """Current attention target."""
        ...
    
    @attention.setter
    def attention(self, value: str) -> None:
        """Set the current attention target."""
        ...
    
    async def send_message(self, message: Message) -> None:
        """Send a message."""
        ...

@dataclass
class ComponentConfig:
    """
    Shared configuration for agent components.
    
    Provides access to agent information and capabilities without creating
    direct dependencies between components.
    
    Attributes:
        agent_name: Name of the agent
        config: Agent configuration
        message_sender: Interface for sending messages and managing attention
    """
    agent_name: str
    config: Dict[str, Any]
    message_sender: MessageSender
