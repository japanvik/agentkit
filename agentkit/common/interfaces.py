"""
Common interfaces and protocols used across the agent system.

This module defines core interfaces and protocols that are used throughout the
AgentKit framework. These interfaces enable loose coupling between components
and facilitate interchangeability of implementations.

The key interfaces defined here are:
- MessageSender: Interface for components that can send messages
- ComponentConfig: Shared configuration for agent components
"""
# Standard library imports
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

# Third-party imports
from networkkit.messages import Message

class MessageSender(Protocol):
    """
    Interface for components that can send messages and manage attention.
    
    This protocol defines the methods that must be implemented by any component
    that needs to send messages and manage attention targeting. It is typically
    implemented by agent classes but can be used by any component that needs
    these capabilities.
    
    The attention property allows components to track which entity they are
    currently focused on for communication purposes.
    
    The close method allows components to clean up resources when they are no longer needed.
    """
    
    @property
    def attention(self) -> Optional[str]:
        """
        Get the current attention target.
        
        The attention target represents the entity this component is currently
        focused on for communication purposes.
        
        Returns:
            Optional[str]: The name of the current attention target or None if not set
        """
        ...
    
    @attention.setter
    def attention(self, value: str) -> None:
        """
        Set the current attention target.
        
        Updates the component's focus to a specific entity for communication purposes.
        
        Args:
            value: The name of the entity to focus attention on
        """
        ...
    
    async def send_message(self, message: Message) -> None:
        """
        Send a message.
        
        This method is responsible for sending the provided message to its
        intended recipient.
        
        Args:
            message: The message object to send
        """
        ...
    
    async def close(self) -> None:
        """
        Close the message sender and clean up resources.
        
        This method is responsible for properly closing any resources used by the
        message sender, such as network connections or client sessions.
        """
        ...

@dataclass
class ComponentConfig:
    """
    Shared configuration for agent components.
    
    This dataclass provides access to agent information and capabilities without creating
    direct dependencies between components. It is passed to components like brains and
    memories during initialization or configuration, allowing them to access agent
    properties and send messages through the agent.
    
    Attributes:
        agent_name (str): Name of the agent this component belongs to
        config (Dict[str, Any]): Agent configuration dictionary containing settings
        message_sender (MessageSender): Interface for sending messages and managing attention
        functions_registry (Optional): Registry of functions/tools available to the agent
    """
    agent_name: str
    config: Dict[str, Any]
    message_sender: MessageSender
    functions_registry: Optional[Any] = None
