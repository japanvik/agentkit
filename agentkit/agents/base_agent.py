"""
Base agent implementation.

This module provides the BaseAgent class, which serves as the foundation for all agent
implementations in the AgentKit framework. It handles message sending, attention management,
and component configuration.

The BaseAgent class implements the MessageSender interface to provide communication
capabilities to other components through ComponentConfig.
"""
# Standard library imports
import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Third-party imports
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.common.interfaces import ComponentConfig, MessageSender
from agentkit.memory.memory_protocol import Memory

class BaseAgent(MessageSender):
    """
    Base agent implementation that handles message sending and attention management.
    
    The BaseAgent class provides core functionality for all agent types in the AgentKit
    framework. It manages message handling, component configuration, and attention
    targeting. It serves as a foundation that can be extended to create specialized
    agent implementations.
    
    Implements the MessageSender interface to provide communication capabilities
    to other components through ComponentConfig.
    
    Attributes:
        name (str): The agent's name, used for message addressing
        config (Dict[str, Any]): Configuration dictionary for the agent
        component_config (ComponentConfig): Configuration shared with components
        message_handlers (Dict[MessageType, List[Callable]]): Registered message handlers
        _brain (Optional[SimpleBrain]): The agent's brain component for decision making
        _memory (Optional[Memory]): The agent's memory component for storing history
        _attention (Optional[str]): The current attention target
        _message_sender (Optional[MessageSender]): External message sender if provided
        _running (bool): Flag indicating if the agent is currently running
        _tasks (List[Awaitable]): List of background tasks
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        brain: Optional[SimpleBrain] = None,
        memory: Optional[Memory] = None,
        message_sender: Optional[MessageSender] = None
    ) -> None:
        """
        Initialize the agent with name, configuration, and optional components.
        
        This constructor sets up the agent with its basic configuration and optional
        components like brain and memory. It also initializes the internal state
        and creates a ComponentConfig object to share with components.
        
        Args:
            name: Agent's name used for message addressing
            config: Configuration dictionary containing agent settings
            brain: Optional brain component for decision making
            memory: Optional memory component for storing conversation history
            message_sender: Optional message sender for delegating communication
                           (if None, the agent will handle sending itself)
        """
        # Store basic info
        self.name = name
        self.config = config
        
        # Initialize internal state
        self._attention: Optional[str] = None
        self._message_sender = message_sender
        self._running = False
        self._tasks: List[Awaitable] = []
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        
        # Create component config with self as MessageSender
        self.component_config = ComponentConfig(
            agent_name=name,
            config=config,
            message_sender=self
        )
        
        # Store components
        self._brain = brain
        self._memory = memory
        
        # Set config on components if they exist
        if hasattr(self._brain, 'set_config'):
            self._brain.set_config(self.component_config)
        
        if hasattr(self._memory, 'set_config'):
            self._memory.set_config(self.component_config)
    
    @property
    def brain(self) -> Optional[SimpleBrain]:
        """
        Get the brain component.
        
        The brain component is responsible for decision making and generating
        responses to messages.
        
        Returns:
            Optional[SimpleBrain]: The current brain component or None if not set
        """
        return self._brain
    
    @brain.setter
    def brain(self, value: Optional[SimpleBrain]) -> None:
        """
        Set the brain component and its config.
        
        This setter not only assigns the brain component but also configures it
        with the agent's ComponentConfig if the brain has a set_config method.
        
        Args:
            value: The brain component to set or None to clear it
        """
        self._brain = value
        if value is not None and hasattr(value, 'set_config'):
            value.set_config(self.component_config)
    
    @property
    def memory(self) -> Optional[Memory]:
        """
        Get the memory component.
        
        The memory component is responsible for storing and retrieving
        conversation history.
        
        Returns:
            Optional[Memory]: The current memory component or None if not set
        """
        return self._memory
    
    @memory.setter
    def memory(self, value: Optional[Memory]) -> None:
        """
        Set the memory component and its config.
        
        This setter not only assigns the memory component but also configures it
        with the agent's ComponentConfig if the memory has a set_config method.
        
        Args:
            value: The memory component to set or None to clear it
        """
        self._memory = value
        if value is not None and hasattr(value, 'set_config'):
            value.set_config(self.component_config)
    
    @property
    def attention(self) -> Optional[str]:
        """
        Get the current attention target.
        
        The attention target represents the entity this agent is currently focused on
        for communication purposes. Messages from this target will be processed
        even if they are not explicitly addressed to this agent.
        
        Returns:
            Optional[str]: The name of the current attention target or None if not set
        """
        return self._attention
    
    @attention.setter
    def attention(self, value: str) -> None:
        """
        Set the current attention target.
        
        Updates the agent's focus to a specific entity for communication purposes.
        Messages from this entity will be processed even if they are not explicitly
        addressed to this agent.
        
        Args:
            value: The name of the entity to focus attention on
        """
        self._attention = value
    
    async def _internal_send_message(self, message: Message) -> None:
        """
        Internal implementation of message sending, delegating to message_sender if provided.
        
        This method handles the actual message sending, either by delegating to an external
        message sender (if provided during initialization) or by implementing
        the sending logic in subclasses. If no message_sender was provided and
        this method is not overridden in a subclass, the message will not be sent.
        
        Args:
            message: Message object to send
        """
        if self._message_sender:
            await self._message_sender.send_message(message)
    
    async def send_message(self, message: Message) -> None:
        """
        Send a message, using the send_message tool if available, otherwise falling back to internal implementation.
        
        This method provides backward compatibility with existing code that calls send_message directly.
        In the new architecture, sending messages should be done through the functions registry using
        the send_message tool.
        
        Args:
            message: Message object to send
        """
        # For backward compatibility, use the internal implementation
        await self._internal_send_message(message)
    
    def register_tools(self, functions_registry) -> None:
        """
        Register agent tools with the functions registry.
        
        This method registers built-in tools like send_message with the functions registry,
        making them available for use by the agent and its components.
        
        Args:
            functions_registry: The functions registry to register tools with
        """
        from functools import partial
        from agentkit.functions.built_in_tools import send_message_tool
        from agentkit.functions.functions_registry import FunctionDescriptor, ParameterDescriptor
        
        # Register send_message tool
        send_message_fn = partial(send_message_tool, self)
        
        descriptor = FunctionDescriptor(
            name="send_message",
            description="Send a message to another agent or entity",
            parameters=[
                ParameterDescriptor(
                    name="recipient",
                    description="The name of the recipient",
                    required=True
                ),
                ParameterDescriptor(
                    name="content",
                    description="The message content",
                    required=True
                ),
                ParameterDescriptor(
                    name="message_type",
                    description="The type of message (default: CHAT)",
                    required=False
                )
            ]
        )
        
        functions_registry.register_function(send_message_fn, descriptor)
    
    def is_intended_for_me(self, message: Message) -> bool:
        """
        Check if a message is intended for this agent.
        
        A message is considered intended for this agent if any of the following are true:
        1. The message is explicitly addressed to this agent (message.to == self.name)
        2. The message is a broadcast message (message.to == 'ALL')
        3. The message was sent by this agent (message.source == self.name)
        4. The agent is currently paying attention to the sender (self._attention == message.source)
        
        Args:
            message: Message object to check
            
        Returns:
            bool: True if message is intended for this agent, False otherwise
        """
        # Check if message is addressed to this agent or a broadcast
        if message.to in [self.name, 'ALL'] or message.source == self.name:
            return True
            
        # Check if this agent is currently paying attention to the sender
        if self._attention and message.source == self._attention:
            return True
            
        return False
    
    async def handle_message(self, message: Message) -> None:
        """
        Handle an incoming message based on its type.
        
        This method first checks if the message is intended for this agent.
        If it is, it looks for registered handlers for the message type.
        If handlers are found, they are called in order.
        If no handlers are found, default handling is applied based on the message type.
        
        For CHAT messages, the brain component is used to handle the message.
        For HELO messages, an ACK response is sent back to the sender.
        
        Args:
            message: Message object to handle
        
        Raises:
            Exception: Any exception raised by message handlers is caught and logged
        """
        # Only handle messages intended for this agent
        if not self.is_intended_for_me(message):
            return
            
        # Check for registered handlers first
        handlers = self.message_handlers.get(message.message_type, [])
        if handlers:
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logging.error(f"Error in message handler: {e}")
            return

        # Default handling if no handlers registered
        if message.message_type == MessageType.CHAT:
            if self.brain:
                await self.brain.handle_chat_message(message)
        elif message.message_type == MessageType.HELO:
            # Respond with ACK
            response = Message(
                source=self.name,
                to=message.source,
                content="",
                message_type=MessageType.ACK
            )
            await self.send_message(response)
    
    def register_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        This method allows for custom handling of specific message types.
        Multiple handlers can be registered for the same message type,
        and they will be called in the order they were registered.
        
        Args:
            message_type: Type of message to handle (from MessageType enum)
            handler: Async function to handle the message, with signature:
                    async def handler(message: Message) -> None
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def start(self) -> None:
        """
        Start the agent's background tasks.
        
        This method initializes the agent and prepares it for operation.
        If the agent is already running, this method does nothing.
        Subclasses should override this method to start any additional
        background tasks or services.
        """
        if self._running:
            return
            
        self._running = True
        logging.info(f"Agent {self.name} started")
    
    async def stop(self) -> None:
        """
        Stop the agent and cleanup resources.
        
        This method stops the agent's operation and performs necessary cleanup:
        1. Sets running flag to False
        2. Cancels all registered background tasks
        3. Closes any open resources (e.g., client sessions)
        4. Logs completion
        
        Subclasses should call super().stop() when overriding this method.
        """
        if not self._running:
            return
            
        self._running = False
        
        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close any resources
        if hasattr(self, '_client_session') and self._client_session:
            await self._client_session.close()
            
        # Close message sender if it has a close method
        if self._message_sender and hasattr(self._message_sender, 'close'):
            try:
                await self._message_sender.close()
                logging.info(f"Closed message sender for agent {self.name}")
            except Exception as e:
                logging.error(f"Error closing message sender: {e}")
        
        logging.info(f"Agent {self.name} stopped")
