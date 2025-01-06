"""Base agent implementation."""
from typing import Optional, Dict, Any, List, Callable, Awaitable
from networkkit.messages import Message, MessageType
from agentkit.common.interfaces import MessageSender, ComponentConfig
from agentkit.memory.memory_protocol import Memory
from agentkit.brains.simple_brain import SimpleBrain
import logging

class BaseAgent(MessageSender):
    """
    Base agent implementation that handles message sending and attention management.
    
    Implements the MessageSender interface to provide communication capabilities
    to other components through ComponentConfig.
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
        
        Args:
            name: Agent's name
            config: Configuration dictionary
            brain: Optional brain component
            memory: Optional memory component
            message_sender: Optional message sender for delegating communication
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
        """Get the brain component."""
        return self._brain
    
    @brain.setter
    def brain(self, value: Optional[SimpleBrain]) -> None:
        """Set the brain component and its config."""
        self._brain = value
        if value is not None and hasattr(value, 'set_config'):
            value.set_config(self.component_config)
    
    @property
    def memory(self) -> Optional[Memory]:
        """Get the memory component."""
        return self._memory
    
    @memory.setter
    def memory(self, value: Optional[Memory]) -> None:
        """Set the memory component and its config."""
        self._memory = value
        if value is not None and hasattr(value, 'set_config'):
            value.set_config(self.component_config)
    
    @property
    def attention(self) -> Optional[str]:
        """Get the current attention target."""
        return self._attention
    
    @attention.setter
    def attention(self, value: str) -> None:
        """Set the current attention target."""
        self._attention = value
    
    async def send_message(self, message: Message) -> None:
        """
        Send a message, delegating to message_sender if provided.
        
        Args:
            message: Message to send
        """
        if self._message_sender:
            await self._message_sender.send_message(message)
    
    def is_intended_for_me(self, message: Message) -> bool:
        """
        Check if a message is intended for this agent.
        
        Args:
            message: Message to check
            
        Returns:
            bool: True if message is intended for this agent, False otherwise
        """
        # Check if message is addressed to this agent or a broadcast
        if message.to in [self.name, 'ALL']:
            return True
            
        # Check if this agent is currently paying attention to the sender
        if self._attention and message.source == self._attention:
            return True
            
        return False
    
    async def handle_message(self, message: Message) -> None:
        """
        Handle an incoming message based on its type.
        
        Args:
            message: Message to handle
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
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def start(self) -> None:
        """Start the agent's background tasks."""
        if self._running:
            return
            
        self._running = True
        logging.info(f"Agent {self.name} started")
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        self._running = False
        logging.info(f"Agent {self.name} stopped")
