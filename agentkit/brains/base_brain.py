"""Base class for brain implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional
from agentkit.common.interfaces import ComponentConfig
from agentkit.memory.memory_protocol import Memory
from networkkit.messages import Message, MessageType
import logging

class BaseBrain(ABC):
    """Abstract base class for brain implementations."""
    
    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        memory_manager: Memory,
        system_prompt: str = "",
        user_prompt: str = "",
        api_config: dict = None
    ) -> None:
        """
        Constructor for the BaseBrain class.

        Args:
            name: The name of the agent
            description: A description of the agent
            model: The name of the LLM model to be used
            memory_manager: An instance of a memory component
            system_prompt: The system prompt template
            user_prompt: The user prompt template
            api_config: Configuration for the LLM API
        """
        self.name = name
        self.description = description
        self.model = model
        self.memory_manager = memory_manager
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.api_config = api_config or {}
        self.component_config: Optional[ComponentConfig] = None
        
        logging.info(f"Initialized brain: {self} with memory {self.memory_manager}")
    
    def set_config(self, config: ComponentConfig) -> None:
        """
        Set the component configuration.

        Args:
            config: The component configuration containing agent information and capabilities.
        """
        self.component_config = config
    
    @abstractmethod
    async def handle_chat_message(self, message: Message) -> None:
        """
        Handle incoming chat messages directed to the agent.

        Args:
            message: The received message object
        """
        pass
    
    @abstractmethod
    async def generate_chat_response(self) -> Message:
        """
        Generate a chat message response.

        Returns:
            The generated message object
        """
        pass
    
    def get_context(self) -> str:
        """Get the context from the memory system."""
        return self.memory_manager.get_context()
    
    def create_chat_messages_prompt(self, system_prompt: str) -> List[dict]:
        """
        Generate chat messages prompt for the LLM.

        Args:
            system_prompt: The formatted system prompt.

        Returns:
            List of message dictionaries for the LLM.
        """
        if not self.component_config:
            raise ValueError("No config set - brain operations require configuration")
            
        system_role = "system"
        user_role = "user"
        assistant_role = "assistant"
        
        messages_prompt = []
        messages_prompt.append({"role": system_role, "content": system_prompt})
        
        target = self.component_config.message_sender.attention
        context = self.memory_manager.chat_log_for(target=target)
        
        for c in context:
            m = {
                "content": c.content.strip(), 
                "role": assistant_role if c.source == self.name else user_role
            }
            messages_prompt.append(m)
        
        return messages_prompt
    
    def format_response(self, reply: str) -> Message:
        """
        Format a reply into a Message object.

        Args:
            reply: The generated reply text.

        Returns:
            Message object ready to send.
        """
        if not self.component_config:
            raise ValueError("No config set - brain operations require configuration")
            
        target = self.component_config.message_sender.attention
        return Message(
            source=self.name,
            to=target,
            content=reply,
            message_type=MessageType.CHAT
        )
