"""
Base class for brain implementations.

This module provides the BaseBrain abstract base class, which defines the interface
and common functionality for all brain implementations in the AgentKit framework.
Brain components are responsible for decision making, generating responses to messages,
and interacting with LLM models.

The BaseBrain class works in conjunction with Memory components to maintain conversation
context and generate contextually relevant responses.
"""
# Standard library imports
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Third-party imports
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.common.interfaces import ComponentConfig
from agentkit.memory.memory_protocol import Memory

class BaseBrain(ABC):
    """
    Abstract base class for brain implementations.
    
    The BaseBrain class defines the interface and common functionality for all brain
    implementations in the AgentKit framework. Brain components are responsible for
    decision making, generating responses to messages, and interacting with LLM models.
    
    This class provides utility methods for formatting prompts, managing context,
    and creating response messages. Concrete implementations must provide the
    handle_chat_message and generate_chat_response methods.
    
    Attributes:
        name (str): The name of the agent this brain belongs to
        description (str): A description of the agent's purpose or capabilities
        model (str): The name of the LLM model to be used
        memory_manager (Memory): The memory component for storing conversation history
        system_prompt (str): The system prompt template for the LLM
        user_prompt (str): The user prompt template for the LLM
        api_config (dict): Configuration for the LLM API
        component_config (Optional[ComponentConfig]): Configuration shared with components
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        memory_manager: Memory,
        system_prompt: str = "",
        user_prompt: str = "",
        api_config: Dict[str, Any] = None
    ) -> None:
        """
        Initialize the brain with name, description, model, and memory manager.
        
        This constructor sets up the brain with its basic configuration and
        required components. It initializes the internal state and prepares
        the brain for handling messages and generating responses.
        
        Args:
            name: The name of the agent this brain belongs to
            description: A description of the agent's purpose or capabilities
            model: The name of the LLM model to be used (e.g., "gpt-4")
            memory_manager: An instance of a memory component for storing conversation history
            system_prompt: The system prompt template for the LLM
            user_prompt: The user prompt template for the LLM
            api_config: Configuration dictionary for the LLM API (e.g., temperature, max_tokens)
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
        
        This method is called by the agent to provide the brain with access to
        agent information and capabilities through the ComponentConfig object.
        The configuration includes the agent's name, configuration dictionary,
        and a message sender interface for sending responses.
        
        Args:
            config: The component configuration containing agent information and capabilities
        """
        self.component_config = config
    
    @abstractmethod
    async def handle_chat_message(self, message: Message) -> None:
        """
        Handle incoming chat messages directed to the agent.
        
        This abstract method must be implemented by concrete brain classes.
        It should process the incoming message, store it in memory if needed,
        and generate a response using the LLM.
        
        Args:
            message: The received message object containing source, content, etc.
        """
        pass
    
    @abstractmethod
    async def generate_chat_response(self) -> Message:
        """
        Generate a chat message response.
        
        This abstract method must be implemented by concrete brain classes.
        It should use the LLM to generate a response based on the conversation
        context and return it as a Message object.
        
        Returns:
            Message: The generated message object ready to be sent
        """
        pass
    
    def get_context(self) -> str:
        """
        Get the context from the memory system.
        
        This method retrieves the conversation context from the memory manager,
        which is used to provide context to the LLM for generating responses.
        
        Returns:
            str: The conversation context as a formatted string
        """
        return self.memory_manager.get_context()
    
    def create_chat_messages_prompt(self, system_prompt: str) -> List[Dict[str, str]]:
        """
        Generate chat messages prompt for the LLM.
        
        This method creates a list of message dictionaries in the format expected
        by most LLM APIs (e.g., OpenAI). It includes a system message with the
        provided system prompt, followed by the conversation history retrieved
        from the memory manager.
        
        Args:
            system_prompt: The formatted system prompt to guide the LLM's behavior
            
        Returns:
            List[Dict[str, str]]: List of message dictionaries for the LLM API,
                                 each with 'role' and 'content' keys
                                 
        Raises:
            ValueError: If component_config is not set
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
        
        This method creates a Message object from the generated reply text,
        setting the appropriate source, target, content, and message type.
        
        Args:
            reply: The generated reply text from the LLM
            
        Returns:
            Message: Message object ready to send
            
        Raises:
            ValueError: If component_config is not set
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
