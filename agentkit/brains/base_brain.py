"""Base class for brain implementations."""

from abc import ABC, abstractmethod
from agentkit.agents.base_agent import BaseAgent
from agentkit.memory.base_memory import BaseMemory
from networkkit.messages import Message


class BaseBrain(ABC):
    """Abstract base class for brain implementations."""
    
    def __init__(self, name: str, description: str, model: str, memory_manager: BaseMemory, system_prompt: str = "", user_prompt: str = "") -> None:
        """
        Constructor for the BaseBrain class.

        Args:
            name: The name of the agent
            description: A description of the agent
            model: The name of the LLM model to be used
            memory_manager: An instance of a memory component
            system_prompt: The system prompt template
            user_prompt: The user prompt template
        """
        self.name = name
        self.description = description
        self.model = model
        self.memory_manager = memory_manager
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
    
    @abstractmethod
    async def handle_chat_message(self, agent: BaseAgent, message: Message):
        """
        Handle incoming chat messages directed to the agent.

        Args:
            agent: The agent object for which the message is received
            message: The received message object
        """
        pass
    
    @abstractmethod
    async def create_completion_message(self, agent: BaseAgent) -> Message:
        """
        Generate a chat message response.

        Args:
            agent: The agent object for which to generate a message

        Returns:
            The generated message object
        """
        pass
