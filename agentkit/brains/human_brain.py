"""Human brain implementation."""
from typing import Optional
from agentkit.brains.base_brain import BaseBrain
from agentkit.memory.memory_protocol import Memory
from agentkit.common.interfaces import ComponentConfig
from networkkit.messages import Message
import logging

class HumanBrain(BaseBrain):
    """
    HumanBrain is a minimal brain implementation for human agents.
    
    Most of the parameters and methods are just for interface consistency,
    as human agents don't need LLM or memory functionality.
    """

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
        Initialize the human brain.
        
        Args:
            name: The name of the agent
            description: A description of the agent
            model: Not used by HumanBrain but included for interface consistency
            memory_manager: Not used by HumanBrain but included for interface consistency
            system_prompt: Not used by HumanBrain but included for interface consistency
            user_prompt: Not used by HumanBrain but included for interface consistency
            api_config: Not used by HumanBrain but included for interface consistency
        """
        super().__init__(
            name=name,
            description=description,
            model=model,
            memory_manager=memory_manager,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_config=api_config
        )
        logging.info(f"Initialized human brain for {name}")

    async def handle_chat_message(self, message: Message) -> None:
        """
        Handle incoming chat messages. For human agents, this is handled at the agent level.

        Args:
            message: The received message object
        """
        # Human agents handle chat messages directly, no brain processing needed
        pass

    async def generate_chat_response(self) -> Message:
        """
        Generate a chat response. For human agents, responses come from user input.

        Returns:
            Message: Empty message as human responses are handled elsewhere
        """
        # Human agents generate responses through user input, not the brain
        return Message(
            source=self.name,
            to="",
            content="",
            message_type="CHAT"
        )
