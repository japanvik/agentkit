"""Simple brain implementation."""
from agentkit.brains.base_brain import BaseBrain
from agentkit.memory.memory_protocol import Memory
from agentkit.processor import llm_chat
from networkkit.messages import Message
import logging

class SimpleBrain(BaseBrain):
    """
    Simple rule-based brain for agent interaction, utilizing LLM for message generation.

    This class serves as the core decision-making component for an agent. It manages conversation flow, relies on memory for
    context, and interacts with an LLM to generate responses to chat messages.

    The brain operates based on pre-defined system and user prompts that can be formatted with placeholders like agent name,
    description, context, and target recipient. These prompts are used to guide the LLM in response generation.
    """

    async def handle_chat_message(self, message: Message) -> None:
        """
        Handle incoming chat messages directed to the agent.

        This method is called whenever a chat message is received by the agent.
        It performs the following actions:

        1. Stores the received message in the memory manager.
        2. Checks if the message source is different from the agent itself (i.e., an incoming message).
        3. If it's an incoming message, updates the agent's attention to the message source.
        4. Generates a response using the LLM.
        5. Sends the generated response message through the message sender.

        Args:
            message (Message): The received message object.
        """
        if not self.component_config:
            logging.error("No config set - brain operations require configuration")
            return

        self.memory_manager.remember(message)
        
        if message.source != self.name:
            # Reply to a chat message from someone
            self.component_config.message_sender.attention = message.source
            response = await self.generate_chat_response()
            await self.component_config.message_sender.send_message(response)

    async def generate_chat_response(self) -> Message:
        """
        Generate a chat response based on the current context.
        
        Returns:
            Message: The generated response message.
        """
        # Format the system prompt
        context = self.get_context()
        target = self.component_config.message_sender.attention
        system_prompt = self.system_prompt.format(
            name=self.name, 
            description=self.description, 
            context=context, 
            target=target
        )
        
        messages = self.create_chat_messages_prompt(system_prompt)
        
        # Extract API configuration
        api_base = self.api_config.get('api_base')
        api_key = self.api_config.get('api_key')
        
        reply = await llm_chat(
            llm_model=self.model,
            messages=messages,
            api_base=api_base,
            api_key=api_key
        )

        return self.format_response(reply)
