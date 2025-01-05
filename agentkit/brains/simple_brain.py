from agentkit.agents.base_agent import BaseAgent
from agentkit.memory.memory_protocol import Memory
from agentkit.processor import llm_chat
from networkkit.messages import Message, MessageType
import logging


class SimpleBrain:
    """
    Simple rule-based brain for agent interaction, utilizing LLM for message generation.

    This class serves as the core decision-making component for an agent. It manages conversation flow, relies on memory for
    context, and interacts with an LLM to generate responses to chat messages.

    The brain operates based on pre-defined system and user prompts that can be formatted with placeholders like agent name,
    description, context, and target recipient. These prompts are used to guide the LLM in response generation.
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
        Constructor for the SimpleBrain class.

        Args:
            name (str): The name of the agent.
            description (str): A description of the agent.
            model (str): The name of the LLM model to be used for response generation.
            memory_manager (Memory): An instance of a memory component implementing the `Memory` protocol for conversation history storage.
            system_prompt (str, optional): The system prompt template for the LLM, formatted with placeholders. Defaults to "".
            user_prompt (str, optional): The user prompt template for the LLM, formatted with placeholders. Defaults to "".
            api_config (dict, optional): Configuration for the LLM API including base URL and API key. Defaults to None.
                Can include:
                - api_base: The base URL for the API
                - api_key: The API key for authentication
                - use_env: Whether to use environment variables for configuration
        """

        self.name = name
        self.description = description
        self.model = model
        self.memory_manager = memory_manager
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.api_config = api_config or {}
        logging.info(f"Initialized brain: {self} with memory {self.memory_manager}")

    async def handle_chat_message(self, agent:BaseAgent, message: Message):
        """
        Handle incoming chat messages directed to the agent.

        This method is called whenever a chat message is received by the agent.
        It performs the following actions:

        1. Stores the received message in the memory manager.
        2. Checks if the message source is different from the agent itself (i.e., an incoming message).
        3. If it's an incoming message, updates the agent's attention to the message source.
        4. Calls `create_chat_message` to generate a response using the LLM.
        5. Sends the generated response message through the agent.

        Args:
            agent: The agent object for which the message is received.
            message (Message): The received message object (type: agentkit.messages.Message).
        """

        self.memory_manager.remember(message)
        if message.source != self.name:
            # Reply to a chat message from someone
            agent.attention = message.source
            response = await self.generate_chat_response(agent)
            await agent.send_message(response)

    async def generate_chat_response(self, agent:BaseAgent):
        """
        Generate a chat response for the agent based on the received message.
        """
        # format the system prompt
        context = self.get_context()
        system_prompt = self.system_prompt.format(name=self.name, description=self.description, context=context, target=agent.attention)
        
        messages = self.create_chat_messages_prompt(agent, system_prompt) # I don't like the wat this is implemented
        # Extract API configuration
        api_base = self.api_config.get('api_base')
        api_key = self.api_config.get('api_key')
        
        reply = await llm_chat(
            llm_model=self.model,
            messages=messages,
            api_base=api_base,
            api_key=api_key
        )

        msg = self.format_response(agent, reply)
        return msg

    def get_context(self):
        # Get the context from the memory system
        context = self.memory_manager.get_context()
        return context
    
    def create_chat_messages_prompt(self, agent, system_prompt:str) -> list:
        """
        Generate a chat message response using the LLM based on the current context and prompts.

        This method constructs system and user prompts for the LLM, incorporating the agent's name, description, conversation context,
        and target recipient. It then utilizes the `llm_processor` function to generate a response using the specified LLM model.
        The generated response is then formatted by removing mentions of the agent and target recipient's names and removing emojis.
        Finally, a Message object is created with the formatted response and returned.

        Args:
            agent: The agent object for which the chat message is being generated.

        Returns:
            Message: The generated message object (type: agentkit.messages.Message) containing the response content.
        """
        system_role = "system"
        user_role = "user"
        assistant_role = "assistant"
        
        messages_prompt = []
        messages_prompt.append({"role":system_role, "content": system_prompt})
        context = self.memory_manager.chat_log_for(target=agent.attention)
        for c in context:
            m = {"content": c.content.strip(), "role": assistant_role if c.source == self.name else user_role}
            messages_prompt.append(m)
        return messages_prompt


    def format_response(self, agent, reply):
        """
        Formats a reply message by removing specific prefixes and emojis, then encapsulates the cleaned message in a Message object.

        Args:
            agent (Agent): The agent object that contains details like whom the message is addressed to.
            reply (str): The original reply message that needs to be formatted.

        Returns:
            Message: A Message object that contains the source, recipient, content, and message type formatted for further processing.

        This method first removes the prefix that includes the name of this object and the agent's attention marker from the reply. 
        It also strips any emojis to ensure the message is plain text. Finally, it creates a new Message object with the specified attributes.
        """
        msg = Message(source=self.name, to=agent.attention, content=reply, message_type=MessageType.CHAT)
        return msg
