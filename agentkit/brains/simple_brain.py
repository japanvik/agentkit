from agentkit.memory.memory_protocol import Memory
from agentkit.messages import Message, MessageType
from agentkit.processor import llm_processor, remove_emojis


class SimpleBrain:
    """
    Simple rule-based brain for agent interaction, utilizing LLM for message generation.

    This class serves as the core decision-making component for an agent. It manages conversation flow, relies on memory for
    context, and interacts with an LLM to generate responses to chat messages.

    The brain operates based on pre-defined system and user prompts that can be formatted with placeholders like agent name,
    description, context, and target recipient. These prompts are used to guide the LLM in response generation.
    """

    def __init__(self, name: str, description: str, model: str, memory_manger: Memory, system_prompt: str = "", user_prompt: str = "") -> None:
        """
        Constructor for the SimpleBrain class.

        Args:
            name (str): The name of the agent.
            description (str): A description of the agent.
            model (str): The name of the LLM model to be used for response generation.
            memory_manger (Memory): An instance of a memory component implementing the `Memory` protocol for conversation history storage.
            system_prompt (str, optional): The system prompt template for the LLM, formatted with placeholders. Defaults to "".
            user_prompt (str, optional): The user prompt template for the LLM, formatted with placeholders. Defaults to "".
        """

        self.name = name
        self.description = description
        self.model = model
        self.memory_manager = memory_manger
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    async def handle_chat_message(self, agent, message: Message):
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
        if message.source != agent.name:
            # Reply to a chat message from someone
            agent.attention = message.source
            response = await self.create_chat_message(agent)
            agent.send_message(response)

    async def create_chat_message(self, agent) -> Message:
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

        context = self.memory_manager.get_chat_context(target=agent.attention)
        system_prompt = self.system_prompt.format(name=self.name, description=self.description, context=context, target=agent.attention)
        user_prompt = self.user_prompt.format(name=self.name, description=self.description, context=context, target=agent.attention)
        reply = await llm_processor(llm_model=self.model, system_prompt=system_prompt, user_prompt=user_prompt, stop=["\n", "<|im_end|>"])

        # Do some formatting
        reply = reply.replace(f"{self.name}:", "").strip()
        reply = reply.replace(f"{agent.attention}:", "").strip()
        reply = remove_emojis(reply)
        msg = Message(source=self.name, to=agent.attention, content=reply, message_type=MessageType.CHAT)
        return msg
