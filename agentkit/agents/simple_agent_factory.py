# agentkit/agents/simple_agent_factory.py

from agentkit.agents.simple_agent import SimpleAgent
from agentkit.brains.simple_brain_instruct import SimpleBrainInstruct
from agentkit.memory.simple_memory import SimpleMemory
from networkkit.messages import MessageType
from networkkit.network import HTTPMessageSender
import logging

def simple_agent_factory(
    name: str,
    description: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    bus_ip: str = "127.0.0.1",
):
    """
    Factory function to create a SimpleAgent with its corresponding Brain and Memory.
    """
    # Initialize the message sender
    message_sender = HTTPMessageSender(publish_address=f"http://{bus_ip}:8000")

    # Create the SimpleAgent instance
    agent = SimpleAgent(
        name=name,
        description=description,
        message_sender=message_sender,
    )

    # Create the SimpleBrainInstruct instance and link it to the agent
    brain = SimpleBrainInstruct(
        name=name,
        description=description,
        model=model,
        memory_manager=SimpleMemory(max_history_length=6),
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # Register the chat message handler with the agent
    agent.add_message_handler(MessageType.CHAT, brain.handle_chat_message)

    logging.info(f"Agent '{name}' created with model '{model}'.")

    return agent
