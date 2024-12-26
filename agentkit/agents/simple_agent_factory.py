# agentkit/agents/simple_agent_factory.py

from agentkit.agents.human_agent import HumanAgent
from agentkit.agents.simple_agent import SimpleAgent
from agentkit.brains.human_brain import HumanBrain
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.brains.simple_brain_instruct import SimpleBrainInstruct
from agentkit.memory.simple_memory import SimpleMemory

MEMORIES = {
    "SimpleMemory": SimpleMemory
}
from networkkit.messages import MessageType
from networkkit.network import HTTPMessageSender
import logging

AGENTS = {
    "SimpleAgent": SimpleAgent,
    "HumanAgent": HumanAgent
}

BRAINS = {
    "SimpleBrain": SimpleBrain,
    "SimpleBrainInstruct": SimpleBrainInstruct,
    "HumanBrain": HumanBrain
}

def simple_agent_factory(
    name: str,
    description: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    agent_type: str,
    brain_type: str,
    memory_type: str = "SimpleMemory",
    bus_ip: str = "127.0.0.1",
    ttl_minutes: int = 5,
    helo_interval: int = 300,
    cleanup_interval: int = 300
):
    """
    Factory function to create a SimpleAgent with its corresponding Brain and Memory.
    """
    # Initialize the message sender
    message_sender = HTTPMessageSender(publish_address=f"http://{bus_ip}:8000")

    # Create the Agent instance
    agent_class = AGENTS.get(agent_type, SimpleAgent)
    agent = agent_class(
        name=name,
        description=description,
        message_sender=message_sender,
        bus_ip=bus_ip,
        ttl_minutes=ttl_minutes,
        helo_interval=helo_interval,
        cleanup_interval=cleanup_interval
    )

    # Create the SimpleBrainInstruct instance and link it to the agent
    brain_class = BRAINS.get(brain_type, SimpleBrainInstruct)
    # Create memory manager instance
    memory_class = MEMORIES.get(memory_type, SimpleMemory)
    memory_manager = memory_class(max_history_length=6)

    brain = brain_class(
        name=name,
        description=description,
        model=model,
        memory_manager=memory_manager,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # Register the chat message handler with the agent
    if agent_type in ["SimpleAgent"]:
        agent.register_message_handler(MessageType.CHAT, brain.handle_chat_message)

    if agent_type in ["HumanAgent"]:
        agent.register_message_handler(MessageType.CHAT, agent.handle_chat_message)

    logging.info(f"Agent '{name}' created with model '{model}'. Handlers: {agent.message_handlers.items()}")

    return agent
