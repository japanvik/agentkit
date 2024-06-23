from agentkit.agents.simple_agent import SimpleAgent
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.simple_memory import SimpleMemory
from agentkit.messages import MessageType
from agentkit.network import HTTPMessageSender

def simple_agent_factory(
    name: str,
    description: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    bus_ip: str = "127.0.0.1",
):
      
    # Load the configuration from the file
    config = {'name': name, 'description': description}
  
    # Extract configuration values with overriden options from arguments (if provided)
    # Create the SimpleAgent instance
    agent = SimpleAgent(
        config=config,
        name=name,
        description=description,
        message_sender=HTTPMessageSender(publish_address=f"http://{bus_ip}:8000"),
    )
  
    # Create the SimpleBrain instance
    brain = SimpleBrain(
        name=name,
        description=description,
        model=model,
        memory_manger=SimpleMemory(max_history_length=6),
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
  
    # Register the chat message handler with the agent
    agent.add_message_handler(MessageType.CHAT, brain.handle_chat_message)
    
    return agent
