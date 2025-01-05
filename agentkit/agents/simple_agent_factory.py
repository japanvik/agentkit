# agentkit/agents/simple_agent_factory.py

import os
import importlib
import logging
from typing import Dict, Type, Any
from networkkit.messages import MessageType
from networkkit.network import HTTPMessageSender

from agentkit.agents.human_agent import HumanAgent
from agentkit.agents.simple_agent import SimpleAgent
from agentkit.brains.human_brain import HumanBrain
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.simple_memory import SimpleMemory

# Built-in implementations
BUILTIN_AGENTS = {
    "SimpleAgent": SimpleAgent,
    "HumanAgent": HumanAgent
}

BUILTIN_BRAINS = {
    "SimpleBrain": SimpleBrain,
    "HumanBrain": HumanBrain
}

BUILTIN_MEMORIES = {
    "SimpleMemory": SimpleMemory
}

def load_custom_plugins(plugins_dir: str) -> tuple[Dict[str, Type], Dict[str, Type], Dict[str, Type]]:
    """
    Load custom plugin implementations from the specified directory.
    
    Args:
        plugins_dir: Path to directory containing custom plugins
        
    Returns:
        Tuple of (agents, brains, memories) dictionaries mapping names to implementations
    """
    custom_agents = {}
    custom_brains = {}
    custom_memories = {}
    
    if not os.path.exists(plugins_dir):
        return custom_agents, custom_brains, custom_memories
        
    # Import custom implementations
    for category in ['agents', 'brains', 'memory']:
        category_dir = os.path.join(plugins_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        for file in os.listdir(category_dir):
            if file.endswith('.py') and not file.startswith('_'):
                module_name = file[:-3]
                try:
                    # Convert file path to module path relative to cwd
                    module_path = f"plugins.{category}.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Look for class definitions
                    for name, obj in module.__dict__.items():
                        if isinstance(obj, type):
                            if category == 'agents' and issubclass(obj, SimpleAgent):
                                custom_agents[name] = obj
                            elif category == 'brains' and issubclass(obj, SimpleBrain):
                                custom_brains[name] = obj
                            elif category == 'memory' and issubclass(obj, SimpleMemory):
                                custom_memories[name] = obj
                                
                except Exception as e:
                    logging.error(f"Error loading plugin {file}: {str(e)}")
                    
    return custom_agents, custom_brains, custom_memories

def simple_agent_factory(
    name: str,
    description: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    agent_type: str,
    brain_type: str,
    memory_type: str = "SimpleMemory",
    plugins_dir: str = "plugins",
    bus_ip: str = "127.0.0.1",
    ttl_minutes: int = 5,
    helo_interval: int = 300,
    cleanup_interval: int = 300,
    api_config: dict = None
):
    """
    Factory function to create an agent with its corresponding Brain and Memory.
    Supports both built-in and custom plugin implementations.
    
    Args:
        name: Agent name
        description: Agent description
        model: Model identifier
        system_prompt: System prompt template
        user_prompt: User prompt template
        agent_type: Type of agent to create
        brain_type: Type of brain to use
        memory_type: Type of memory to use
        plugins_dir: Directory containing custom plugins
        bus_ip: Message bus IP address
        ttl_minutes: Time-to-live for agent availability
        helo_interval: HELO message interval
        cleanup_interval: Cleanup interval
        api_config: Optional dictionary containing API configuration:
            - api_base: Base URL for the API
            - api_key: API key for authentication
            - use_env: Whether to use environment variables
            If not provided or None, defaults to environment variables or local settings
    """
    # Load custom plugins if directory exists
    custom_agents, custom_brains, custom_memories = load_custom_plugins(plugins_dir)
    
    # Initialize the message sender
    message_sender = HTTPMessageSender(publish_address=f"http://{bus_ip}:8000")

    # Get component classes (try custom plugins first, fall back to built-in)
    agent_class = custom_agents.get(agent_type) or BUILTIN_AGENTS.get(agent_type)
    brain_class = custom_brains.get(brain_type) or BUILTIN_BRAINS.get(brain_type)
    memory_class = custom_memories.get(memory_type) or BUILTIN_MEMORIES.get(memory_type)

    if not agent_class or not brain_class or not memory_class:
        missing = []
        if not agent_class: missing.append(f"agent_type: {agent_type}")
        if not brain_class: missing.append(f"brain_type: {brain_type}")
        if not memory_class: missing.append(f"memory_type: {memory_type}")
        raise ValueError(f"Could not find implementations for: {', '.join(missing)}")

    # Create the Agent instance
    agent = agent_class(
        name=name,
        description=description,
        message_sender=message_sender,
        bus_ip=bus_ip,
        ttl_minutes=ttl_minutes,
        helo_interval=helo_interval,
        cleanup_interval=cleanup_interval,
        api_config=api_config
    )

    # Create memory manager instance
    memory_manager = memory_class(max_history_length=6)

    # Create brain instance and link it to the agent
    brain = brain_class(
        name=name,
        description=description,
        model=model,
        memory_manager=memory_manager,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_config=api_config
    )

    # Register message handlers
    if agent_type == "SimpleAgent":
        agent.register_message_handler(MessageType.CHAT, brain.handle_chat_message)
    elif agent_type == "HumanAgent":
        agent.register_message_handler(MessageType.CHAT, agent.handle_chat_message)

    logging.info(f"Agent '{name}' created with model '{model}'. Handlers: {agent.message_handlers.items()}")

    return agent
