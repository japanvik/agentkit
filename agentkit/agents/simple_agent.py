"""Simple agent implementation."""
from typing import Optional, Dict, Any
from agentkit.agents.base_agent import BaseAgent
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.simple_memory import SimpleMemory
from networkkit.messages import Message, MessageType

# Built-in implementations
BUILTIN_BRAINS = {
    "SimpleBrain": SimpleBrain,
    "HumanBrain": None  # Imported on demand to avoid circular imports
}

BUILTIN_MEMORIES = {
    "SimpleMemory": SimpleMemory,
    "AdvancedMemory": None  # Imported on demand to avoid circular imports
}

class SimpleAgent(BaseAgent):
    """
    Simple agent implementation that uses configurable brain and memory components.
    
    This agent supports both built-in and custom plugin implementations for its
    brain and memory components.
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        brain: Optional[SimpleBrain] = None,
        memory: Optional[SimpleMemory] = None,
        message_sender: Optional['MessageSender'] = None
    ) -> None:
        """
        Initialize the simple agent.
        
        Args:
            name: Agent's name
            config: Configuration dictionary
            brain: Optional brain component (will create based on config if None)
            memory: Optional memory component (will create based on config if None)
            message_sender: Optional message sender for delegating communication
        """
        # Create memory component if not provided
        if memory is None:
            memory_type = config.get('memory_type', 'SimpleMemory')
            
            # Import custom memory type if specified
            if memory_type not in BUILTIN_MEMORIES or BUILTIN_MEMORIES[memory_type] is None:
                try:
                    # Try to import from plugins
                    from plugins.memory.advanced_memory import AdvancedMemory
                    BUILTIN_MEMORIES['AdvancedMemory'] = AdvancedMemory
                except ImportError:
                    # Fall back to SimpleMemory
                    memory_type = 'SimpleMemory'
            
            memory_class = BUILTIN_MEMORIES[memory_type]
            memory = memory_class()
            
        # Create brain component if not provided
        if brain is None:
            brain_type = config.get('brain_type', 'SimpleBrain')
            
            # Import custom brain type if specified
            if brain_type not in BUILTIN_BRAINS or BUILTIN_BRAINS[brain_type] is None:
                try:
                    # Try to import from plugins
                    from agentkit.brains.human_brain import HumanBrain
                    BUILTIN_BRAINS['HumanBrain'] = HumanBrain
                except ImportError:
                    # Fall back to SimpleBrain
                    brain_type = 'SimpleBrain'
            
            brain_class = BUILTIN_BRAINS[brain_type]
            brain = brain_class(
                name=name,
                description=config.get('description', ''),
                model=config.get('model', 'llama2'),
                memory_manager=memory,
                system_prompt=config.get('system_prompt', ''),
                user_prompt=config.get('user_prompt', ''),
                api_config=config.get('api_config', {})
            )
        
        # Initialize base agent with components
        super().__init__(
            name=name,
            config=config,
            brain=brain,
            memory=memory,
            message_sender=message_sender
        )
