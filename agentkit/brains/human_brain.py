# Human brain
from agentkit.memory.memory_protocol import Memory
from litellm import Message

class HumanBrain:
    def __init__(self, name: str, description: str, model: str, memory_manager: Memory, system_prompt: str = "", user_prompt: str = "") -> None:
        # Most of these are just dummy data.
        self.name = name
        self.description = description
        self.model = model
        #self.memory_manager = memory_manager
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

