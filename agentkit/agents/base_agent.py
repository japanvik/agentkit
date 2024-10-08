# agentkit/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Any
from networkkit.messages import Message

class BaseAgent(ABC):
    def __init__(self, name: str, description: str, system_prompt: str, user_prompt: str, model: str):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

    @abstractmethod
    async def handle_message(self, message: Message) -> Any:
        pass

    @abstractmethod
    def is_intended_for_me(self, message: Message) -> bool:
        pass

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass
