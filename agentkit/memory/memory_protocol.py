from typing import List, Protocol
from agentkit.messages import Message


class Memory(Protocol):
    
    def remember(self, message: Message) -> None:
        ...
    
    def get_history(self) -> List[Message]:
        ...    