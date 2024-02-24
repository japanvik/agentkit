
from agentkit.messages import Message


class SimpleMemory:
    def __init__(self, max_history_length:int=10) -> None:
        self.history = []
        self.max_history_length = max_history_length

    def remember(self, message: Message) -> None:
        # Store the conversation in the history
        if len(self.history) >= self.max_history_length:
            self.history.pop(0) # remove oldest message
        self.history.append(message)
        
    def get_history(self) -> list[Message]:
        return self.history
    
    def get_chat_context(self ,target:str, prefix:str="") -> str:
        chat_log = [x for x in self.get_history() if (target in [x.to, x.source]) and x.message_type=="CHAT"]
        context =  "\n".join(f"{prefix}{x.source}: {x.content.strip()}" for x in chat_log)
        return context

