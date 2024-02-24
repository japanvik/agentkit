
from agentkit.memory.memory_protocol import Memory
from agentkit.messages import Message, MessageType
from agentkit.processor import llm_processor, remove_emojis


class SimpleBrain:
    def __init__(self, name:str, description:str, model:str, memory_manger:Memory, system_prompt:str="", user_prompt:str="") -> None:
        self.name = name
        self.description = description
        self.model = model
        self.memory_manager = memory_manger
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        
    async def handle_chat_message(self, agent, message:Message):
        self.memory_manager.remember(message)
        if message.source != agent.name:
            # Reply to a chat message from someone
            agent.attention = message.source
            response = await self.create_chat_message(agent)
            agent.send_message(response)

    async def create_chat_message(self, agent)->Message:
        """prepare the CHAT message"""
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
    