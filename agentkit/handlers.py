from agentkit.messages import Message, MessageType

async def default_handle_helo_message(agent, message:Message):
    agent.attention = message.source
    response = Message(source=agent.name, to=message.source, content=agent.description, message_type=MessageType.ACK)
    agent.send_message(response)
        
async def print_chat_message(agent, message:Message):
    agent.attention = message.source
    print(f"\n## {message.source} to {message.to}[{message.message_type}]: {message.content}")