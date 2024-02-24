import aioconsole
from agentkit.messages import Message, MessageType

async def ainput(agent) -> None:
    prompt = f"##{agent.name} ({agent.attention}):"
    message = await aioconsole.ainput(prompt)
    if message != "exit":
        if not message.startswith("TO:"):
            destination = agent.attention
            content = message.strip()
        else:
            first_part = message.split(",")[0]
            content = message.replace(first_part, "").strip()[1:]
            destination = first_part.replace("TO:", "").strip()
        msg = Message(source=agent.name, to=destination, content=content, message_type=MessageType.CHAT)
        agent.message_sender.send_message(msg)
    else:
        raise ValueError("Exit was called")
