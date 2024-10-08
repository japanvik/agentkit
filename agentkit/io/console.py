import aioconsole
from networkkit.messages import Message, MessageType


async def ainput(agent) -> None:
    """
    Asynchronous function to get user input for the specified agent.

    This function utilizes `aioconsole.ainput` to display a prompt and capture user input for the given agent.
    It constructs a message object based on the user input and sends it through the agent's message sender.

    The prompt format includes the agent's name and current attention. 

    Expected user input format can be one of the following:

    1. Free-form message: If the message doesn't start with "TO:", it is assumed to be directed to the agent's current attention.
    2. Directed message: If the message starts with "TO:" followed by a recipient name, it is considered a directed message to that recipient. The content of the message is extracted from the remaining part after removing the "TO:" and recipient name.

    The function raises a `ValueError` if the user enters "exit" to signal termination.

    Args:
        agent: The agent object for which to get user input.

    Raises:
        ValueError: If the user enters "exit".
    """

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
        await agent.message_sender.send_message(msg)
    else:
        raise ValueError("Exit was called")
