import argparse
import asyncio
import logging

from agentkit.agents.simple_agent import SimpleAgent
from agentkit.io import console
from agentkit.handlers import print_chat_message
from networkkit.messages import MessageType
from networkkit.network import HTTPMessageSender, ZMQMessageReceiver

# Task definitions
async def user_input_task(agent):
    while agent.running:
        await console.ainput(agent)
        await asyncio.sleep(0.1)

async def main(name:str, description:str, bus_ip:str="127.0.0.1"):
    agent = SimpleAgent(name=name, description=description, message_sender=HTTPMessageSender(publish_address=f"http://{bus_ip}:8000"))
    # Register the tasks to the agent
    agent.add_task("user_input", user_input_task(agent))
    agent.add_message_handler(MessageType.CHAT, print_chat_message)
   
    # Initialize the Message Reciever 
    message_receiver = ZMQMessageReceiver(subscribe_address=f"tcp://{bus_ip}:5555")
    message_receiver.register_subscriber(agent)

    try:
        receiver_task = asyncio.create_task(message_receiver.start())
        agent_task = asyncio.create_task(agent.start())
        await asyncio.gather(receiver_task, agent_task)
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed. Stopping agent.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Ensure that stop is called for all the tasks
        if hasattr(message_receiver, 'stop') and callable(getattr(message_receiver, 'stop')):
            message_receiver.stop()
        if hasattr(agent, 'stop') and callable(getattr(agent, 'stop')):
            await agent.stop()  # If stop is an async method

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a console conversational UI for humans")
    parser.add_argument("--name", help="Name of the Human agent. Default 'Human'", default="Human")
    parser.add_argument("--description", help="Description of the agent. Introduce yourself to the world. Default 'I am a human agent. Ask me anything!'", default="I am a human agent. Ask me anything!")
    parser.add_argument("--bus_ip", default="127.0.0.1", help="The IP Address of the bus to subscribe to. Default 127.0.0.1")
    args = parser.parse_args()
    asyncio.run(main(name=args.name, 
                     description=args.description, 
                     bus_ip=args.bus_ip))
