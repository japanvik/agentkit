import argparse
import asyncio
import json
import logging

from agentkit.agents.simple_agent import SimpleAgent
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.messages import MessageType
from agentkit.network import HTTPMessageSender, ZMQMessageReceiver
from agentkit.memory.simple_memory import SimpleMemory

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def get_config_value(config, parameter_name, override):
    # 
    if override:
        return override
    else:
        if parameter_name in config['agent'].keys():
            return config['agent'][parameter_name]
        else:
            raise ValueError(f'Required Parameter "{parameter_name}" is not defined in the config file or instance creation')


async def main(name:str, description:str, config_file:str, model:str, bus_ip:str="127.0.0.1"):
    # Load the configuration
    config = load_config(config_file)
    name=get_config_value(config, "name", name)
    description=get_config_value(config, "description", description)
    model = get_config_value(config, "model", model)
    system_prompt = get_config_value(config, "system_prompt", "")
    user_prompt = get_config_value(config, "user_prompt", "")
    
    agent = SimpleAgent(
        config=config,
        name=name,
        description=description,
        message_sender=HTTPMessageSender(publish_address=f"http://{bus_ip}:8000"),
    )

    brain = SimpleBrain(name=name,
                        description=description,
                        model=model,
                        memory_manger=SimpleMemory(max_history_length=6),
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                        )
    agent.add_message_handler(MessageType.CHAT, brain.handle_chat_message)
    
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
    parser = argparse.ArgumentParser(description="Run an agent with specified name and description")
    parser.add_argument("--config", help="Path to the configuration file", required=True)
    parser.add_argument("--name", help="Name of the agent. Optional, overriding the config file")
    parser.add_argument("--description", help="Description of the agent. Optional overriding the config file")
    parser.add_argument("--model", help="The litellm model you want to use. Optional overriding the config file")
    parser.add_argument("--bus_ip", default="127.0.0.1", help="The IP Address of the bus to subscribe to. Default 127.0.0.1")
    args = parser.parse_args()

    asyncio.run(main(name=args.name, 
                     description=args.description, 
                     config_file=args.config, 
                     model=args.model, 
                     bus_ip=args.bus_ip))
