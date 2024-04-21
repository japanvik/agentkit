import argparse
import asyncio
import json
import logging

# Import modules from agentkit framework
from agentkit.agents.simple_agent import SimpleAgent
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.messages import MessageType
from agentkit.network import HTTPMessageSender, ZMQMessageReceiver
from agentkit.memory.simple_memory import SimpleMemory

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


def load_config(file_path):
  """
  Loads the configuration from a JSON file.

  This function opens the specified file path, reads its JSON content,
  and returns the parsed configuration data.

  Args:
      file_path (str): The path to the configuration file.

  Returns:
      dict: The parsed configuration data.
  """

  with open(file_path, 'r') as file:
    return json.load(file)


def get_config_value(config, parameter_name, override):
  """
  Retrieves a configuration value from the provided dictionary with an optional override.

  This function checks for the provided parameter name within the 'agent' section of the configuration dictionary.
  If an override value is provided, it returns the override. Otherwise, it raises a ValueError if the parameter is not found.

  Args:
      config (dict): The configuration dictionary.
      parameter_name (str): The name of the configuration parameter to retrieve.
      override (str, optional): An optional override value for the parameter. Defaults to "".

  Returns:
      str: The retrieved configuration value or the provided override value.

  Raises:
      ValueError: If the parameter is not found in the configuration and no override is provided.
  """

  if override:
    return override
  else:
    if parameter_name in config['agent'].keys():
      return config['agent'][parameter_name]
    else:
      raise ValueError(f'Required Parameter "{parameter_name}" is not defined in the config file or instance creation')


async def main(name: str, description: str, config_file: str, model: str, bus_ip: str = "127.0.0.1"):
  """
  The main asynchronous entry point for the chatbot agent.

  This function performs the following steps:

  1. Loads the configuration from the specified file using `load_config`.
  2. Extracts configuration values for agent name, description, model, system prompt, and user prompt using `get_config_value`.
  3. Creates a `SimpleAgent` instance with the loaded configuration, name, description, and an `HTTPMessageSender` for sending messages.
  4. Creates a `SimpleBrain` instance with the name, description, model, `SimpleMemory` instance for managing conversation history, system prompt, and user prompt.
  5. Registers a message handler for `MessageType.CHAT` with the agent, associating it with the `SimpleBrain.handle_chat_message` method.
  6. Creates a `ZMQMessageReceiver` for receiving messages from the message bus.
  7. Registers the agent as a subscriber with the message receiver.
  8. Starts the message receiver and agent tasks using `asyncio.create_task` and waits for them to complete with `asyncio.gather`.
  9. Handles exceptions like `KeyboardInterrupt` (Ctrl+C) for graceful shutdown and logs informative messages.
  10. Ensures proper shutdown by calling `stop` methods on both the message receiver and agent (if available and callable).

  Args:
      name (str): The name of the agent (optional, can be overridden by config).
      description (str): A description of the agent (optional, can be overridden by config).
      config_file (str): The path to the configuration file.
      model (str): The LLM model to be used by the chatbot (optional, can be overridden by config).
      bus_ip (str, optional): The IP address of the message bus (default: "127.0.0.1").
  """

  # Load the configuration from the file
  config = load_config(config_file)

  # Extract configuration values with overriden options from arguments (if provided)
  name = get_config_value(config, "name", name)
  description = get_config_value(config, "description", description)
  model = get_config_value(config, "model", model)
  system_prompt = get_config_value(config, "system_prompt", "")
  
  user_prompt = get_config_value(config, "user_prompt", "")

  # Create the SimpleAgent instance
  agent = SimpleAgent(
      config=config,
      name=name,
      description=description,
      message_sender=HTTPMessageSender(publish_address=f"http://{bus_ip}:8000"),
  )

  # Create the SimpleBrain instance
  brain = SimpleBrain(
      name=name,
      description=description,
      model=model,
      memory_manger=SimpleMemory(max_history_length=6),
      system_prompt=system_prompt,
      user_prompt=user_prompt
  )

  # Register the chat message handler with the agent
  agent.add_message_handler(MessageType.CHAT, brain.handle_chat_message)

  # Create the message receiver for the message bus
  message_receiver = ZMQMessageReceiver(subscribe_address=f"tcp://{bus_ip}:5555")
  message_receiver.register_subscriber(agent)

  try:
    # Start the message receiver and agent tasks asynchronously
    receiver_task = asyncio.create_task(message_receiver.start())
    agent_task = asyncio.create_task(agent.start())

    # Wait for both tasks to complete
    await asyncio.gather(receiver_task, agent_task)

  except KeyboardInterrupt:
    logging.info("Ctrl+C pressed. Stopping agent.")

  except Exception as e:
    logging.error(f"An error occurred: {e}")

  finally:
    # Ensure proper shutdown by calling stop methods (if available)
    if hasattr(message_receiver, 'stop') and callable(getattr(message_receiver, 'stop')):
      message_receiver.stop()
    if hasattr(agent, 'stop') and callable(getattr(agent, 'stop')):
      await agent.stop()  # If stop is an async method

if __name__ == "__main__":
  # Parse command-line arguments for configuration, name, description, model, and bus IP
  parser = argparse.ArgumentParser(description="Run an agent with specified name and description")
  parser.add_argument("--config", help="Path to the configuration file", required=True)
  parser.add_argument("--name", help="Name of the agent. Optional, overriding the config file")
  parser.add_argument("--description", help="Description of the agent. Optional overriding the config file")
  parser.add_argument("--model", help="The litellm model you want to use. Optional overriding the config file")
  parser.add_argument("--bus_ip", default="127.0.0.1", help="The IP Address of the bus to subscribe to. Default 127.0.0.1")
  args = parser.parse_args()

  # Run the main function asynchronously
  asyncio.run(main(name=args.name,
                   description=args.description,
                   config_file=args.config,
                   model=args.model,
                   bus_ip=args.bus_ip))
