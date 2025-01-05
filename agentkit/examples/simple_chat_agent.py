import argparse
import asyncio
import json
import logging

# Import modules from agentkit framework
from agentkit.agents.simple_agent import SimpleAgent
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.simple_memory import SimpleMemory
from networkkit.messages import MessageType
from networkkit.network import HTTPMessageSender, ZMQMessageReceiver

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

async def main():
  """
  An example AI Agent that can send and recieve chat messages over the network.
  It will generate chat messages using an LLM (ollama/llama3.2 in this example) and converse with any other agent including humans.
  
  This function performs the following steps:

  1. Creates a `SimpleAgent` instance with the name, description, and an `HTTPMessageSender` for sending messages.
  2. Creates a `SimpleBrain` instance with the name, description, model, `SimpleMemory` instance for managing conversation history, system prompt, and user prompt.
  3. Registers a message handler for `MessageType.CHAT` with the agent, associating it with the `SimpleBrain.handle_chat_message` method.
  4. Creates a `ZMQMessageReceiver` for receiving messages from the message bus.
  5. Registers the agent as a subscriber with the message receiver.
  6. Starts the message receiver and agent tasks using `asyncio.create_task` and waits for them to complete with `asyncio.gather`.
  7. Handles exceptions like `KeyboardInterrupt` (Ctrl+C) for graceful shutdown and logs informative messages.
  8. Ensures proper shutdown by calling `stop` methods on both the message receiver and agent (if available and callable).
  """
  # Define some values
  model = "ollama/llama3.2" # The llm to use in the brain
  name = "Julia" #Name of the agent
  description = "A friendly AI assistant"
  system_prompt = """You are a friendly and helpful AI agent called Julia.
  You will be given the current conversation history so continue the conversation.
  """
  user_prompt = "Continue the conversation:\n{context}\n{name}:"
  bus_ip = "127.0.0.1" #IP address of the networkkit bus 

  # Create a SimpleAgent instance
  agent = SimpleAgent(
    name=name,
    description=description,
    message_sender=HTTPMessageSender(publish_address=f"http://{bus_ip}:8000")
  )

  # Create a brain instance using a SimpleMemory memory manager
  brain = SimpleBrain(
    name=name,
    description=description,
    model=model,
    memory_manager=SimpleMemory(max_history_length=6),
    system_prompt=system_prompt,
    user_prompt=user_prompt
  )
  
  # Register the brain to handle CHAT type messages
  agent.register_message_handler(MessageType.CHAT, brain.handle_chat_message)
  
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
  
  # Run the main function asynchronously
  asyncio.run(main())
