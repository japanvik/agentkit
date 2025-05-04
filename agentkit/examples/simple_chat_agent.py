import asyncio
import logging

# Import modules from agentkit framework
from agentkit.utils import AgentRunner

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

async def main():
  """
  An example AI Agent that can send and receive chat messages over the network.
  It will generate chat messages using an LLM (ollama/qwen3 in this example) and converse with any other agent including humans.
  
  This example uses the AgentRunner utility to simplify agent setup and ensure proper lifecycle management,
  including graceful shutdown when Ctrl+C is pressed.
  
  The AgentRunner handles:
  1. Creating the agent with the specified configuration
  2. Setting up the message receiver and registering the agent
  3. Starting the agent and message receiver
  4. Handling signals for graceful shutdown
  5. Properly cleaning up resources during shutdown
  """
  # Create and run an agent with AgentRunner
  runner = AgentRunner(
    name="Julia",
    description="A friendly AI assistant",
    model="ollama/qwen3",
    system_prompt="""You are a friendly and helpful AI agent called Julia.
    You will be given the current conversation history so continue the conversation.
    """,
    user_prompt="Continue the conversation:\n{context}\n{name}:"
  )
  
  # Run the agent with proper lifecycle management
  await runner.run()

if __name__ == "__main__":
  # Run the main function asynchronously
  asyncio.run(main())
