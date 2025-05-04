"""
Example demonstrating the use of the send_message tool.

This example shows how to create a TaskAwareAgent that uses the send_message tool
to communicate with other agents, rather than directly calling the send_message method.
"""

import asyncio
import logging
import os
from typing import Dict, Any

from networkkit.messages import Message, MessageType

from agentkit.agents.task_aware_agent import TaskAwareAgent
from agentkit.functions.functions_registry import DefaultFunctionsRegistry, FunctionDescriptor, ParameterDescriptor


async def main():
    """Run the example."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a registry for functions
    functions_registry = DefaultFunctionsRegistry()

    # Create two agents
    agent1 = TaskAwareAgent(name="Agent1", config={"name": "Agent1"})
    agent2 = TaskAwareAgent(name="Agent2", config={"name": "Agent2"})

    # Register tools with the agents
    agent1.register_tools(functions_registry)
    agent2.register_tools(functions_registry)

    # Set the functions registry on the agents
    agent1.functions_registry = functions_registry
    agent2.functions_registry = functions_registry

    # Start the agents
    await agent1.start()
    await agent2.start()

    try:
        # Agent1 sends a message to Agent2 using the send_message tool
        logging.info("Agent1 is sending a message to Agent2 using the send_message tool")
        result = await agent1.functions_registry.execute(
            function="send_message",
            parameters={
                "recipient": "Agent2",
                "content": "Hello from Agent1!",
                "message_type": "CHAT"
            }
        )
        logging.info(f"Message sent: {result}")

        # Wait a moment for the message to be processed
        await asyncio.sleep(1)

        # Agent2 sends a response to Agent1 using the send_message tool
        logging.info("Agent2 is sending a response to Agent1 using the send_message tool")
        result = await agent2.functions_registry.execute(
            function="send_message",
            parameters={
                "recipient": "Agent1",
                "content": "Hello back from Agent2!",
                "message_type": "CHAT"
            }
        )
        logging.info(f"Message sent: {result}")

        # Wait a moment for the message to be processed
        await asyncio.sleep(1)

    finally:
        # Stop the agents
        await agent1.stop()
        await agent2.stop()


if __name__ == "__main__":
    asyncio.run(main())
