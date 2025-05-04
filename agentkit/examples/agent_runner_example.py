#!/usr/bin/env python3
"""
Example script demonstrating the use of the AgentRunner utility.

This script shows how to create and run an agent using the AgentRunner utility,
which simplifies agent lifecycle management including graceful shutdown.

Usage:
    python agent_runner_example.py [--model MODEL] [--bus_ip BUS_IP]

Example:
    python agent_runner_example.py --model ollama/qwen3 --bus_ip 127.0.0.1
"""

import argparse
import asyncio
import logging

from agentkit.utils.agent_runner import AgentRunner

async def main(model: str = "ollama/qwen3", bus_ip: str = "127.0.0.1"):
    """
    Run a simple agent using the AgentRunner utility.
    
    Args:
        model: The LLM model to use for the agent's brain
        bus_ip: The IP address of the message bus
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the agent using AgentRunner
    runner = AgentRunner(
        name="Julia",
        description="A friendly AI assistant",
        model=model,
        system_prompt="You are a friendly and helpful AI agent called Julia. "
                      "You will be given the current conversation history so continue the conversation.",
        bus_ip=bus_ip
    )
    
    # This will block until the agent is stopped (e.g., by pressing Ctrl+C)
    await runner.run()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a simple agent using AgentRunner.")
    parser.add_argument(
        "--model", 
        help="LLM model to use (default: ollama/qwen3)",
        default="ollama/qwen3"
    )
    parser.add_argument(
        "--bus_ip", 
        help="IP address of the message bus (default: 127.0.0.1)",
        default="127.0.0.1"
    )
    
    args = parser.parse_args()
    
    try:
        # Run the main function with the provided arguments
        asyncio.run(main(model=args.model, bus_ip=args.bus_ip))
    except KeyboardInterrupt:
        # The AgentRunner's signal handler will take care of the cleanup
        pass
