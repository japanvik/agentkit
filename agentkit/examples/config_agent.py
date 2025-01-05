import argparse
import asyncio
import logging
import signal
import sys

from agentkit.agents.agent_manager import AgentManager

# Global variable to store the agent manager instance
manager = None

def signal_handler(signum, frame):
    """Handle shutdown signals by initiating cleanup."""
    logging.info(f"Received exit signal {signal.Signals(signum).name}...")
    logging.info("Initiating shutdown sequence...")
    if manager:
        # Schedule the manager's shutdown
        loop = asyncio.get_event_loop()
        loop.create_task(shutdown())

async def shutdown():
    """Clean shutdown of the agent manager and application."""
    global manager
    if manager:
        try:
            await manager.stop()
            logging.info("All agents and message receiver have been shut down.")
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            # Force exit after cleanup
            sys.exit(0)

async def main(config_path: str, log_level: str = "INFO"):
    """A chat agent using a config file and the AgentManager to spin up agents quickly.
       This will create the same agent as the simple_chat_agent.py
       Notice that the AgentManager does a lot of the boilerplate work for agent creation and the code is a lot simpler.

    Args:
        config_path (str): path to the configuration file in JSON
        log_level (str): log level of the logger (default: "INFO")
    """
    global manager

    # Convert the provided loglevel string to a logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_handler)
        logging.info(f"Registered signal handler for {signal.Signals(sig).name}")

    # Load configuration using AgentManager
    config = AgentManager.load_config(config_path)
    if not config:
        return

    # Create and run the agent manager
    manager = AgentManager(config)
    await manager.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start multiple agents based on configuration.")
    parser.add_argument(
        "--config", 
        help="Path to the JSON configuration file.", 
        required=True
    )
    parser.add_argument(
        "--loglevel", 
        help="Logging level (e.g. DEBUG, INFO, WARN, ERROR, CRITICAL).", 
        default="INFO"
    )

    args = parser.parse_args()
    try:
        asyncio.run(main(config_path=args.config, log_level=args.loglevel))
    except KeyboardInterrupt:
        # The signal handler will take care of the cleanup
        pass
