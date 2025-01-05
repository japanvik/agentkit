import argparse
import asyncio
import logging

from agentkit.agents.agent_manager import AgentManager

async def main(config_path: str, log_level: str):
    """A chat agent using a config file and the AgentManager to spin up agents quickly.
       This will create the same agent as the simple_chat_agent.py
       Notice that the AgentManager does a lot of the boilerplate work for agent creation and the code is a lot simpler.

    Args:
        config_path (str): path to the configuration file in JSON
        loglevel (str): log level of the logger
    """
    # Convert the provided loglevel string to a logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
        logging.info("Application stopped by user.")
        
        
import argparse
import asyncio
import logging

from agentkit.agents.agent_manager import AgentManager

async def main(config_path: str):
    """A chat agent using a config file and the AgentManager to spin up agents quickly.
       This will create the same agent as the simple_chat_agent.py
       Notice that the AgentManager does a lot of the boilerplate work for agent creation and the code is a lot simpler.

    Args:
        config_path (str): path to the configuration file in JSON
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load configuration using AgentManager
    config = AgentManager.load_config(config_path)
    if not config:
        return

    # Create and run the agent manager
    manager = AgentManager(config)
    await manager.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start multiple agents based on configuration.")
    parser.add_argument("--config", help="Path to the JSON configuration file.", required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main(config_path=args.config))
    except KeyboardInterrupt:
        logging.info("Application stopped by user.")
