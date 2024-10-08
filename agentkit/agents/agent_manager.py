# agent_manager.py

import asyncio
import logging
import signal
from typing import Any, Dict, List

from networkkit.network import ZMQMessageReceiver
from agentkit.agents.simple_agent_factory import simple_agent_factory
from networkkit.messages import Message

class AgentManager:
    """
    Manages multiple agents and coordinates their lifecycle.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AgentManager with the provided configuration.

        Args:
            config (dict): The configuration dictionary containing agent and bus information.
        """
        self.config = config
        self.agents: List[Any] = []
        self.message_receiver = None
        self.receiver_task = None
        self.agent_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

    def load_agents(self):
        """
        Initializes agents based on the configuration and registers them with the message receiver.
        """
        agents_config = self.config.get("agents", [])
        if not agents_config:
            logging.warning("No agents defined in the configuration.")

        bus_ip = self.config.get("bus_ip", "127.0.0.1")

        for agent_conf in agents_config:
            try:
                agent = simple_agent_factory(
                    name=agent_conf["name"],
                    description=agent_conf["description"],
                    model=agent_conf["model"],
                    system_prompt=agent_conf.get("system_prompt", ""),
                    user_prompt=agent_conf.get("user_prompt", ""),
                    bus_ip=bus_ip
                )
                self.agents.append(agent)
                logging.info(f"Loaded agent: {agent_conf['name']}")
            except KeyError as e:
                logging.error(f"Missing required agent configuration parameter: {e}")
            except Exception as e:
                logging.error(f"Error creating agent {agent_conf.get('name', 'Unknown')}: {e}")

    async def start(self):
        """
        Starts the message receiver and all agents.
        """
        bus_ip = self.config.get("bus_ip", "127.0.0.1")
        self.message_receiver = ZMQMessageReceiver(subscribe_address=f"tcp://{bus_ip}:5555")

        # Register all agents with the message receiver
        for agent in self.agents:
            self.message_receiver.register_subscriber(agent)

        # Create tasks
        self.receiver_task = asyncio.create_task(
            self.message_receiver.start(),
            name="ReceiverTask"
        )
        logging.info("Message receiver started.")

        for agent in self.agents:
            agent_task = asyncio.create_task(
                agent.start(),
                name=f"AgentTask-{agent.name}"
            )
            self.agent_tasks.append(agent_task)
            logging.info(f"Agent task started: {agent.name}")

    async def shutdown(self):
        """
        Initiates the shutdown sequence for all agents and the message receiver.
        """
        logging.info("Initiating shutdown sequence...")

        # Stop the message receiver
        if self.message_receiver:
            await self.message_receiver.stop()

        # Stop all agents
        for agent in self.agents:
            await agent.stop()
            logging.info(f"Agent stopped: {agent.name}")

        # Cancel the receiver task if it's still running
        if self.receiver_task and not self.receiver_task.done():
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                logging.info("Receiver task cancelled successfully.")

        # Cancel all agent tasks if they are still running
        for task in self.agent_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logging.info(f"Task '{task.get_name()}' cancelled successfully.")

        logging.info("All agents and message receiver have been shut down.")

    def register_signal_handlers(self):
        """
        Registers signal handlers for graceful shutdown.
        """
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(self.handle_signal(sig)))
                logging.info(f"Registered signal handler for {sig.name}")
            except NotImplementedError:
                # Signals are not implemented on some platforms (e.g., Windows)
                logging.warning(f"Signal handler for {sig.name} not implemented on this platform.")

    async def handle_signal(self, sig):
        """
        Handles received shutdown signals.

        Args:
            sig: The signal received.
        """
        logging.info(f"Received exit signal {sig.name}...")
        self.shutdown_event.set()

    async def run(self):
        """
        Runs the AgentManager, handling initialization and shutdown.
        """
        self.register_signal_handlers()
        self.load_agents()
        await self.start()

        # Wait until a shutdown signal is received
        await self.shutdown_event.wait()

        # Proceed with shutdown
        await self.shutdown()
