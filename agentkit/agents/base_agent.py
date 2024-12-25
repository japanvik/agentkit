# agentkit/agents/base_agent.py

from abc import ABC, abstractmethod
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Callable

from agentkit.handlers import handle_ack_message, handle_helo_message
from networkkit.messages import Message, MessageType
from networkkit.network import MessageSender  # Assuming MessageSender is properly defined
from rich.console import Console
from itertools import cycle

# Initialize Rich Console for color output
console = Console()

class BaseAgent(ABC):
    """
    Abstract BaseAgent class that provides common functionalities for all agents,
    including HELO/ACK handling and managing available agents.
    """

    AVAILABLE_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red", "bright_magenta", "bright_green"]
    color_cycle = cycle(AVAILABLE_COLORS)

    def __init__(
        self,
        name: str,
        description: str,
        message_sender: MessageSender,
        bus_ip: str = "127.0.0.1",
        ttl_minutes: int = 5,
        helo_interval: int = 300,  # seconds
        cleanup_interval: int = 360  # seconds
    ):
        self.name = name
        self.description = description
        self.message_sender = message_sender
        self.bus_ip = bus_ip
        self.ttl = timedelta(minutes=ttl_minutes)
        self.helo_interval = helo_interval
        self.cleanup_interval = cleanup_interval

        # Available agents: {agent_name: {"description": str, "last_seen": datetime}}
        self.available_agents: Dict[str, Dict[str, any]] = {}
        # Agent colors: {agent_name: color}
        self.agent_colors: Dict[str, str] = {}

        # Initialize message handlers dictionary
        self.message_handlers: Dict[MessageType, Callable[[Message], None]] = {}

        # Initialize asyncio tasks
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = True

        # Assign color to self for consistent messaging
        self.agent_colors[self.name] = next(self.color_cycle)
        
        # Define attention
        self.attention = "ALL"

        # Register HELO and ACK handlers
        self.register_message_handler(MessageType.HELO, handle_helo_message)
        self.register_message_handler(MessageType.ACK, handle_ack_message)

    async def handle_message(self, message: Message) -> Any:
        """
        Handle an incoming message received by the agent.
        """
        logging.info(f"Agent '{self.name}' received message: {message}")
        await self.message_queue.put(message)

    def register_message_handler(self, message_type: MessageType, handler: Callable[[Message], None]):
        """
        Registers a handler function for a specific message type.

        Args:
            message_type (MessageType): The message type for which to register the handler.
            handler (Callable[[Message], None]): The handler function.
        """
        self.message_handlers[message_type] = handler
        logging.info(f"Handler registered for message type: {message_type}")

    async def helo_broadcast_task(self):
        """
        Periodically sends HELO messages to all agents to announce availability.
        """
        while self.running:
            try:
                helo_message = Message(
                    source=self.name,
                    to="ALL",
                    content=self.description,
                    message_type=MessageType.HELO
                )
                await self.send_message(helo_message)
                logging.info("Broadcasted HELO message to ALL.")
            except Exception as e:
                logging.error(f"Error broadcasting HELO message: {e}")
            await asyncio.sleep(self.helo_interval)

    async def cleanup_available_agents_task(self):
        """
        Periodically removes agents that haven't been seen within the TTL.
        """
        while self.running:
            try:
                current_time = datetime.now()
                expired_agents = [
                    name for name, info in self.available_agents.items()
                    if current_time - info["last_seen"] > self.ttl
                ]
                for name in expired_agents:
                    del self.available_agents[name]
                    del self.agent_colors[name]
                    logging.info(f"Agent '{name}' expired and removed from available_agents.")
            except Exception as e:
                logging.error(f"Error during cleanup of available_agents: {e}")
            await asyncio.sleep(self.cleanup_interval)

    def add_task(self, task_name: str, coro: asyncio.coroutine):
        """
        Adds and starts a new asynchronous task.

        Args:
            task_name (str): The name to associate with the task.
            coro (asyncio.coroutine): The coroutine to be executed as a task.
        """
        if task_name in self.tasks:
            logging.warning(f"Task '{task_name}' already exists. Skipping.")
            return
        task = asyncio.create_task(coro, name=task_name)
        self.tasks[task_name] = task
        logging.info(f"Task '{task_name}' started.")

    async def send_message(self, message: Message):
        """
        Sends a message using the message sender component.

        Args:
            message (Message): The message to send.
        """
        try:
            await self.message_sender.send_message(message)
            logging.info(f"Sent {message.message_type} message to {message.to}.")
        except Exception as e:
            logging.error(f"Error sending message to {message.to}: {e}")

    def is_intended_for_me(self, message: Message) -> bool:
        """
        Checks if the message is intended for this agent.

        Args:
            message (Message): The message to check.

        Returns:
            bool: True if the message is intended for this agent, False otherwise.
        """
        return message.to == self.name or message.to == "ALL"

    async def start(self):
        """
        Starts the agent by initiating background tasks.
        """
        logging.info(f"Agent '{self.name}' starting.")
        # Start HELO broadcasting and cleanup tasks
        self.add_task("helo_broadcast", self.helo_broadcast_task())
        self.add_task("cleanup_available_agents", self.cleanup_available_agents_task())
        await self.run()

    async def run(self):
        """
        The main run loop of the agent. Should be overridden by subclasses to implement specific behaviors.
        """
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        """
        Stops the agent by cancelling all tasks and setting the running flag to False.
        """
        logging.info(f"Agent '{self.name}' stopping.")
        self.running = False
        for task_name, task in self.tasks.items():
            task.cancel()
            try:
                await task
                logging.info(f"Task '{task_name}' cancelled successfully.")
            except asyncio.CancelledError:
                logging.info(f"Task '{task_name}' cancelled.")
        self.tasks.clear()
        logging.info(f"Agent '{self.name}' stopped.")
