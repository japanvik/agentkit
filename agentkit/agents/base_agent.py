# agentkit/agents/base_agent.py

from abc import ABC, abstractmethod
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Callable

from agentkit.handlers import handle_ack_message, handle_helo_message
from networkkit.messages import Message, MessageType
from networkkit.network import MessageSender
from rich.console import Console
from itertools import cycle

# Initialize Rich Console for color output
console = Console()

class BaseAgent(ABC):
    """
    Abstract BaseAgent class that provides common functionalities for all agents,
    including HELO/ACK handling and managing available agents.
    """

    AVAILABLE_COLORS = [
        "cyan", "magenta", "green", "yellow",
        "blue", "red", "bright_magenta", "bright_green"
    ]
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
        """
        Initialize the BaseAgent with a name, description, and optional configuration
        parameters such as bus IP, TTL, and intervals for sending HELO and performing cleanup.
        """
        self.name = name
        self.description = description
        self.message_sender = message_sender
        self.bus_ip = bus_ip
        self.ttl = timedelta(minutes=ttl_minutes)
        self.helo_interval = helo_interval
        self.cleanup_interval = cleanup_interval

        # Available agents: {agent_name: {"description": str, "last_seen": datetime}}
        self.available_agents: Dict[str, Dict[str, Any]] = {}

        # Agent colors: {agent_name: color}
        self.agent_colors: Dict[str, str] = {}

        # Initialize message handlers dictionary: {MessageType: handler_func}
        self.message_handlers: Dict[MessageType, Callable[[Any, Any], None]] = {}

        # Initialize asyncio tasks
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = True

        # Assign a color to this agent for consistent messaging
        self.agent_colors[self.name] = next(self.color_cycle)

        # Attention can be used to determine if messages are for "ALL" or a specific agent
        self.attention = "ALL"

        # Create an asyncio Queue to store incoming messages for dispatch
        self.message_queue = asyncio.Queue()

        # Register default handlers for HELO and ACK message types
        self.register_message_handler(MessageType.HELO, handle_helo_message)
        self.register_message_handler(MessageType.ACK, handle_ack_message)

    async def handle_message(self, message: Message) -> Any:
        """
        Handle an incoming message received by the agent.
        Logs the message and places it into the message_queue for dispatch.
        """
        logging.info(f"Agent '{self.name}' received message: {message}")
        await self.message_queue.put(message)

    def register_message_handler(
        self, 
        message_type: MessageType, 
        handler: Callable[[Any, Any], None]
    ):
        """
        Registers a handler function for a specific message type.

        Args:
            message_type (MessageType): The message type for which to register the handler.
            handler (Callable[[agent, message], None]): The handler function that accepts
                (agent, message) if you want to pass 'self' along with the message.
        """
        self.message_handlers[message_type] = handler
        logging.info(f"Handler registered for message type: {message_type}")

    async def message_dispatcher(self):
        """
        Continuously retrieves messages from self.message_queue and invokes the registered handlers.
        """
        while self.running:
            try:
                msg = await self.message_queue.get()  # Pull next incoming message
                handler = self.message_handlers.get(msg.message_type)

                if handler:
                    # Pass both `self` and `msg` to match the (agent, message) signature
                    await handler(self, msg)
                else:
                    logging.warning(
                        f"Agent '{self.name}' received unhandled message type: {msg.message_type}."
                    )

                self.message_queue.task_done()

            except asyncio.CancelledError:
                logging.info(f"Agent '{self.name}' message dispatcher task cancelled.")
                break
            except Exception as e:
                logging.error(
                    f"Agent '{self.name}' encountered an error in message_dispatcher "
                    f"while handling {msg}: {e}",
                    exc_info=True
                )

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
        Periodically removes agents from available_agents that haven't been seen within the TTL.
        """
        while self.running:
            try:
                current_time = datetime.now()
                expired_agents = [
                    agent_name for agent_name, info in self.available_agents.items()
                    if current_time - info["last_seen"] > self.ttl
                ]
                for agent_name in expired_agents:
                    del self.available_agents[agent_name]
                    if agent_name in self.agent_colors:
                        del self.agent_colors[agent_name]
                    logging.info(
                        f"Agent '{agent_name}' expired and removed from available_agents."
                    )
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
        Starts the agent by initiating background tasks, including:
          - The message_dispatcher loop
          - HELO broadcasts
          - Cleanup for available agents
        """
        logging.info(f"Agent '{self.name}' starting.")

        # Launch the dispatcher to handle incoming messages
        self.add_task("message_dispatcher", self.message_dispatcher())

        # Start periodic tasks
        self.add_task("helo_broadcast", self.helo_broadcast_task())
        self.add_task("cleanup_available_agents", self.cleanup_available_agents_task())

        # Await the main run logic (subclasses can override run())
        await self.run()

    @abstractmethod
    async def run(self):
        """
        The main run loop of the agent. Should be overridden by subclasses
        to implement specific behaviors. This method is awaited in start().
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