# agentkit/agents/simple_agent.py

import asyncio
import logging
from typing import Any, Callable, Dict

from networkkit.network import MessageSender
from networkkit.messages import Message, MessageType
from agentkit.agents.base_agent import BaseAgent

class SimpleAgent(BaseAgent):
    """
    Core agent class implementing message handling, dispatching, and interaction logic.
    Inherits from BaseAgent to utilize centralized HELO/ACK functionalities.
    """

    def __init__(
        self, 
        name: str, 
        description: str, 
        message_sender: MessageSender,
        bus_ip: str = "127.0.0.1",
        ttl_minutes: int = 5,
        helo_interval: int = 60,
        cleanup_interval: int = 60,
        system_prompt: str = "",
        user_prompt: str = "",
        model: str = ""
    ):
        """
        Constructor for the SimpleAgent class.

        Args:
            name (str): The name of the agent.
            description (str): A description of the agent.
            message_sender (MessageSender): An instance of a message sender component for sending messages.
            bus_ip (str): The IP address of the bus to subscribe to.
            ttl_minutes (int): Time-to-live for agent availability.
            helo_interval (int): Interval in seconds to send HELO messages.
            cleanup_interval (int): Interval in seconds to clean up expired agents.
            system_prompt (str): System prompt for the agent's brain.
            user_prompt (str): User prompt for the agent's brain.
            model (str): Model identifier for the agent's brain.
        """
        super().__init__(
            name=name,
            description=description,
            message_sender=message_sender,
            bus_ip=bus_ip,
            ttl_minutes=ttl_minutes,
            helo_interval=helo_interval,
            cleanup_interval=cleanup_interval
        )
        
        # Additional attributes specific to SimpleAgent
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

        # Initialize message queue
        self.message_queue = asyncio.Queue()

        # Register additional message handlers if necessary
        # Example: self.register_message_handler(MessageType.INFO, self.handle_info_message)

        # Add the message_dispatcher task
        self.add_task("message_dispatcher", self.message_dispatcher())

    def add_task(self, task_name: str, coro):
        """
        Add a new coroutine as a task to the agent and start its execution.

        Args:
            task_name (str): The name to associate with the new task.
            coro: The coroutine (asynchronous function) to be executed as a task.
        """
        if task_name in self.tasks:
            asyncio.create_task(self.remove_task(task_name))  # Schedule task removal
        task = asyncio.create_task(coro, name=task_name)
        logging.info(f"Starting task: {task_name}")
        self.tasks[task_name] = task

    async def remove_task(self, task_name: str):
        """
        Cancel and remove a running task from the agent.

        Args:
            task_name (str): The name of the task to be removed.
        """
        task = self.tasks.pop(task_name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logging.info(f"Task '{task_name}' cancelled successfully.")

    async def message_dispatcher(self):
        """
        Continuously retrieve messages from the queue and dispatch them to the appropriate handlers.
        """
        while self.running:
            try:
                message = await self.message_queue.get()
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    await handler(agent=self, message=message)
                else:
                    logging.warning(f"Agent '{self.name}' received unhandled message type: {message.message_type}.")
                await asyncio.sleep(0.1)  # Prevent tight loop
                self.message_queue.task_done()
            except asyncio.CancelledError:
                logging.info(f"Agent '{self.name}' message dispatcher task cancelled.")
                break
            except Exception as e:
                logging.error(f"Agent '{self.name}' encountered an error in message_dispatcher[{message}]: {e}")

    async def send_message(self, message: Message) -> Any:
        """
        Send a message through the message sender component.

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
        Determine if an incoming message is directed to this agent.

        Args:
            message (Message): The message to check.

        Returns:
            bool: True if the message is intended for this agent, False otherwise.
        """
        for_me = message.to == self.name or message.to == "ALL"
        chat_by_me = message.source == self.name and message.message_type == MessageType.CHAT  # for storing as history
        not_my_helo = message.source != self.name and message.message_type == MessageType.HELO
        return for_me or not_my_helo or chat_by_me

    async def start(self):
        """
        Start the agent by running its tasks.
        """
        logging.info(f"Agent '{self.name}' started.")
        await super().start()

    async def stop(self):
        """
        Stop the agent by stopping its tasks and setting the running flag to False.
        """
        logging.info(f"Agent '{self.name}' stopping...")
        await super().stop()
