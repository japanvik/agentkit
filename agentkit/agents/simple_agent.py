# agentkit/agents/simple_agent.py

import asyncio
import logging
from typing import Any, Callable, Dict

from networkkit.network import MessageSender
from networkkit.messages import Message, MessageType
from agentkit.agents.base_agent import BaseAgent
from agentkit.handlers import default_handle_helo_message


class SimpleAgent(BaseAgent):
    """
    Core agent class implementing message handling, dispatching, and interaction logic.
    """

    def __init__(self, name: str, description: str, message_sender: MessageSender):
        """
        Constructor for the SimpleAgent class.

        Args:
            name (str): The name of the agent.
            description (str): A description of the agent.
            message_sender (MessageSender): An instance of a message sender component 
              for sending messages.
        """
        super().__init__(name=name, description=description, system_prompt="", user_prompt="", model="")  # Initialize BaseAgent
        self.message_sender = message_sender

        self.message_queue = asyncio.Queue()
        self.running = True
        self.tasks: Dict[str, asyncio.Task] = {}
        self.message_handlers: Dict[MessageType, Callable[[Message], Any]] = {
            MessageType.HELO: default_handle_helo_message
        }
        self.attention: str = "ALL"

        # Add default tasks
        self.add_task("message_dispatcher", self.message_dispatcher())

        # Send HELO message upon startup
        self.send_helo()

    def add_task(self, task_name: str, coro):
        """
        Add a new coroutine as a task to the agent and start its execution.

        Args:
            task_name (str): The name to associate with the new task.
            coro: The coroutine (asynchronous function) to be executed as a task.
        """
        if task_name in self.tasks:
            self.remove_task(task_name)  # Cancel and remove existing task with the same name
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

    def add_message_handler(self, message_type: MessageType, handler: Callable[[Message], Any]) -> None:
        """
        Register a message handler function for a specific message type.

        Args:
            message_type (MessageType): The message type for which to register the handler.
            handler (Callable[[Message], Any]): The handler function.
        """
        self.message_handlers[message_type] = handler
        logging.info(f"Added handler for message type: {message_type.name}")
        return handler

    async def stop_all_tasks(self):
        """
        Stop all currently running tasks associated with the agent.
        """
        for task_name in list(self.tasks.keys()):
            await self.remove_task(task_name)

    async def handle_message(self, message: Message) -> Any:
        """
        Handle an incoming message received by the agent.
        """
        logging.info(f"Agent '{self.name}' received message: {message}")
        await self.message_queue.put(message)

    async def message_dispatcher(self):
        """
        Continuously retrieve messages from the queue and dispatch them to the appropriate handlers.
        """
        while self.running:
            try:
                message = await self.message_queue.get()
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    await handler(self, message)  # Pass 'self' as the agent
                else:
                    logging.warning(f"Agent '{self.name}' received unhandled message type: {message.message_type}.")
                await asyncio.sleep(0.1)  # Prevent tight loop
                self.message_queue.task_done()
            except asyncio.CancelledError:
                logging.info(f"Agent '{self.name}' message dispatcher task cancelled.")
                break
            except Exception as e:
                logging.error(f"Agent '{self.name}' encountered an error in message_dispatcher[{message}]: {e}")

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

    def send_helo(self) -> Any:
        """
        Send a HELO message upon agent startup.
        """
        msg = Message(source=self.name, to="ALL", content=self.description, message_type=MessageType.HELO)
        return self.send_message(msg)

    def send_message(self, message: Message) -> Any:
        """
        Send a message through the message sender component.
        """
        return self.message_sender.send_message(message)

    async def start(self):
        """
        Start the agent by running its tasks.
        """
        logging.info(f"Agent '{self.name}' started.")
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        """
        Stop the agent by stopping its tasks and setting the running flag to False.
        """
        logging.info(f"Agent '{self.name}' stopping...")
        self.running = False
        await self.stop_all_tasks()
        logging.info(f"Agent '{self.name}' stopped.")
