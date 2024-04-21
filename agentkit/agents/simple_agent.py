import asyncio
import logging
import requests
from typing import Any
from agentkit.messages import Message, MessageType
from agentkit.network import MessageSender
from agentkit.handlers import default_handle_helo_message


class SimpleAgent:
    """
    Core agent class implementing message handling, dispatching, and interaction logic.

    This class represents an agent within the system. It manages message reception, routing, task execution, and interaction
    with other agents through message sending. The agent relies on a configuration dictionary for initialization and utilizes
    a message queue for asynchronous message processing.

    The agent can register message handlers for specific message types to handle incoming messages appropriately.
    """

    def __init__(self, config: dict, message_sender: MessageSender, name: str = "", description: str = ""):
        """
        Constructor for the SimpleAgent class.

        Args:
            config (dict): Configuration dictionary containing agent-specific settings.
            message_sender (MessageSender): An instance of a message sender component for sending messages.
            name (str, optional): The name of the agent. Defaults to "".
            description (str, optional): A description of the agent. Defaults to "".
        """

        self.config: dict = config
        self.message_sender: MessageSender = message_sender
        # Get these from instatiation or config
        self.name: str = self.get_config_value("name", name)
        self.description: str = self.get_config_value("description", description)

        self.message_queue = asyncio.Queue()
        self.running = True
        self.tasks = {}
        self.message_handlers = {
            MessageType.HELO: default_handle_helo_message
        }
        self.attention: str = "ALL"
        # Add default tasks
        self.add_task("message_dispatcher", self.message_dispatcher())
        self.send_helo()

    def get_config_value(self, parameter_name: str, override: str) -> str:
        """
        Retrieve a configuration value from the agent's configuration dictionary.

        This method checks for the provided parameter name in the 'agent' section of the configuration dictionary.
        If an override value is provided, it returns the override. Otherwise, it raises a ValueError if the parameter is not found.

        Args:
            parameter_name (str): The name of the configuration parameter to retrieve.
            override (str, optional): An optional override value for the parameter. Defaults to "".

        Returns:
            str: The retrieved configuration value or the provided override value.

        Raises:
            ValueError: If the parameter is not found in the configuration and no override is provided.
        """

        if override:
            return override
        else:
            if parameter_name in self.config['agent'].keys():
                return self.config['agent'][parameter_name]
            else:
                raise ValueError(f'Required Parameter "{parameter_name}" is not defined in the config file or instance creation')

    def add_task(self, task_name: str, coro):
        """
        Add a new coroutine as a task to the agent and start its execution.

        This method adds the provided coroutine (asynchronous function) to the agent's task list with a specified name.
        If a task with the same name already exists, it removes it before adding the new one.

        Args:
            task_name (str): The name to associate with the new task.
            coro: The coroutine (asynchronous function) to be executed as a task.
        """

        if task_name in self.tasks:
            self.remove_task(task_name)  # Cancel and remove existing task with the same name
        task = asyncio.create_task(coro)
        logging.info(f"Starting task:{task_name}")
        self.tasks[task_name] = task

    async def remove_task(self, task_name: str):
        """
        Cancel and remove a running task from the agent.

        This method cancels the specified task by name and removes it from the agent's task list.
        It attempts to wait for the task to finish after cancellation but gracefully handles cancellation exceptions.

        Args:
            task_name (str): The name of the task to be removed.
        """

        task = self.tasks.pop(task_name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # Task cancellation is expected

    def add_message_handler(self, message_type: MessageType, handler) -> None:
        """
        Register a message handler function for a specific message type.

        This method associates a handler function with a particular message type. The handler function will be invoked
        whenever a message of that type is received by the agent.

        Args:
            message_type (MessageType): The message type for which to register the handler.
            handler: The function to be called when a message of the specified type is received.
        """

        self.message_handlers[message_type] = handler
        logging.info(f"Added handler for message type: {message_type}")
        return handler

    async def stop_all_tasks(self):
        """
        Stop all currently running tasks associated with the agent.

        This method iterates through the tasks and cancels them one by one. It waits for each task to finish
        after cancellation but handles cancellation exceptions gracefully.
        """

        for task_name in list(self.tasks.keys()):
            await self.remove_task(task_name)

    async def handle_message(self, message: Message) -> Any:
        """
        Handle an incoming message received by the agent.

        This method adds the received message to the agent's message queue for asynchronous processing.

        Args:
            message (Message): The received message object.
        """

        logging.info(f"Received message: {message}")
        await self.message_queue.put(message)

    async def message_dispatcher(self):
        """
        Continuously retrieve messages from the queue and dispatch them to the appropriate handlers.

        This method runs as an asynchronous task and continuously checks the message queue for new messages.
        When a message is received, it retrieves the corresponding handler function based on the message type
        and calls the handler with the message object. The dispatcher introduces a short delay between handling messages.

        The message queue is processed as long as the agent is running.
        """

        while self.running:
            message = await self.message_queue.get()
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(self, message)
                await asyncio.sleep(0.1)
            self.message_queue.task_done()

    def is_intended_for_me(self, message: Message) -> bool:
        """
        Determine if an incoming message is directed to this agent.

        This method checks various conditions to determine if the received message is intended for this specific agent.
        A message is considered intended for this agent if:

        * The message target (`message.to`) is either the agent's name or "ALL".
        * The message is sent by the agent itself (`message.source == self.name`) and is of type CHAT (for history storage).
        * The message is of type HELO and not sent by the agent itself (`message.source != self.name and message.message_type == "HELO"`).

        Args:
            message (Message): The message to check.

        Returns:
            bool: True if the message is intended for this agent, False otherwise.
        """

        for_me = message.to == self.name or message.to == "ALL"
        chat_by_me = message.source == self.name and message.message_type == "CHAT"  # for storing as history
        not_my_helo = message.source != self.name and message.message_type == "HELO"
        return for_me or not_my_helo or chat_by_me

    def send_helo(self) -> requests.Response:
        """
        Send a HELO message upon agent startup.

        This method creates a HELO message with the agent's description and sends it to all recipients ("ALL").

        Returns:
            requests.Response: The response object from the message sender after sending the HELO message.
        """

        msg = Message(source=self.name, to="ALL", content=self.description, message_type=MessageType.HELO)
        return self.send_message(msg)

    def send_message(self, message: Message):
        """
        Send a message through the message sender component.

        This method takes a message object and delegates the sending process to the message sender component.

        Args:
            message (Message): The message to send.

        Returns:
            requests.Response: The response object from the message sender after sending the message.
        """
        return self.message_sender.send_message(message)

    async def start(self):
        """
        Start the agent by running its tasks.

        This method initiates the execution of all the agent's tasks, effectively starting the agent's operation.
        The agent continues to run until the `stop` method is called.
        """

        while self.running:
            await asyncio.sleep(1)
        await self.stop()

    async def stop(self):
        """
        Stop the agent by stopping its tasks and setting the running flag to False.

        This method gracefully stops the agent by canceling all its running tasks using `stop_all_tasks`.
        It then sets the `running` flag to False to indicate that the agent should stop processing messages.
        """

        await self.stop_all_tasks()
        self.running = False
