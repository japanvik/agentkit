import asyncio
import logging
import requests
from typing import Any
from agentkit.messages import Message, MessageType
from agentkit.network import MessageSender
from agentkit.handlers import default_handle_helo_message

class SimpleAgent:
    def __init__(self, config:dict, message_sender: MessageSender, name:str="", description:str=""):
        self.config:dict = config
        self.message_sender: MessageSender = message_sender
        # Get these from instatiation or config
        self.name:str = self.get_config_value("name", name)
        self.description:str = self.get_config_value("description", description)

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

    def get_config_value(self, parameter_name, override):
        # 
        if override:
            return override
        else:
            if parameter_name in self.config['agent'].keys():
                return self.config['agent'][parameter_name]
            else:
                raise ValueError(f'Required Parameter "{parameter_name}" is not defined in the config file or instance creation')

    def add_task(self, task_name, coro):
        """Add a new task and start it."""
        if task_name in self.tasks:
            self.remove_task(task_name)  # Cancel and remove existing task with the same name
        task = asyncio.create_task(coro)
        logging.info(f"Starting task:{task_name}")
        self.tasks[task_name] = task

    async def remove_task(self, task_name):
        """Cancel and remove a task."""
        task = self.tasks.pop(task_name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # Task cancellation is expected
    
    def add_message_handler(self, message_type:MessageType, handler) -> None:
        self.message_handlers[message_type] = handler
        logging.info(f"Added handler for message type: {message_type}")
        return handler
            
    async def stop_all_tasks(self):
        """Stop all running tasks."""
        for task_name in list(self.tasks.keys()):
            await self.remove_task(task_name)

    async def handle_message(self, message: Message) -> Any:
        logging.info(f"Received message: {message}")
        await self.message_queue.put(message)
    
    async def message_dispatcher(self):
        while self.running:
            message = await self.message_queue.get()
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(self, message)
                await asyncio.sleep(0.1)
            self.message_queue.task_done()        
            
    def is_intended_for_me(self, message: Message) -> bool:
        # Subscriber Implementation
        for_me = message.to == self.name or message.to == "ALL"
        chat_by_me = message.source == self.name and message.message_type == "CHAT" # for storing as history
        not_my_helo = message.source != self.name and message.message_type == "HELO"
        return for_me or not_my_helo or chat_by_me
        
    def send_helo(self)->requests.Response:
        """Send the HELO message"""
        msg = Message(source=self.name, to="ALL", content=self.description, message_type=MessageType.HELO)
        return self.send_message(msg)
   
    def send_message(self, message: Message):
        return self.message_sender.send_message(message) 
    
    async def start(self):
        while self.running:
            await asyncio.sleep(1)
        await self.stop()

    async def stop(self):
        await self.stop_all_tasks()
        self.running = False
