# agentkit/agents/simple_agent.py

import asyncio
import logging
from typing import Any

from networkkit.network import MessageSender
from networkkit.messages import Message, MessageType
from agentkit.agents.base_agent import BaseAgent


class SimpleAgent(BaseAgent):
    """
    A straightforward agent that inherits from BaseAgent.
    BaseAgent handles:
      - The message queue
      - The dispatcher (message_dispatcher)
      - HELO/ACK registration
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
            message_sender (MessageSender): Used to send messages to the message bus.
            bus_ip (str): The IP address of the bus. Defaults to "127.0.0.1".
            ttl_minutes (int): Time-to-live for agent availability in minutes.
            helo_interval (int): Interval in seconds between sending HELO messages.
            cleanup_interval (int): Interval in seconds for cleaning up expired agents.
            system_prompt (str): Optional system prompt for agent logic.
            user_prompt (str): Optional user prompt for agent logic.
            model (str): Optional model identifier for the agent's logic.
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

        # If needed, you could register more handlers here:
        # self.register_message_handler(MessageType.TYPE, self.my_custom_handler)

    async def send_message(self, message: Message) -> Any:
        """
        Send a message through the message sender component.

        Args:
            message (Message): The message to send.
        """
        try:
            await self.message_sender.send_message(message)
            logging.info(f"Agent '{self.name}' sent {message.message_type} message to {message.to}.")
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
        # Handle if 'to' is this agent's name or "ALL",
        # or if the message is from this agent (for storing chat history).
        for_me = (message.to == self.name or message.to == "ALL")
        chat_by_me = (message.source == self.name and message.message_type == MessageType.CHAT)
        not_my_helo = (message.source != self.name and message.message_type == MessageType.HELO)
        return for_me or not_my_helo or chat_by_me

    async def start(self):
        """
        Start the agent by running its tasks (as defined in the BaseAgent).
        If you have extra logic specific to SimpleAgent, add it here before/after super().start().
        """
        logging.info(f"Agent '{self.name}' starting (SimpleAgent).")
        await super().start()

    async def stop(self):
        """
        Stop the agent by stopping its tasks and setting the running flag to False.
        Ensures proper cleanup of resources including aiohttp sessions.
        """
        logging.info(f"Agent '{self.name}' stopping (SimpleAgent).")
        try:
            # Close any aiohttp sessions if they exist
            if hasattr(self, 'message_sender') and hasattr(self.message_sender, 'session'):
                if not self.message_sender.session.closed:
                    await self.message_sender.session.close()
            
            # Call parent class stop method
            await super().stop()
            
            # Allow a moment for cleanup
            await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error during stop: {e}")

    async def run(self):
        """
        Overriding run() from BaseAgent if needed. 
        Otherwise, BaseAgent's default run loop sleeps indefinitely 
        until stop() is called.
        """
        try:
            while self.running:
                await asyncio.sleep(0.1)  # Reduced sleep time for faster response
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.running = False
