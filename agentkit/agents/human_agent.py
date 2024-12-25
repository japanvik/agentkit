# human_agent.py

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from agentkit.agents.base_agent import BaseAgent
from networkkit.network import MessageSender
from networkkit.messages import MessageType, Message
from rich.console import Console
from rich.prompt import Prompt

# Initialize Rich Console
rich_console = Console()
# Initialize a ThreadPoolExecutor for handling blocking I/O
executor = ThreadPoolExecutor(max_workers=2)

class HumanAgent(BaseAgent):
    """
    HumanAgent is a subclass of BaseAgent tailored for human interaction.
    It manages user input, supports color-coded conversations, and enhanced target selection.
    """

    def __init__(
        self,
        name: str,
        description: str,
        message_sender: MessageSender,
        bus_ip: str = "127.0.0.1",
        ttl_minutes: int = 5,
        helo_interval: int = 300,  # seconds
        cleanup_interval: int = 300  # seconds
    ):

        super().__init__(
            name=name, 
            description=description, 
            message_sender=message_sender, 
            bus_ip=bus_ip,
            ttl_minutes=ttl_minutes,
            helo_interval=helo_interval,
            cleanup_interval=cleanup_interval
        )

        # Initialize message queue (if not handled by BaseAgent)
        self.message_queue = asyncio.Queue()

        # Initialize additional tasks
        self.add_task("user_input", self.handle_user_input())

    async def handle_chat_message(self, message: Message):
        """
        Handles incoming CHAT messages by printing them with color coding.
        """
        logging.info(f"Agent '{self.name}' received CHAT message: {message.content}")
        try:
            sender = message.source
            content = message.content
            color = self.agent_colors.get(sender, "white")
            formatted_message = f"[bold {color}]{sender}[/bold {color}]: {content}"
            loop = asyncio.get_event_loop()
            # Offload the blocking print to the executor
            await loop.run_in_executor(executor, rich_console.print, formatted_message)
        except Exception as e:
            logging.error(f"Error handling CHAT message from {sender}: {e}")

    async def handle_user_input(self):
        """
        Handles user input from the console, allowing sending messages to agents.
        Supports @ALL for broadcasting and comma-separated @Agent1,@Agent2 for multiple targets.
        Also supports commands like /ls.
        """
        while self.running:
            try:
                loop = asyncio.get_event_loop()
                # Offload the blocking Prompt.ask to the executor
                user_input = await loop.run_in_executor(executor, Prompt.ask, "[bold blue]You[/bold blue]")
                user_input = user_input.strip()

                # Check for commands
                if user_input.startswith("/"):
                    command = user_input[1:].lower()
                    if command == "ls":
                        await self.list_available_agents()
                    else:
                        # Offload the blocking print to the executor
                        await loop.run_in_executor(executor, rich_console.print, f"[bold red]Unknown command: {command}[/bold red]")
                    continue

                # Regex to parse @ALL or multiple @Agent1,@Agent2
                match = re.match(r'^@([^ ]+)\s+(.*)', user_input)
                if match:
                    target_str = match.group(1)
                    content = match.group(2)
                    # Split targets by comma if multiple
                    targets = [t.strip() for t in target_str.split(',')]
                    # Validate targets
                    valid_targets = []
                    for target in targets:
                        if target.upper() == "ALL" or target in self.available_agents:
                            valid_targets.append(target)
                        else:
                            # Offload the blocking print to the executor
                            await loop.run_in_executor(executor, rich_console.print, f"[bold red]Error: Unknown or unavailable agent '{target}'.[/bold red]")
                    if not valid_targets:
                        continue
                else:
                    # Default target if none specified
                    targets = ["ALL"]
                    content = user_input

                # Create and send messages
                for target in targets:
                    message = Message(
                        source=self.name,
                        to=target,
                        content=content,
                        message_type=MessageType.CHAT
                    )
                    await self.send_message(message)
            except Exception as e:
                logging.error(f"Error handling user input: {e}")

    async def list_available_agents(self):
        """
        Displays the list of currently available agents with their descriptions.
        """
        loop = asyncio.get_event_loop()
        if not self.available_agents:
            await loop.run_in_executor(executor, rich_console.print, "[bold yellow]No agents are currently available.[/bold yellow]")
            return

        await loop.run_in_executor(executor, rich_console.print, "\n[bold underline]Available Agents:[/bold underline]")
        for agent_name, info in self.available_agents.items():
            color = self.agent_colors.get(agent_name, "red")
            description = info["description"]
            await loop.run_in_executor(executor, rich_console.print, f"[bold {color}]{agent_name}[/bold {color}]: {description}")

    def is_intended_for_me(self, message: Message) -> bool:
        """
        Determine if an incoming message is directed to this agent.

        Args:
            message (Message): The message to check.

        Returns:
            bool: True if the message is intended for this agent, False otherwise.
        """
        logging.debug(f"Checking if message is intended for {self.name}: {message}")
        for_me = message.to == self.name or message.to == "ALL"
        chat_by_me = message.source == self.name and message.message_type == MessageType.CHAT  # for storing as history
        not_my_helo = message.source != self.name and message.message_type == MessageType.HELO
        return for_me or not_my_helo or chat_by_me

    async def stop(self):
        """
        Override the stop method to shut down the executor.
        """
        await super().stop()
        executor.shutdown(wait=True)
