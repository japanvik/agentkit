# human_agent.py

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

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
        config: Dict[str, Any],
        brain: Optional['SimpleBrain'] = None,
        memory: Optional['Memory'] = None,
        message_sender: Optional[MessageSender] = None
    ):
        """
        Initialize the human agent.
        
        Args:
            name: Agent's name
            config: Configuration dictionary
            brain: Optional brain component (not used by HumanAgent)
            memory: Optional memory component (not used by HumanAgent)
            message_sender: Optional message sender for delegating communication
        """
        # Initialize base agent
        super().__init__(
            name=name,
            config=config,
            brain=brain,
            memory=memory,
            message_sender=message_sender
        )

        # Register a custom chat handler for human agents
        self.register_message_handler(MessageType.CHAT, self.handle_chat_message)

    async def start(self) -> None:
        """Start the agent's background tasks."""
        await super().start()
        
        # Create and start user input task
        self._tasks.append(asyncio.create_task(
            self.handle_user_input(),
            name=f"{self.name}-user-input"
        ))
        logging.info(f"Started user input task for {self.name}")

    async def handle_chat_message(self, message: Message) -> None:
        """
        Handle incoming CHAT messages by printing them with color coding.

        Args:
            message (Message): The incoming message object.
        """
        logging.info(f"Human Agent '{self.name}' received CHAT message: {message.content}")
        try:
            sender = message.source
            content = message.content
            logging.debug("A. Attempting to get color")

            # Default color scheme
            agent_colors = {
                self.name: "cyan",
                "System": "yellow",
                "Error": "red"
            }
            color = agent_colors.get(sender, "cyan")
            logging.debug(f"B. got {color} as agent color")

            formatted_message = f"[bold {color}]{sender}[/bold {color}]: {content}"
            logging.debug(f"C. message is {formatted_message}")

            loop = asyncio.get_event_loop()
            # Offload the blocking print to the executor
            logging.debug("D. Attempting to print the formatted message to console...")
            await loop.run_in_executor(executor, rich_console.print, formatted_message)

        except Exception as e:
            logging.error(f"Error handling CHAT message from {sender}: {e}")

    async def handle_user_input(self):
        """
        Handles user input from the console, allowing sending messages to agents.
        Supports @ALL for broadcasting and comma-separated @Agent1,@Agent2 for multiple targets.
        Also supports commands like /ls.
        """
        logging.info(f"User input handler started for {self.name}")
        while self._running:
            try:
                loop = asyncio.get_event_loop()
                # Offload the blocking Prompt.ask to the executor
                prompt = f"[bold cyan]{self.name}[/bold cyan]> "
                user_input = await loop.run_in_executor(executor, Prompt.ask, prompt)
                user_input = user_input.strip()

                if not user_input:
                    continue

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
                    # Send to each target
                    for target in targets:
                        message = Message(
                            source=self.name,
                            to=target.upper() if target.upper() == "ALL" else target,
                            content=content,
                            message_type=MessageType.CHAT
                        )
                        await self.send_message(message)
                else:
                    # Default to broadcast if no target specified
                    message = Message(
                        source=self.name,
                        to="ALL",
                        content=user_input,
                        message_type=MessageType.CHAT
                    )
                    await self.send_message(message)

            except Exception as e:
                logging.error(f"Error handling user input: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def list_available_agents(self):
        """
        Displays the list of currently available agents.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, rich_console.print, "\n[bold underline]Available Agents:[/bold underline]")
        await loop.run_in_executor(
            executor,
            rich_console.print,
            f"[bold cyan]{self.name}[/bold cyan]: Human Agent"
        )

    def is_intended_for_me(self, message: Message) -> bool:
        """
        Determine if an incoming message is directed to this agent.

        Args:
            message (Message): The message to check.

        Returns:
            bool: True if the message is intended for this agent, False otherwise.
        """
        logging.debug(f"Checking if message is intended for {self.name}: {message}")
        for_me = (message.to == self.name or message.to == "ALL")
        chat_by_me = (message.source == self.name and message.message_type == MessageType.CHAT)
        not_my_helo = (message.source != self.name and message.message_type == MessageType.HELO)
        return for_me or not_my_helo or chat_by_me

    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        await super().stop()
        executor.shutdown(wait=True)
