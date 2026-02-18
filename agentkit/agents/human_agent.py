import asyncio
import json
import logging
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

from agentkit.agents.base_agent import BaseAgent
from networkkit.messages import Message, MessageType
from networkkit.network import MessageSender

try:
    from rich.console import Console
    from rich.prompt import Prompt
except ImportError:  # pragma: no cover - fallback when rich is unavailable
    class Console:  # type: ignore[no-redef]
        def __init__(self, soft_wrap: bool = True) -> None:
            _ = soft_wrap

        def print(self, *args: Any, **kwargs: Any) -> None:
            _ = kwargs
            print(*args)

    class Prompt:  # type: ignore[no-redef]
        @staticmethod
        def ask(prompt: str, console: Optional[Console] = None) -> str:
            _ = console
            clean_prompt = re.sub(r"\[[^\]]+\]", "", prompt)
            return input(clean_prompt)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class ParsedInput:
    kind: str
    content: str = ""
    targets: Optional[List[str]] = None
    command: str = ""
    args: str = ""


class HumanAgent(BaseAgent):
    """Terminal-first human agent with chat-room style UX."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        brain: Optional["SimpleBrain"] = None,  # type: ignore[name-defined]
        memory: Optional["Memory"] = None,  # type: ignore[name-defined]
        message_sender: Optional[MessageSender] = None,
    ) -> None:
        super().__init__(
            name=name,
            config=config,
            brain=brain,
            memory=memory,
            message_sender=message_sender,
        )

        if not isinstance(self.config.get("capabilities"), dict):
            self.config["capabilities"] = {}
        self.config["capabilities"].setdefault("role", "human")

        self.console = Console(soft_wrap=True)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.available_agents: Dict[str, Dict[str, Any]] = {}
        self.current_channel = self._normalize_channel(self.config.get("default_channel", "general"))
        self.known_channels: set[str] = {self.current_channel}
        self.last_sender: Optional[str] = None
        self.recent_messages: deque[str] = deque(maxlen=int(self.config.get("history_size", 200)))

        self.register_message_handler(MessageType.CHAT, self.handle_chat_message)
        self.register_message_handler(MessageType.HELO, self.handle_helo_message)
        self.register_message_handler(MessageType.ACK, self.handle_ack_message)

    async def start(self) -> None:
        await super().start()
        self._print_system(
            f"Connected as {self.name}. Channel: #{self.current_channel}. Type /help for commands."
        )
        self.create_background_task(self.handle_user_input(), name=f"{self.name}-user-input")

    async def stop(self) -> None:
        await super().stop()
        self.executor.shutdown(wait=True)

    async def handle_chat_message(self, message: Message) -> None:
        sender = message.source
        content = message.content
        self.last_sender = sender

        if sender != self.name:
            self._record_agent_identity(sender, None)

        channel, stripped = self._extract_channel(content)
        if channel:
            self.known_channels.add(channel)

        ts = datetime.now().strftime("%H:%M:%S")
        direction = "-> you" if message.to == self.name else f"-> {message.to}"
        channel_label = f" [#{channel}]" if channel else ""
        line = f"[{ts}] {sender}{channel_label} {direction}: {stripped}"
        self.recent_messages.append(line)

        color = "cyan" if sender == self.name else "green"
        if message.to == self.name:
            color = "magenta"
        if sender == "System":
            color = "yellow"

        await self._print_async(f"[bold {color}]{line}[/bold {color}]")

    async def handle_user_input(self) -> None:
        while self._running:
            try:
                prompt = f"[bold cyan]{self.name}[/bold cyan] [#${self.current_channel}]> ".replace("#$", "#")
                raw = await self._prompt_async(prompt)
                parsed = self.parse_user_input(raw)

                if parsed.kind == "empty":
                    continue

                if parsed.kind == "command":
                    should_continue = await self._run_command(parsed.command, parsed.args)
                    if not should_continue:
                        break
                    continue

                if parsed.kind == "error":
                    await self._print_async(f"[bold red]{parsed.content}[/bold red]")
                    continue

                for outgoing in self.build_outbound_messages(parsed):
                    await self.send_message(outgoing)
            except EOFError:
                self._print_system("Input stream closed. Stopping human agent.")
                self._running = False
                break
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.error("Error handling user input: %s", exc, exc_info=True)
                await asyncio.sleep(0.2)

    def parse_user_input(self, text: str) -> ParsedInput:
        user_input = text.strip()
        if not user_input:
            return ParsedInput(kind="empty")

        if user_input.startswith("/"):
            command_blob = user_input[1:]
            command, _, args = command_blob.partition(" ")
            return ParsedInput(kind="command", command=command.lower(), args=args.strip())

        legacy = re.match(r"^@([^ ]+)\s+(.*)", user_input)
        if legacy:
            target_str = legacy.group(1)
            content = legacy.group(2).strip()
            targets = [t.strip() for t in target_str.split(",") if t.strip()]
            if not targets:
                return ParsedInput(kind="error", content="No valid targets found.")
            return ParsedInput(kind="message", content=content, targets=targets)

        return ParsedInput(kind="message", content=user_input)

    def build_outbound_messages(self, parsed: ParsedInput) -> List[Message]:
        targets = parsed.targets or ["ALL"]
        payload = parsed.content

        if targets == ["ALL"] and not self._has_channel_prefix(payload):
            payload = self._add_channel_prefix(payload, self.current_channel)

        messages: List[Message] = []
        for target in targets:
            normalized_target = target.upper() if target.upper() == "ALL" else target
            messages.append(
                Message(
                    source=self.name,
                    to=normalized_target,
                    content=payload,
                    message_type=MessageType.CHAT,
                )
            )
        return messages

    async def _run_command(self, command: str, args: str) -> bool:
        if command in {"quit", "exit"}:
            self._print_system("Exiting chat loop.")
            self._running = False
            return False

        if command in {"help", "h", "?"}:
            await self._print_help()
            return True

        if command in {"agents", "ls"}:
            await self.list_available_agents()
            return True

        if command == "channels":
            await self._print_channels()
            return True

        if command == "history":
            await self._print_history(args)
            return True

        if command == "join":
            if not args:
                await self._print_async("[bold red]Usage: /join <channel>[/bold red]")
                return True
            channel = self._normalize_channel(args)
            self.current_channel = channel
            self.known_channels.add(channel)
            self._print_system(f"Joined #{channel}")
            return True

        if command == "focus":
            if not args or args.lower() in {"none", "clear"}:
                self.attention = None
                self._print_system("Focus cleared")
                return True
            self.attention = args.strip()
            self._print_system(f"Focused on {self.attention}")
            return True

        if command == "reply":
            if not self.last_sender:
                await self._print_async("[bold red]No sender to reply to yet.[/bold red]")
                return True
            if not args:
                await self._print_async("[bold red]Usage: /reply <message>[/bold red]")
                return True
            parsed = ParsedInput(kind="message", content=args, targets=[self.last_sender])
            for outgoing in self.build_outbound_messages(parsed):
                await self.send_message(outgoing)
            return True

        if command in {"dm", "msg", "to"}:
            target_blob, content = self._split_target_and_content(args)
            if not target_blob or not content:
                await self._print_async("[bold red]Usage: /dm <agent[,agent2]> <message>[/bold red]")
                return True
            parsed = ParsedInput(
                kind="message",
                content=content,
                targets=[part.strip() for part in target_blob.split(",") if part.strip()],
            )
            for outgoing in self.build_outbound_messages(parsed):
                await self.send_message(outgoing)
            return True

        if command == "all":
            if not args:
                await self._print_async("[bold red]Usage: /all <message>[/bold red]")
                return True
            parsed = ParsedInput(kind="message", content=args, targets=["ALL"])
            for outgoing in self.build_outbound_messages(parsed):
                await self.send_message(outgoing)
            return True

        await self._print_async(f"[bold red]Unknown command: /{command}[/bold red]")
        return True

    async def handle_helo_message(self, message: Message) -> None:
        self._record_agent_identity(message.source, message.content)
        await self._acknowledge_helo(message)

    async def handle_ack_message(self, message: Message) -> None:
        self._record_agent_identity(message.source, message.content)

    async def list_available_agents(self) -> None:
        await self._print_async("\n[bold underline]Available Agents[/bold underline]")
        if not self.available_agents:
            await self._print_async("[italic]No agents discovered yet.[/italic]")
            return

        for agent_name, info in sorted(self.available_agents.items()):
            description = info.get("description") or "No description"
            last_seen = info.get("last_seen", "unknown")
            await self._print_async(f"[bold cyan]{agent_name}[/bold cyan] ({last_seen}): {description}")

    def is_intended_for_me(self, message: Message) -> bool:
        for_me = message.to in {self.name, "ALL"}
        chat_by_me = message.source == self.name and message.message_type == MessageType.CHAT
        not_my_helo = message.source != self.name and message.message_type == MessageType.HELO
        if for_me or chat_by_me or not_my_helo:
            return True
        if self._attention and message.source == self._attention:
            return True
        return False

    def _record_agent_identity(self, agent_name: str, raw_content: Optional[str]) -> None:
        description: Optional[str] = None
        capabilities: Dict[str, Any] = {}
        if raw_content:
            try:
                payload = json.loads(raw_content)
                description = payload.get("description") or payload.get("about")
                capabilities = payload.get("capabilities") or {}
                agent_name = payload.get("name") or agent_name
            except (json.JSONDecodeError, TypeError):
                description = raw_content

        self.available_agents[agent_name] = {
            "description": description or "No description provided",
            "capabilities": capabilities,
            "last_seen": datetime.utcnow().isoformat(),
        }

    async def _print_help(self) -> None:
        lines = [
            "[bold underline]Commands[/bold underline]",
            "/help - Show this help",
            "/agents (or /ls) - List discovered agents",
            "/channels - List known channels",
            "/join <channel> - Switch default channel",
            "/all <message> - Broadcast to all agents",
            "/dm <agent[,agent2]> <message> - Direct message",
            "/reply <message> - Reply to the last sender",
            "/focus <agent|none> - Follow one sender even if not addressed",
            "/history [N] - Show recent timeline (default 20)",
            "/quit - Exit the human input loop",
            "",
            "Message shortcuts:",
            "@Agent hello - Direct message (legacy compatibility)",
            "plain text - Broadcast to current channel",
            "",
            "Japanese input is supported. Type naturally (e.g., こんにちは、状況を教えて).",
        ]
        for line in lines:
            await self._print_async(line)

    async def _print_channels(self) -> None:
        channels = ", ".join(f"#{name}" for name in sorted(self.known_channels))
        await self._print_async(f"Known channels: {channels}")

    async def _print_history(self, arg: str) -> None:
        count = 20
        if arg:
            try:
                count = max(1, int(arg))
            except ValueError:
                await self._print_async("[bold red]Usage: /history [N][/bold red]")
                return

        await self._print_async("\n[bold underline]Recent Messages[/bold underline]")
        if not self.recent_messages:
            await self._print_async("[italic]No messages yet.[/italic]")
            return

        for line in list(self.recent_messages)[-count:]:
            await self._print_async(line)

    @staticmethod
    def _split_target_and_content(args: str) -> tuple[str, str]:
        blob = args.strip()
        if not blob:
            return "", ""
        target, _, content = blob.partition(" ")
        return target.strip(), content.strip()

    @staticmethod
    def _normalize_channel(raw: str) -> str:
        channel = raw.strip()
        if channel.startswith("#"):
            channel = channel[1:]
        channel = re.sub(r"\s+", "-", channel)
        return channel.lower() or "general"

    @staticmethod
    def _has_channel_prefix(content: str) -> bool:
        return bool(re.match(r"^\s*\[#[-a-zA-Z0-9_]+\]\s*", content))

    @staticmethod
    def _add_channel_prefix(content: str, channel: str) -> str:
        return f"[#{channel}] {content}"

    @staticmethod
    def _extract_channel(content: str) -> tuple[Optional[str], str]:
        match = re.match(r"^\s*\[#([-a-zA-Z0-9_]+)\]\s*(.*)$", content, flags=re.DOTALL)
        if not match:
            return None, content
        return match.group(1).lower(), match.group(2)

    async def _prompt_async(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        ask = partial(Prompt.ask, prompt, console=self.console)
        return await loop.run_in_executor(self.executor, ask)

    async def _print_async(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self.console.print, text)

    def _print_system(self, text: str) -> None:
        self.console.print(f"[bold yellow][system][/bold yellow] {text}")
