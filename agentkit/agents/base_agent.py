"""
Base agent implementation.

This module provides the BaseAgent class, which serves as the foundation for all agent
implementations in the AgentKit framework. It handles message sending, attention management,
and component configuration.

The BaseAgent class implements the MessageSender interface to provide communication
capabilities to other components through ComponentConfig.
"""
# Standard library imports
import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

# Third-party imports
from networkkit.messages import Message, MessageType

# Local imports
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.common.interfaces import ComponentConfig, MessageSender
from agentkit.memory.memory_protocol import Memory

logger = logging.getLogger(__name__)
T = TypeVar("T")

class BaseAgent(MessageSender):
    """
    Base agent implementation that handles message sending and attention management.
    
    The BaseAgent class provides core functionality for all agent types in the AgentKit
    framework. It manages message handling, component configuration, and attention
    targeting. It serves as a foundation that can be extended to create specialized
    agent implementations.
    
    Implements the MessageSender interface to provide communication capabilities
    to other components through ComponentConfig.
    
    Attributes:
        name (str): The agent's name, used for message addressing
        config (Dict[str, Any]): Configuration dictionary for the agent
        component_config (ComponentConfig): Configuration shared with components
        message_handlers (Dict[MessageType, List[Callable]]): Registered message handlers
        _brain (Optional[SimpleBrain]): The agent's brain component for decision making
        _memory (Optional[Memory]): The agent's memory component for storing history
        _attention (Optional[str]): The current attention target
        _message_sender (Optional[MessageSender]): External message sender if provided
        _running (bool): Flag indicating if the agent is currently running
        _tasks (List[asyncio.Task]): List of background tasks
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        brain: Optional[SimpleBrain] = None,
        memory: Optional[Memory] = None,
        message_sender: Optional[MessageSender] = None
    ) -> None:
        """
        Initialize the agent with name, configuration, and optional components.
        
        This constructor sets up the agent with its basic configuration and optional
        components like brain and memory. It also initializes the internal state
        and creates a ComponentConfig object to share with components.
        
        Args:
            name: Agent's name used for message addressing
            config: Configuration dictionary containing agent settings
            brain: Optional brain component for decision making
            memory: Optional memory component for storing conversation history
            message_sender: Optional message sender for delegating communication
                           (if None, the agent will handle sending itself)
        """
        # Store basic info
        self.name = name
        self.config = config
        
        # Initialize internal state
        self._attention: Optional[str] = None
        self._message_sender = message_sender
        self._running = False
        self._tasks: List[asyncio.Task[Any]] = []
        self._mcp_manager = None
        self._mcp_registry_configured = False
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        
        # Create component config with self as MessageSender
        self.component_config = ComponentConfig(
            agent_name=name,
            config=config,
            message_sender=self
        )
        
        # Store components
        self._brain = brain
        self._memory = memory
        
        # Set config on components if they exist
        if hasattr(self._brain, 'set_config'):
            self._brain.set_config(self.component_config)
        
        if hasattr(self._memory, 'set_config'):
            self._memory.set_config(self.component_config)
    
    @property
    def brain(self) -> Optional[SimpleBrain]:
        """
        Get the brain component.
        
        The brain component is responsible for decision making and generating
        responses to messages.
        
        Returns:
            Optional[SimpleBrain]: The current brain component or None if not set
        """
        return self._brain
    
    @brain.setter
    def brain(self, value: Optional[SimpleBrain]) -> None:
        """
        Set the brain component and its config.
        
        This setter not only assigns the brain component but also configures it
        with the agent's ComponentConfig if the brain has a set_config method.
        
        Args:
            value: The brain component to set or None to clear it
        """
        self._brain = value
        if value is not None and hasattr(value, 'set_config'):
            value.set_config(self.component_config)
    
    @property
    def memory(self) -> Optional[Memory]:
        """
        Get the memory component.
        
        The memory component is responsible for storing and retrieving
        conversation history.
        
        Returns:
            Optional[Memory]: The current memory component or None if not set
        """
        return self._memory
    
    @memory.setter
    def memory(self, value: Optional[Memory]) -> None:
        """
        Set the memory component and its config.
        
        This setter not only assigns the memory component but also configures it
        with the agent's ComponentConfig if the memory has a set_config method.
        
        Args:
            value: The memory component to set or None to clear it
        """
        self._memory = value
        if value is not None and hasattr(value, 'set_config'):
            value.set_config(self.component_config)
    
    @property
    def attention(self) -> Optional[str]:
        """
        Get the current attention target.
        
        The attention target represents the entity this agent is currently focused on
        for communication purposes. Messages from this target will be processed
        even if they are not explicitly addressed to this agent.
        
        Returns:
            Optional[str]: The name of the current attention target or None if not set
        """
        return self._attention
    
    @attention.setter
    def attention(self, value: str) -> None:
        """
        Set the current attention target.
        
        Updates the agent's focus to a specific entity for communication purposes.
        Messages from this entity will be processed even if they are not explicitly
        addressed to this agent.
        
        Args:
            value: The name of the entity to focus attention on
        """
        self._attention = value
    
    async def _internal_send_message(self, message: Message) -> None:
        """
        Internal implementation of message sending, delegating to message_sender if provided.
        
        This method handles the actual message sending, either by delegating to an external
        message sender (if provided during initialization) or by implementing
        the sending logic in subclasses. If no message_sender was provided and
        this method is not overridden in a subclass, the message will not be sent.
        
        Args:
            message: Message object to send
        """
        if self._message_sender:
            await self._message_sender.send_message(message)
    
    async def send_message(self, message: Message) -> None:
        """
        Send a message, using the send_message tool if available, otherwise falling back to internal implementation.
        
        This method provides backward compatibility with existing code that calls send_message directly.
        In the new architecture, sending messages should be done through the functions registry using
        the send_message tool.
        
        Args:
            message: Message object to send
        """
        # For backward compatibility, use the internal implementation
        await self._internal_send_message(message)
    
    def register_tools(self, functions_registry) -> None:
        """
        Register agent tools with the functions registry.
        
        This method registers built-in tools like send_message with the functions registry,
        making them available for use by the agent and its components.
        
        Args:
            functions_registry: The functions registry to register tools with
        """
        from agentkit.functions.built_in_tools import send_message_tool
        from agentkit.functions.execution_tools import (
            python_execution_tool,
            shell_command_tool,
        )
        from agentkit.functions.filesystem_tools import (
            list_directory_tool,
            read_file_tool,
            write_file_tool,
        )
        from agentkit.functions.reminder_tools import schedule_reminder_tool
        from agentkit.functions.delegation_tools import (
            delegate_task_tool,
            escalate_task_tool,
        )
        from agentkit.functions.functions_registry import (
            FunctionDescriptor,
            ParameterDescriptor,
        )
        
        descriptor = FunctionDescriptor(
            name="send_message",
            description="Send a message to another agent or entity",
            parameters=[
                ParameterDescriptor(
                    name="recipient",
                    description="The name of the recipient",
                    required=True
                ),
                ParameterDescriptor(
                    name="content",
                    description="The message content",
                    required=True
                ),
                ParameterDescriptor(
                    name="message_type",
                    description="The type of message (default: CHAT)",
                    required=False
                )
            ]
        )
        
        self._configure_mcp_manager(functions_registry)

        if functions_registry.has_function("send_message"):
            return
        
        functions_registry.register_function(
            send_message_tool,
            descriptor,
            pass_context=True,
        )

        if not functions_registry.has_function("shell_command"):
            descriptor = FunctionDescriptor(
                name="shell_command",
                description="Execute a shell command using bash.",
                parameters=[
                    ParameterDescriptor(
                        name="command",
                        description="Shell command to execute.",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="timeout",
                        description="Maximum execution time in seconds (default 15).",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="working_dir",
                        description="Optional working directory for the command.",
                        required=False,
                    ),
                ],
                categories=["execution"],
            )
            functions_registry.register_function(
                shell_command_tool,
                descriptor,
                pass_context=True,
            )

        if not functions_registry.has_function("schedule_reminder"):
            descriptor = FunctionDescriptor(
                name="schedule_reminder",
                description="Schedule a reminder that sends a message at a future time.",
                parameters=[
                    ParameterDescriptor(
                        name="content",
                        description="Reminder message content",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="recipient",
                        description="Optional reminder recipient (default self)",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="run_at",
                        description="ISO timestamp when the reminder should trigger",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="delay_seconds",
                        description="Delay before reminder triggers, in seconds",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="repeat_seconds",
                        description="If provided, reminder repeats at this interval (seconds)",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="description",
                        description="Optional description for planner tracking",
                        required=False,
                    ),
                ],
                categories=["reminder"],
            )
            functions_registry.register_function(
                schedule_reminder_tool,
                descriptor,
                pass_context=True,
            )

        if not functions_registry.has_function("delegate_task"):
            descriptor = FunctionDescriptor(
                name="delegate_task",
                description="Delegate a task to another agent and schedule follow-up reminders.",
                parameters=[
                    ParameterDescriptor(
                        name="target_agent",
                        description="Name of the agent to delegate to",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="content",
                        description="Instructions for the delegated agent",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="reminder_interval",
                        description="Seconds between reminder pings (optional)",
                        required=False,
                    ),
                ],
                categories=["coordination"],
            )
            functions_registry.register_function(
                delegate_task_tool,
                descriptor,
                pass_context=True,
            )

        if not functions_registry.has_function("escalate_task"):
            descriptor = FunctionDescriptor(
                name="escalate_task",
                description="Escalate a task to a stakeholder when automation cannot proceed.",
                parameters=[
                    ParameterDescriptor(
                        name="reason",
                        description="Explain why escalation is required",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="details",
                        description="Optional additional context",
                        required=False,
                    ),
                ],
                categories=["coordination"],
            )
            functions_registry.register_function(
                escalate_task_tool,
                descriptor,
                pass_context=True,
            )

        if not functions_registry.has_function("list_directory"):
            descriptor = FunctionDescriptor(
                name="list_directory",
                description="List files within the configured filesystem root.",
                parameters=[
                    ParameterDescriptor(
                        name="path",
                        description="Optional directory path relative to root",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="pattern",
                        description="Optional glob pattern",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="recursive",
                        description="Set true to search recursively",
                        required=False,
                    ),
                ],
                categories=["filesystem"],
            )
            functions_registry.register_function(
                list_directory_tool,
                descriptor,
                pass_context=True,
            )

        if not functions_registry.has_function("read_file"):
            descriptor = FunctionDescriptor(
                name="read_file",
                description="Read the contents of a file relative to the filesystem root.",
                parameters=[
                    ParameterDescriptor(
                        name="path",
                        description="Path of the file to read",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="max_bytes",
                        description="Optional maximum bytes to read",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="encoding",
                        description="Text encoding (default utf-8)",
                        required=False,
                    ),
                ],
                categories=["filesystem"],
            )
            functions_registry.register_function(
                read_file_tool,
                descriptor,
                pass_context=True,
            )

        if not functions_registry.has_function("write_file"):
            descriptor = FunctionDescriptor(
                name="write_file",
                description="Write text content to a file within the filesystem root.",
                parameters=[
                    ParameterDescriptor(
                        name="path",
                        description="Path of the file to write",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="content",
                        description="Text content to write",
                        required=True,
                    ),
                    ParameterDescriptor(
                        name="mode",
                        description="'overwrite' (default) or 'append'",
                        required=False,
                    ),
                    ParameterDescriptor(
                        name="encoding",
                        description="Text encoding (default utf-8)",
                        required=False,
                    ),
                ],
                categories=["filesystem"],
            )
            functions_registry.register_function(
                write_file_tool,
                descriptor,
                pass_context=True,
            )
    
    def is_intended_for_me(self, message: Message) -> bool:
        """
        Check if a message is intended for this agent.
        
        A message is considered intended for this agent if any of the following are true:
        1. The message is explicitly addressed to this agent (message.to == self.name)
        2. The message is a broadcast message (message.to == 'ALL')
        3. The message was sent by this agent (message.source == self.name)
        4. The agent is currently paying attention to the sender (self._attention == message.source)
        
        Args:
            message: Message object to check
            
        Returns:
            bool: True if message is intended for this agent, False otherwise
        """
        # Check if message is addressed to this agent or a broadcast
        if message.to in [self.name, 'ALL'] or message.source == self.name:
            return True
            
        # Check if this agent is currently paying attention to the sender
        if self._attention and message.source == self._attention:
            return True
            
        return False
    
    async def handle_message(self, message: Message) -> None:
        """
        Handle an incoming message based on its type.
        
        This method first checks if the message is intended for this agent.
        If it is, it looks for registered handlers for the message type.
        If handlers are found, they are called in order.
        If no handlers are found, default handling is applied based on the message type.
        
        For CHAT messages, the brain component is used to handle the message.
        For HELO messages, an ACK response is sent back to the sender.
        
        Args:
            message: Message object to handle
        
        Raises:
            Exception: Any exception raised by message handlers is caught and logged
        """
        # Only handle messages intended for this agent
        if not self.is_intended_for_me(message):
            return
            
        # Check for registered handlers first
        handlers = self.message_handlers.get(message.message_type, [])
        if handlers:
            for handler in handlers:
                try:
                    await handler(message)
                except Exception:
                    logger.exception("Error in message handler")
            return

        # Default handling if no handlers registered
        if message.message_type == MessageType.CHAT:
            if self.brain:
                await self.brain.handle_chat_message(message)
        elif message.message_type == MessageType.HELO:
            await self._acknowledge_helo(message)
    
    def register_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        This method allows for custom handling of specific message types.
        Multiple handlers can be registered for the same message type,
        and they will be called in the order they were registered.
        
        Args:
            message_type: Type of message to handle (from MessageType enum)
            handler: Async function to handle the message, with signature:
                    async def handler(message: Message) -> None
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def start(self) -> None:
        """
        Start the agent's background tasks.
        
        This method initializes the agent and prepares it for operation.
        If the agent is already running, this method does nothing.
        Subclasses should override this method to start any additional
        background tasks or services.
        """
        if self._running:
            return
            
        self._running = True
        logger.info("Agent %s started", self.name)
        try:
            await self._send_helo()
        except Exception:
            logger.exception("Agent %s failed to broadcast HELO", self.name)
        await self._start_mcp_manager()

    async def _acknowledge_helo(self, message: Message) -> None:
        """
        Reply to an incoming HELO with our identity payload.
        """
        payload = json.dumps(self._build_identity_payload())
        response = Message(
            source=self.name,
            to=message.source,
            content=payload,
            message_type=MessageType.ACK,
        )
        await self.send_message(response)

    async def _send_helo(self) -> None:
        if not self._message_sender:
            return
        payload = json.dumps(self._build_identity_payload())
        message = Message(
            source=self.name,
            to="ALL",
            content=payload,
            message_type=MessageType.HELO,
        )
        await self.send_message(message)

    def _build_identity_payload(self) -> Dict[str, Any]:
        capabilities = self.config.get("capabilities")
        description = self.config.get("description")
        payload: Dict[str, Any] = {"name": self.name}
        if description:
            payload["description"] = description
        if isinstance(capabilities, dict):
            payload["capabilities"] = capabilities
        return payload

    def _track_task(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        """
        Track a background task and ensure cleanup when it completes.
        """
        self._tasks.append(task)

        def _cleanup(completed: asyncio.Task[Any]) -> None:
            if completed in self._tasks:
                self._tasks.remove(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                task_name = completed.get_name() if hasattr(completed, "get_name") else None
                logger.exception(
                    "Background task %s raised an exception",
                    task_name or "<unnamed>",
                )

        task.add_done_callback(_cleanup)
        return task

    def create_background_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: Optional[str] = None,
    ) -> asyncio.Task[T]:
        """
        Create and track a background task associated with this agent.

        Args:
            coro: Coroutine object to execute in the background.
            name: Optional task name for easier debugging.

        Returns:
            asyncio.Task: The created task.
        """
        task = asyncio.create_task(coro, name=name)
        return self._track_task(task)

    def track_task(self, task: asyncio.Task[T]) -> asyncio.Task[T]:
        """
        Track an externally created task so it participates in agent shutdown.

        Args:
            task: The task to track.

        Returns:
            The same task instance for convenience.
        """
        return self._track_task(task)
    
    async def stop(self) -> None:
        """
        Stop the agent and cleanup resources.
        
        This method stops the agent's operation and performs necessary cleanup:
        1. Sets running flag to False
        2. Cancels all registered background tasks
        3. Closes any open resources (e.g., client sessions)
        4. Logs completion
        
        Subclasses should call super().stop() when overriding this method.
        """
        if not self._running:
            return
            
        self._running = False
        
        # Cancel all background tasks
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        
        # Close any resources
        if hasattr(self, '_client_session') and self._client_session:
            await self._client_session.close()
            
        # Close message sender if it has a close method
        if self._message_sender and hasattr(self._message_sender, 'close'):
            try:
                await self._message_sender.close()
                logger.info("Closed message sender for agent %s", self.name)
            except Exception:
                logger.exception("Error closing message sender")
        if self._mcp_manager:
            try:
                await asyncio.wait_for(self._mcp_manager.stop(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("Timed out while stopping MCP manager for agent %s", self.name)
        
        logger.info("Agent %s stopped", self.name)

    # ------------------------------------------------------------------ #
    # MCP integration helpers
    # ------------------------------------------------------------------ #

    def _configure_mcp_manager(self, functions_registry) -> None:
        if self._mcp_registry_configured:
            return

        server_configs = self.config.get("mcp_servers") or []
        if not server_configs:
            self._mcp_registry_configured = True
            return

        try:
            from agentkit.mcp import MCPClientManager
        except ImportError:
            logger.warning("agentkit.mcp package missing; MCP servers will be unavailable.")
            self._mcp_registry_configured = True
            return

        manager = MCPClientManager.from_config(self, functions_registry, server_configs)
        if manager is None:
            self._mcp_registry_configured = True
            return

        self._mcp_manager = manager
        self._mcp_registry_configured = True

    async def _start_mcp_manager(self) -> None:
        if self._mcp_manager:
            try:
                await self._mcp_manager.start()
            except Exception:
                logger.exception("Failed to start MCP client manager for agent %s", self.name)
