"""
Task-aware agent module for conversation-aware agents.

This module provides the TaskAwareAgent class, which extends BaseAgent to support
task management. It uses ThreadedMemory to maintain separate conversation contexts
and processes tasks in order of priority.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import count

from networkkit.messages import Message, MessageType

from agentkit.agents.base_agent import BaseAgent
from agentkit.functions.functions_registry import ToolExecutionContext
from agentkit.memory.conversation.context import ConversationContext
from agentkit.memory.conversation.task import Task
from agentkit.memory.threaded_memory import ThreadedMemory
from agentkit.planning import AgentPlanner, PlannerConfig

logger = logging.getLogger(__name__)


class TaskAwareAgent(BaseAgent):
    """
    Extends BaseAgent to support task management.
    
    TaskAwareAgent uses ThreadedMemory to maintain separate conversation contexts
    and processes tasks in order of priority. It provides methods for adding tasks,
    processing tasks, and managing conversation contexts.
    
    Attributes:
        name: The name of the agent.
        config: The configuration for the agent.
        memory: The memory used by the agent (ThreadedMemory by default).
        task_queue: A priority queue of tasks to process.
        _current_task: The task currently being processed.
        _task_processor_task: The asyncio task that processes tasks from the queue.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        """
        Initialize a new TaskAwareAgent.
        
        Args:
            name: The name of the agent.
            config: The configuration for the agent.
            **kwargs: Additional keyword arguments.
        """
        # Initialize functions registry if provided or create a new one
        from agentkit.functions.functions_registry import DefaultFunctionsRegistry
        self.functions_registry = kwargs.pop('functions_registry', DefaultFunctionsRegistry())

        # Ensure we have a message sender wired up for tool/fallback responses
        if "message_sender" not in kwargs or kwargs["message_sender"] is None:
            publish_address = config.get("bus_publish_address")
            if not publish_address and "bus_ip" in config:
                publish_address = f"http://{config['bus_ip']}:8000"

            if publish_address:
                try:
                    from networkkit.network import HTTPMessageSender
                    kwargs["message_sender"] = HTTPMessageSender(publish_address=publish_address)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to initialize HTTPMessageSender for agent %s", name
                    )
                    kwargs["message_sender"] = None
       
        super().__init__(name, config, **kwargs)
        
        # Register built-in tools
        self.register_tools(self.functions_registry)
        
        # Update component_config to include functions_registry
        self.component_config.functions_registry = self.functions_registry
        
        # Use ThreadedMemory by default
        if not kwargs.get('memory'):
            self.memory = ThreadedMemory()
            
        self.task_queue = asyncio.PriorityQueue()
        self._queue_counter = count()
        self._current_task: Optional[Task] = None
        self._task_processor_task: Optional[asyncio.Task] = None
        self._delegate_waiters: Dict[str, Dict[str, str]] = {}

        persistence_root = Path(
            self.config.get("planner_state_dir", "agent_state")
        )
        planner_config = PlannerConfig(
            persistence_dir=persistence_root,
            task_generation_model=self.config.get("planner_model"),
            task_generation_api_config=self.config.get("planner_api_config", {}) or {},
        )
        self.planner = AgentPlanner(
            self,
            config=planner_config,
            functions_registry=self.functions_registry,
        )
    
    async def start(self) -> None:
        """Start the agent and the task processor."""
        await super().start()
        await self.planner.start()
        # Start task processor
        self._task_processor_task = self.create_background_task(
            self._process_tasks(),
            name=f"{self.name}-task-processor"
        )
        
    async def stop(self) -> None:
        """Stop the agent and the task processor."""
        task = self._task_processor_task
        self._task_processor_task = None
        
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        await self.planner.stop()
        await super().stop()
    
    async def handle_message(self, message: Message) -> None:
        """
        Handle a message by creating a task and adding it to the queue.
        
        Args:
            message: The message to handle.
        """
        # Task-aware orchestration should only react to directly addressed
        # traffic (or broadcasts), not attention-based side channels.
        if message.to not in {self.name, "ALL"} and message.source != self.name:
            return

        # Only handle messages intended for this agent
        if not self.is_intended_for_me(message):
            return

        raw_kind = getattr(message.message_type, "value", message.message_type)
        message_kind = str(raw_kind).upper()
        if message_kind not in {"CHAT", "HELO", "ACK"}:
            return

        # Ignore our own HELO/ACK frames to prevent self-ack chatter.
        if message.source == self.name:
            logger.debug("Self-message observed type=%s normalized=%s", message.message_type, message_kind)
        if message.source == self.name and message_kind in {"HELO", "ACK"}:
            logger.debug("Ignoring self %s frame", message_kind)
            return

        if await self._handle_delegate_reply_shortcut(message):
            return
        
        conversation_id = None
        conversation = None
        if self.memory:
            self.memory.remember(message)
            if hasattr(self.memory, "conversation_manager"):
                conversation = self.memory.conversation_manager.get_or_create_conversation(message)
                conversation_id = conversation.conversation_id
        logger.debug(
            "Enqueueing task for message type=%s from=%s conv=%s",
            message.message_type,
            message.source,
            conversation_id,
        )
        if not conversation_id:
            conversation_id = self.memory.get_conversation_id(message) if self.memory else f"conversation:{uuid.uuid4()}"
        task = Task.create(
            conversation_id=conversation_id,
            message=message,
            description=f"Process {message.message_type} message from {message.source}"
        )

        conversation_path = self._build_conversation_path(conversation) if conversation else [message.source, self.name]
        task.metadata["delegation_path"] = conversation_path
        if conversation:
            conversation.set_state("delegation_path", conversation_path)
        
        # Adjust priority based on message type
        if message.message_type == MessageType.HELO:
            task.priority = 10  # High priority for connection establishment
        elif message.message_type == MessageType.CHAT:
            task.priority = 5   # Medium priority for chat messages
        
        # Add task to queue with effective priority
        await self.task_queue.put((-task.calculate_effective_priority(), next(self._queue_counter), task))
        logger.debug("Task queue size is now %s", self.task_queue.qsize())

    def register_delegate_wait(
        self,
        *,
        delegate: str,
        requester: str,
        intent: str = "general",
    ) -> None:
        delegate_name = (delegate or "").strip()
        requester_name = (requester or "").strip()
        if not delegate_name or not requester_name:
            return
        self._delegate_waiters[delegate_name] = {
            "requester": requester_name,
            "intent": intent or "general",
        }

    async def _handle_delegate_reply_shortcut(self, message: Message) -> bool:
        if message.message_type != MessageType.CHAT:
            return False

        record = self._delegate_waiters.get(message.source)
        if not isinstance(record, dict):
            return False

        if record.get("intent") == "long_running_followup":
            return False

        requester = str(record.get("requester", "")).strip()
        if not requester:
            return False

        self._delegate_waiters.pop(message.source, None)
        await self.functions_registry.execute(
            function="send_message",
            parameters={
                "recipient": requester,
                "content": message.content,
                "message_type": "CHAT",
            },
            context=ToolExecutionContext(agent=self),
        )
        return True
    
    async def _process_tasks(self) -> None:
        """Process tasks from the queue in order of priority."""
        while self._running:
            try:
                # Get highest priority task with a timeout to allow for cancellation
                try:
                    _, _, task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No task available, continue the loop
                    continue
                except asyncio.CancelledError:
                    # Task was cancelled, exit the loop
                    break
                
                logger.debug(
                    "Processing task %s for conversation %s (queue size approx %s)",
                    task.task_id,
                    task.conversation_id,
                    self.task_queue.qsize(),
                )

                try:
                    # Set current task
                    self._current_task = task
                    task.mark_in_progress()
                    
                    # Get conversation context
                    conversation = self.memory.get_conversation_by_id(task.conversation_id)
                    if not conversation:
                        logger.warning(f"Conversation {task.conversation_id} not found for task {task.task_id}")
                        task.mark_failed("Conversation not found")
                        continue
                    
                    # Process task based on message type
                    message = task.message
                    action = await self.planner.plan_for_message(
                        message,
                        task.conversation_id,
                    )

                    completed = await self._execute_action(
                        action,
                        conversation_id=task.conversation_id,
                        source_message=message,
                        conversation=conversation,
                    )

                    if completed:
                        task.mark_completed()
                    
                except Exception as e:
                    logger.error(f"Error processing task: {e}", exc_info=True)
                    if self._current_task:
                        self._current_task.mark_failed(str(e))
                    failed_message = task.message if task else None
                    if (
                        failed_message
                        and str(getattr(failed_message.message_type, "value", failed_message.message_type)).upper() == "CHAT"
                        and failed_message.source != self.name
                        and failed_message.to in {self.name, "ALL"}
                    ):
                        try:
                            await self.functions_registry.execute(
                                function="send_message",
                                parameters={
                                    "recipient": failed_message.source,
                                    "content": "I hit an internal error while processing your chat request.",
                                    "message_type": "CHAT",
                                },
                                context=ToolExecutionContext(agent=self),
                            )
                        except Exception:
                            logger.exception("Failed to send error fallback message")
                finally:
                    self._current_task = None
                    self.task_queue.task_done()
            except asyncio.CancelledError:
                # Task was cancelled, exit the loop
                break

    async def add_task(self, description: str, conversation_id: str = None,
                      priority: int = 0, due_time: Optional[datetime] = None) -> Task:
        """
        Add a custom task to the queue.

        Args:
            description: A description of the task.
            conversation_id: The ID of the conversation to add the task to.
            priority: The priority of the task.
            due_time: The due time of the task.

        Returns:
            The created task.
        """
        # If no conversation_id provided, use the current conversation if available
        if not conversation_id and self._current_task:
            conversation_id = self._current_task.conversation_id
            
        # If still no conversation_id, create a default one
        if not conversation_id:
            conversation_id = f"task:{str(uuid.uuid4())}"

        # Create a dummy message if needed
        message = Message(
            source=self.name,
            to=self.name,
            content=description,
            message_type=MessageType.SYSTEM
        )

        # Create and queue the task
        task = Task.create(
            conversation_id=conversation_id,
            message=message,
            description=description,
            priority=priority,
            due_time=due_time
        )
        
        # Create a conversation for this task if it doesn't exist
        if self.memory:
            # Create a conversation context for this task if it doesn't exist
            if not self.memory.get_conversation_by_id(conversation_id):
                # Create a conversation directly with the ID we want
                conversation = self.memory.conversation_manager.conversations[conversation_id] = \
                    ConversationContext(conversation_id, message)
            
            # Add the task to the conversation manager
            self.memory.conversation_manager.add_task(task)
            
            logger.debug(f"Added task {task.task_id} to conversation {conversation_id}")

        await self.task_queue.put((-task.calculate_effective_priority(), next(self._queue_counter), task))
        logger.debug(
            "Added manual task %s; queue size=%s",
            task.task_id,
            self.task_queue.qsize(),
        )
        return task
    
    def get_current_task(self) -> Optional[Task]:
        """
        Get the task currently being processed.
        
        Returns:
            The task currently being processed, or None if no task is being processed.
        """
        return self._current_task

    def log_debug_state(self) -> None:
        state = self.planner.describe_state()
        logger.debug("Planner state snapshot: %s", state)
        logger.debug("Task queue depth: %s", self.task_queue.qsize())
    
    async def get_pending_tasks(self) -> List[Task]:
        """
        Get all pending tasks.
        
        Returns:
            A list of all pending tasks.
        """
        if not self.memory:
            return []
        
        # Get all tasks from the conversation manager
        tasks = list(self.memory.conversation_manager.tasks.values())
        
        # Filter for pending tasks
        pending_tasks = [task for task in tasks if task.status == "pending"]
        
        return pending_tasks
    
    async def get_active_conversations(self, max_age_minutes: int = 60) -> List[str]:
        """
        Get all active conversations.
        
        Args:
            max_age_minutes: The maximum age of conversations to consider active, in minutes.
            
        Returns:
            A list of IDs of active conversations.
        """
        if not self.memory:
            return []
        conversations = self.memory.get_active_conversations(max_age_minutes)
        return [conv.conversation_id for conv in conversations]
    
    async def _determine_action(self, task: Task, conversation: ConversationContext) -> Dict[str, Any]:
        """
        Determine what action to take for a task using LLM.
        
        This method uses the LLM to decide what action to take for a task, based on
        the task message and conversation context. The LLM can decide to send a message
        or use another tool.
        
        Args:
            task: The task to determine an action for
            conversation: The conversation context
            
        Returns:
            A dictionary describing the action to take
        """
        import json
        from agentkit.processor import llm_chat, extract_json
        
        # Set attention to the message source
        self.component_config.message_sender.attention = task.message.source
        
        # Get the available tools from the functions registry
        tools_prompt = self.functions_registry.prompt()
        
        # Get recent messages from the conversation
        recent_messages = conversation.get_recent_messages(5)  # Get last 5 messages
        context = "\n".join([f"{msg.source}: {msg.content}" for msg in recent_messages])
        
        # Known agents the planner has seen via HELOs
        known_agents = []
        if hasattr(self, "planner") and getattr(self.planner, "known_agents", None):
            for agent_name, info in self.planner.known_agents.items():
                caps = info.get("capabilities") or {}
                caps_summary = ", ".join(sorted(caps.keys())) if isinstance(caps, dict) and caps else json.dumps(caps)
                last_seen = info.get("last_seen", "unknown time")
                known_agents.append(f"- {agent_name}: capabilities={caps_summary or 'unknown'} (last seen {last_seen})")
        if not known_agents:
            known_agents.append("- None recorded")
        known_agents_block = "\n".join(known_agents)

        # Create system prompt for the LLM
        system_prompt = f"""
        You are an AI assistant named {self.name}. You are processing a message and need to decide what action to take.

        Available tools:
        {tools_prompt}

        Known agents on the network (from recent HELO messages):
        {known_agents_block}

        Based on the message and conversation context, decide whether to:
        1. Send a message using the send_message tool
        2. Use another tool from the available tools

        Guiding principles:
        - First check whether the requested information or action can be satisfied with what you already know (conversation history, known agents, prior tool outputs). If it can, respond immediately using send_message without calling another tool.
        - If the user asks who is available/online, summarize the entries in the known agents section instead of stating you lack that capability.
        - Only invoke a tool when you need new information or need to perform an external action.
        - Prefer providing direct answers when the relevant data is already present in context.

        Your response should be a JSON object with the following structure:
        {{
            "action_type": "send_message" or "use_tool",
            "tool_name": "name of the tool to use",
            "parameters": {{
                "param1": "value1",
                "param2": "value2",
                ...
            }}
        }}
        """
        
        # Create user prompt for the LLM
        user_prompt = f"""
        Message from: {task.message.source}
        Message content: {task.message.content}
        
        Recent conversation context:
        {context}
        
        What action should I take in response to this message?
        """
        
        # Get response from LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await llm_chat(
            llm_model=self.config.get('model', 'gpt-4'),
            messages=messages
        )
        
        try:
            # Extract JSON from response
            action = extract_json(response)
            logger.debug(f"Determined action: {action}")
            return action
        except Exception as e:
            logger.error(f"Error parsing LLM response for action determination: {e}")
            # Fall back to sending a message
            return {
                "action_type": "send_message",
                "tool_name": "send_message",
                "parameters": {
                    "recipient": task.message.source,
                    "content": f"I'm sorry, I couldn't process your request properly. Could you please rephrase or provide more details?",
                    "message_type": "CHAT"
                }
            }
    
    def _build_conversation_path(self, conversation: Optional[ConversationContext]) -> List[str]:
        path: List[str] = []
        if conversation:
            stored = conversation.get_state("delegation_path")
            if isinstance(stored, list) and stored:
                path = list(stored)
            else:
                for msg in conversation.history:
                    if msg.source not in path:
                        path.append(msg.source)
        if self.name not in path:
            path.append(self.name)
        return path

    async def _execute_action(
        self,
        action: Dict[str, Any],
        *,
        conversation_id: Optional[str] = None,
        source_message: Optional[Message] = None,
        conversation: Optional[ConversationContext] = None,
    ) -> bool:
        """
        Execute an action determined by the LLM.
        
        This method executes an action determined by the LLM, which could be
        sending a message or using another tool.
        
        Args:
            action: A dictionary describing the action to take
        """
        action_type = action.get("action_type")
        tool_name = action.get("tool_name")
        parameters = action.get("parameters", {})
        
        if action_type == "noop":
            return True

        if action_type == "send_message" or tool_name == "send_message":
            logger.debug("Executing send_message action with params=%s", parameters)
            await self.functions_registry.execute(
                function="send_message",
                parameters=parameters,
                context=ToolExecutionContext(agent=self),
            )
            return True

        if action_type == "use_brain":
            logger.debug("Delegating to brain for conversation %s", conversation_id)
            if self.brain and source_message is not None:
                if hasattr(self.brain, "set_active_context"):
                    current_task_id = self._current_task.task_id if self._current_task else None
                    delegation_path = self._current_task.metadata.get("delegation_path") if self._current_task else None
                    if conversation and (not delegation_path or not isinstance(delegation_path, list)):
                        delegation_path = self._build_conversation_path(conversation)
                    self.brain.set_active_context(
                        conversation_id,
                        source_message,
                        current_task_id,
                        delegation_path if isinstance(delegation_path, list) else None,
                    )
                await self.brain.handle_chat_message(source_message)
                if hasattr(self.brain, "clear_active_context"):
                    self.brain.clear_active_context()
                return True
            return False

        if action_type == "use_tool" and tool_name:
            logger.debug("Executing tool %s with params=%s", tool_name, parameters)
            metadata = {
                "conversation_id": conversation_id,
                "agent_name": self.name,
            }
            if source_message is not None:
                metadata.update(
                    {
                        "requester": source_message.source,
                        "source": source_message.source,
                        "target": source_message.source,
                        "last_message_content": source_message.content,
                    }
                )
            if conversation is not None:
                path = self._build_conversation_path(conversation)
                conversation.set_state("delegation_path", path)
            elif self._current_task and isinstance(
                self._current_task.metadata.get("delegation_path"), list
            ):
                path = list(self._current_task.metadata.get("delegation_path"))
            elif source_message is not None:
                path = [source_message.source, self.name]
            else:
                path = [self.name]
            metadata["delegation_path"] = path
            if self._current_task:
                metadata["task_id"] = self._current_task.task_id
                self._current_task.metadata["delegation_path"] = path
            result = await self.functions_registry.execute(
                function=tool_name,
                parameters=parameters,
                context=ToolExecutionContext(
                    agent=self,
                    session_id=conversation_id,
                    metadata=metadata,
                ),
            )
            logger.debug("Tool %s returned result=%s", tool_name, result)
            if conversation_id and source_message:
                follow_up = await self.planner.notify_tool_result(
                    conversation_id=conversation_id,
                    result=result or {},
                    original_message=source_message,
                )
                if follow_up.get("action_type") and follow_up.get("tool_name"):
                    logger.debug("Executing follow-up action %s", follow_up)
                    return await self._execute_action(
                        follow_up,
                        conversation_id=conversation_id,
                        source_message=source_message,
                        conversation=conversation,
                    )
            return True

        logger.warning(f"Unknown action type: {action_type}")
        return False
