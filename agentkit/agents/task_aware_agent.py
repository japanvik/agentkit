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
from typing import Dict, List, Optional, Any, Tuple

from networkkit.messages import Message, MessageType

from agentkit.agents.base_agent import BaseAgent
from agentkit.memory.threaded_memory import ThreadedMemory
from agentkit.memory.conversation.task import Task
from agentkit.memory.conversation.context import ConversationContext

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
        self.functions_registry = kwargs.get('functions_registry', DefaultFunctionsRegistry())
        
        super().__init__(name, config, **kwargs)
        
        # Register built-in tools
        self.register_tools(self.functions_registry)
        
        # Update component_config to include functions_registry
        self.component_config.functions_registry = self.functions_registry
        
        # Use ThreadedMemory by default
        if not kwargs.get('memory'):
            self.memory = ThreadedMemory()
            
        self.task_queue = asyncio.PriorityQueue()
        self._current_task: Optional[Task] = None
        self._task_processor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the agent and the task processor."""
        await super().start()
        # Start task processor
        self._task_processor_task = asyncio.create_task(self._process_tasks())
        self._tasks.append(self._task_processor_task)
        
    async def stop(self) -> None:
        """Stop the agent and the task processor."""
        # Cancel the task processor task
        if self._task_processor_task and not self._task_processor_task.done():
            self._task_processor_task.cancel()
            try:
                await self._task_processor_task
            except asyncio.CancelledError:
                pass
        
        # Clear the tasks list
        self._tasks.clear()
        
        # Call the parent class's stop method
        await super().stop()
    
    async def handle_message(self, message: Message) -> None:
        """
        Handle a message by creating a task and adding it to the queue.
        
        Args:
            message: The message to handle.
        """
        # Only handle messages intended for this agent
        if not self.is_intended_for_me(message):
            return
        
        # Store message in memory
        if self.memory:
            self.memory.remember(message)
        
        # Create task from message
        conversation_id = self.memory.get_conversation_id(message)
        task = Task.create(
            conversation_id=conversation_id,
            message=message,
            description=f"Process {message.message_type} message from {message.source}"
        )
        
        # Adjust priority based on message type
        if message.message_type == MessageType.HELO:
            task.priority = 10  # High priority for connection establishment
        elif message.message_type == MessageType.CHAT:
            task.priority = 5   # Medium priority for chat messages
        
        # Add task to queue with effective priority
        await self.task_queue.put((-task.calculate_effective_priority(), task))
    
    async def _process_tasks(self) -> None:
        """Process tasks from the queue in order of priority."""
        while self._running:
            try:
                # Get highest priority task with a timeout to allow for cancellation
                try:
                    _, task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No task available, continue the loop
                    continue
                except asyncio.CancelledError:
                    # Task was cancelled, exit the loop
                    break
                
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
                    if message.message_type == MessageType.CHAT:
                        if self.brain:
                            # Determine what action to take using LLM
                            action = await self._determine_action(task, conversation)
                            
                            # Execute the determined action
                            await self._execute_action(action)
                    
                    elif message.message_type == MessageType.HELO:
                        # Respond with ACK
                        response = Message(
                            source=self.name,
                            to=message.source,
                            content="",
                            message_type=MessageType.ACK
                        )
                        await self._internal_send_message(response)
                    
                    # Mark task as completed
                    task.mark_completed()
                    
                except Exception as e:
                    logger.error(f"Error processing task: {e}", exc_info=True)
                    if self._current_task:
                        self._current_task.mark_failed(str(e))
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

        await self.task_queue.put((-task.calculate_effective_priority(), task))
        return task
    
    def get_current_task(self) -> Optional[Task]:
        """
        Get the task currently being processed.
        
        Returns:
            The task currently being processed, or None if no task is being processed.
        """
        return self._current_task
    
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
        
        # Create system prompt for the LLM
        system_prompt = f"""
        You are an AI assistant named {self.name}. You are processing a message and need to decide what action to take.
        
        Available tools:
        {tools_prompt}
        
        Based on the message and conversation context, decide whether to:
        1. Send a message using the send_message tool
        2. Use another tool from the available tools
        
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
    
    async def _execute_action(self, action: Dict[str, Any]) -> None:
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
        
        if action_type == "send_message" or tool_name == "send_message":
            # Use the send_message tool
            await self.functions_registry.execute(
                function="send_message",
                parameters=parameters
            )
        elif action_type == "use_tool" and tool_name:
            # Use another tool
            await self.functions_registry.execute(
                function=tool_name,
                parameters=parameters
            )
        else:
            logger.warning(f"Unknown action type: {action_type}")
