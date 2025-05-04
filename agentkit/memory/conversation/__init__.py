"""
Conversation management module for the AgentKit framework.

This module provides classes for managing conversations, tasks, and context
across multiple agents. It enables agents to maintain separate conversation
contexts and manage tasks with priorities.

Classes:
    ConversationContext: Manages the context of a conversation, including messages,
        participants, and tasks.
    Task: Represents a task to be performed by an agent, with priority and status.
    ConversationManager: Manages multiple conversations and their associated tasks.
"""

from agentkit.memory.conversation.context import ConversationContext
from agentkit.memory.conversation.task import Task
from agentkit.memory.conversation.manager import ConversationManager

__all__ = ["ConversationContext", "Task", "ConversationManager"]
