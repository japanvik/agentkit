"""Delegation tools enabling agents to coordinate with peers."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from agentkit.functions.functions_registry import ToolExecutionContext


async def delegate_task_tool(
    context: ToolExecutionContext,
    target_agent: str,
    content: str,
    reminder_interval: Optional[float] = None,
) -> Dict[str, str]:
    """Delegate a task to another agent and register planner follow-up."""
    agent = context.agent
    planner = getattr(agent, "planner", None)
    if planner is None:
        raise RuntimeError("Planner not available on this agent; cannot delegate tasks")

    metadata = context.metadata or {}
    task_id = metadata.get("task_id")
    conversation_id = metadata.get("conversation_id")
    requester = metadata.get("requester") or agent.name
    path: Sequence[str] = metadata.get("delegation_path") or []

    if not task_id:
        raise ValueError("Delegation requires the current task_id in context metadata")

    augmented_path = list(path)
    if not augmented_path or augmented_path[-1] != agent.name:
        augmented_path.append(agent.name)
    augmented_path.append(target_agent)

    record = await planner.schedule_delegation(
        task_id=task_id,
        target_agent=target_agent,
        reminder_interval=reminder_interval,
        instructions=content,
        path=augmented_path,
    )

    await agent.functions_registry.execute(
        "send_message",
        parameters={
            "recipient": target_agent,
            "content": content,
            "message_type": "CHAT",
        },
        context=ToolExecutionContext(
            agent=agent,
            session_id=conversation_id,
            metadata={
                "conversation_id": conversation_id,
                "requester": requester,
                "agent_name": agent.name,
                "delegation_path": augmented_path,
                "task_id": task_id,
            },
        ),
    )

    return {
        "status": record.status,
        "delegation_id": record.delegation_id,
        "target_agent": record.target_agent,
        "reminder_interval": str(record.reminder_interval),
    }


async def escalate_task_tool(
    context: ToolExecutionContext,
    reason: str,
    details: Optional[str] = None,
) -> Dict[str, str]:
    """Escalate a stalled task to the nearest stakeholder in the delegation chain."""
    agent = context.agent
    planner = getattr(agent, "planner", None)
    if planner is None:
        raise RuntimeError("Planner not available on this agent; cannot escalate tasks")

    metadata = context.metadata or {}
    path = list(metadata.get("delegation_path") or [])
    if agent.name not in path:
        path.append(agent.name)
    task_id = metadata.get("task_id")
    conversation_id = metadata.get("conversation_id")
    requester = metadata.get("requester") or agent.name

    target = planner.choose_escalation_target(
        path=path,
        current_agent=agent.name,
        fallback=requester,
    )

    message_lines = [reason]
    if details:
        message_lines.append(details)
    escalation_message = "\n".join(message_lines)

    augmented_path = list(path)
    if not augmented_path or augmented_path[-1] != agent.name:
        augmented_path.append(agent.name)
    augmented_path.append(target)

    record = await planner.schedule_delegation(
        task_id=task_id or path[0] if path else task_id or agent.name,
        target_agent=target,
        reminder_interval=None,
        instructions=escalation_message,
        path=augmented_path,
    )

    await agent.functions_registry.execute(
        "send_message",
        parameters={
            "recipient": target,
            "content": escalation_message,
            "message_type": "CHAT",
        },
        context=ToolExecutionContext(
            agent=agent,
            session_id=conversation_id,
            metadata={
                "conversation_id": conversation_id,
                "requester": requester,
                "agent_name": agent.name,
                "delegation_path": augmented_path,
                "task_id": task_id,
            },
        ),
    )

    return {
        "status": record.status,
        "delegation_id": record.delegation_id,
        "target_agent": record.target_agent,
    }
