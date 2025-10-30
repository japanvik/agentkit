"""Reminder-related tools that interact with the planner."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

from agentkit.functions.functions_registry import ToolExecutionContext


async def schedule_reminder_tool(
    context: ToolExecutionContext,
    content: str,
    recipient: Optional[str] = None,
    run_at: Optional[str] = None,
    delay_seconds: Optional[float] = None,
    repeat_seconds: Optional[float] = None,
    description: Optional[str] = None,
) -> Dict[str, str]:
    agent = context.agent
    planner = getattr(agent, "planner", None)
    if planner is None:
        raise RuntimeError("Planner not available on this agent; cannot schedule reminders")

    parsed_run_at: Optional[datetime] = None
    if run_at:
        try:
            parsed_run_at = datetime.fromisoformat(run_at)
        except ValueError as exc:
            raise ValueError("run_at must be an ISO 8601 datetime string") from exc

    reminder = await planner.schedule_reminder(
        content=content,
        recipient=recipient,
        run_at=parsed_run_at,
        delay_seconds=delay_seconds,
        repeat_seconds=repeat_seconds,
        description=description,
    )

    return {
        "status": reminder.status,
        "reminder_id": reminder.reminder_id,
        "next_run": reminder.next_run.isoformat(),
        "recipient": reminder.recipient,
    }
