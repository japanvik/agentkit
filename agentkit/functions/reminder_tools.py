"""Reminder-related tools that interact with the planner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from agentkit.functions.functions_registry import ToolExecutionContext


def _parse_iso_timestamp(timestamp: str) -> datetime:
    """
    Parse an ISO-8601 timestamp string, accepting a trailing 'Z' for UTC.

    Returns a timezone-naive UTC datetime so reminder scheduling logic can compare
    it directly with utc_now() which also returns naive UTC values.
    """
    cleaned = timestamp.strip()
    if cleaned.lower().endswith("z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


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
            parsed_run_at = _parse_iso_timestamp(run_at)
        except ValueError as exc:
            raise ValueError("run_at must be an ISO 8601 datetime string") from exc

    metadata = dict(context.metadata or {})
    conversation_id = metadata.get("conversation_id") or context.session_id
    requester = metadata.get("requester") or metadata.get("source") or context.agent.name
    target = recipient or metadata.get("target") or requester

    reminder_metadata = {
        "action": "send_message" if target else metadata.get("action", "plan"),
        "target": target,
        "message": content,
        "conversation_id": conversation_id,
        "requested_by": requester,
        "delegation_path": metadata.get("delegation_path"),
    }

    reminder = await planner.schedule_reminder(
        content=content,
        recipient=recipient or context.agent.name,
        run_at=parsed_run_at,
        delay_seconds=delay_seconds,
        repeat_seconds=repeat_seconds,
        description=description,
        metadata=reminder_metadata,
    )

    return {
        "status": reminder.status,
        "reminder_id": reminder.reminder_id,
        "next_run": reminder.next_run.isoformat(),
        "recipient": reminder.recipient,
        "metadata": reminder.metadata,
    }
