from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from networkkit.messages import Message, MessageType

from agentkit.functions.functions_registry import (
    DefaultFunctionsRegistry,
    ToolExecutionContext,
)
from agentkit.planning.state import (
    DelegationRecord,
    PlannerStateStore,
    PlannerTaskState,
    ReminderRecord,
    utc_now,
)

logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    persistence_dir: Path
    reminder_interval_seconds: float = 60.0
    reminder_multiplier: float = 2.0
    max_reminder_interval: float = 3600.0
    max_reminder_attempts: int = 8
    reminder_loop_interval: float = 30.0
    default_deadline_hours: float = 6.0


class AgentPlanner:
    """
    High-level planner that orchestrates tool usage, delegation, and reminders.
    """

    def __init__(
        self,
        agent,
        *,
        config: PlannerConfig,
        functions_registry: DefaultFunctionsRegistry,
    ):
        self.agent = agent
        self.config = config
        self.functions_registry = functions_registry

        persistence_path = (
            config.persistence_dir / f"{self.agent.name}_planner_state.json"
        )
        self._store = PlannerStateStore(persistence_path)
        loaded = self._store.load()

        self.tasks: Dict[str, PlannerTaskState] = loaded["tasks"]
        self.delegations: Dict[str, DelegationRecord] = loaded["delegations"]
        self.reminders: Dict[str, ReminderRecord] = loaded["reminders"]
        self.known_agents: Dict[str, Dict] = loaded["known_agents"]

        self._reminder_task: Optional[asyncio.Task] = None
        self._save_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._reminder_task is None:
            self._reminder_task = self.agent.create_background_task(
                self._reminder_loop(), name=f"{self.agent.name}-planner-reminders"
            )

    async def stop(self) -> None:
        if self._reminder_task:
            self._reminder_task.cancel()
            try:
                await self._reminder_task
            except asyncio.CancelledError:
                pass
            self._reminder_task = None
        await self._save_state()

    async def _save_state(self) -> None:
        async with self._save_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._store.save(
                    tasks=self.tasks,
                    delegations=self.delegations,
                    known_agents=self.known_agents,
                    reminders=self.reminders,
                ),
            )

    async def reset_state(self) -> None:
        self.tasks.clear()
        self.delegations.clear()
        self.reminders.clear()
        self.known_agents.clear()
        await self._save_state()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._store.reset)

    def describe_state(self) -> Dict[str, Any]:
        return {
            "tasks": {
                task_id: {
                    "status": task.status,
                    "waiting_on_delegation": task.waiting_on_delegation,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                }
                for task_id, task in self.tasks.items()
            },
            "delegations": {
                deleg_id: {
                    "status": deleg.status,
                    "target": deleg.target_agent,
                    "next_due": (deleg.next_due().isoformat() if deleg.next_due() else None),
                    "attempts": deleg.reminder_attempts,
                }
                for deleg_id, deleg in self.delegations.items()
            },
            "reminders": {
                reminder_id: {
                    "status": reminder.status,
                    "recipient": reminder.recipient,
                    "next_run": reminder.next_run.isoformat(),
                    "repeat": reminder.repeat_interval,
                }
                for reminder_id, reminder in self.reminders.items()
            },
            "known_agents": self.known_agents,
        }

    def register_helo(
        self,
        *,
        agent_name: str,
        capabilities: Optional[Dict] = None,
        last_seen: Optional[datetime] = None,
    ) -> None:
        self.known_agents[agent_name] = {
            "capabilities": capabilities or {},
            "last_seen": (last_seen or utc_now()).isoformat(),
        }

    async def plan_for_message(self, message: Message, conversation_id: str) -> Dict:
        """
        Decide the next action for an incoming message.
        Returns an action dictionary compatible with TaskAwareAgent._execute_action.
        """
        task_state = self._ensure_task(message, conversation_id)

        logger.debug(
            "Planner received message (conv=%s, from=%s, type=%s, content=%s)",
            conversation_id,
            message.source,
            message.message_type,
            message.content,
        )

        if message.message_type == MessageType.HELO:
            self.register_helo(agent_name=message.source)
            task_state.status = "completed"
            await self._save_state()
            return {
                "action_type": "send_message",
                "tool_name": "send_message",
                "parameters": {
                    "recipient": message.source,
                    "content": "",
                    "message_type": "ACK",
                },
            }

    ### heuristics start
        if message.source == self.agent.name:
            task_state.status = "completed"
            await self._save_state()
            return {
                "action_type": "noop",
                "tool_name": "",
                "parameters": {},
            }

        lower_content = (message.content or "").lower()
        filesystem_keywords = [
            "cwd",
            "current directory",
            "working directory",
            "pwd",
            "file system",
            "filesystem",
            "list files",
            "directory listing",
        ]
        if any(keyword in lower_content for keyword in filesystem_keywords):
            apology = (
                "I don't have the ability to inspect your local filesystem or working directory."
            )
            logger.debug(
                "Planner detected filesystem request and will send apology (conv %s)",
                conversation_id,
            )
            return {
                "action_type": "send_message",
                "tool_name": "send_message",
                "parameters": {
                    "recipient": message.source,
                    "content": apology,
                    "message_type": "CHAT",
                },
            }

        capability_keywords = [
            "tool",
            "capability",
            "ability",
            "available function",
            "what can you do",
        ]
        if any(keyword in lower_content for keyword in capability_keywords):
            tool_names = ", ".join(sorted(self.functions_registry.function_map.keys()))
            summary = (
                "Here's a list of tools I can use right now: "
                f"{tool_names}. Let me know which one you'd like me to use."
            )
            logger.debug(
                "Planner responding with capability summary for conversation %s",
                conversation_id,
            )
            return {
                "action_type": "send_message",
                "tool_name": "send_message",
                "parameters": {
                    "recipient": message.source,
                    "content": summary,
                    "message_type": "CHAT",
                },
            }

        task_state.status = "running"
        await self._save_state()
        logger.debug(
            "Planner deferring to brain for conversation %s", conversation_id
        )

        return {
            "action_type": "use_brain",
            "tool_name": "",
            "parameters": {},
        }

    def _ensure_task(self, message: Message, conversation_id: str) -> PlannerTaskState:
        message_identifier = (
            getattr(message, "message_id", None)
            or getattr(message, "id", None)
            or getattr(message, "_planner_message_id", None)
            or str(uuid.uuid4())
        )
        setattr(message, "_planner_message_id", message_identifier)
        key = f"{conversation_id}:{message_identifier}"
        if key not in self.tasks:
            self.tasks[key] = PlannerTaskState(
                task_id=key,
                conversation_id=conversation_id,
                origin=message.source,
                metadata={"message_id": message_identifier},
            )
        return self.tasks[key]

    async def notify_tool_result(
        self,
        *,
        conversation_id: str,
        result: Dict,
        original_message: Message,
    ) -> Dict:
        """
        Handle the output of a tool and decide on the follow-up action.
        """
        message_identifier = (
            getattr(original_message, "message_id", None)
            or getattr(original_message, "id", None)
            or getattr(original_message, "_planner_message_id", None)
            or str(uuid.uuid4())
        )
        key = f"{conversation_id}:{message_identifier}"
        task_state = self.tasks.get(key)
        if not task_state:
            logger.warning("Planner missing task for conversation %s", conversation_id)
            return {
                "action_type": "noop",
                "tool_name": "",
                "parameters": {},
            }

        task_state.status = "completed"
        if task_state.actions:
            task_state.actions[-1].status = "completed"
            task_state.actions[-1].result = result
        await self._save_state()

        stdout = result.get("stdout", "").strip()
        content = stdout or "I executed the command."
        result_action = {
            "action_type": "send_message",
            "tool_name": "send_message",
            "parameters": {
                "recipient": original_message.source,
                "content": content,
                "message_type": "CHAT",
            },
        }
        logger.debug(
            "Planner generated follow-up send_message for conversation %s with content=%s",
            conversation_id,
            content,
        )
        return result_action

    async def schedule_delegation(
        self,
        *,
        task_id: str,
        target_agent: str,
        reminder_interval: Optional[float] = None,
    ) -> DelegationRecord:
        record = DelegationRecord(
            delegation_id=str(uuid.uuid4()),
            task_id=task_id,
            target_agent=target_agent,
            reminder_interval=reminder_interval
            or self.config.reminder_interval_seconds,
            reminder_multiplier=self.config.reminder_multiplier,
            max_reminder_interval=self.config.max_reminder_interval,
            max_attempts=self.config.max_reminder_attempts,
            deadline=utc_now()
            + timedelta(hours=self.config.default_deadline_hours),
        )
        self.delegations[record.delegation_id] = record
        task_state = self.tasks.get(task_id)
        if task_state:
            task_state.waiting_on_delegation = record.delegation_id
            task_state.status = "waiting"
        await self._save_state()
        return record

    async def resolve_delegation(
        self,
        *,
        delegation_id: str,
        status: str,
    ) -> None:
        record = self.delegations.get(delegation_id)
        if not record:
            return
        record.status = status
        task_state = self.tasks.get(record.task_id)
        if task_state and task_state.waiting_on_delegation == delegation_id:
            task_state.waiting_on_delegation = None
            if status == "completed":
                task_state.status = "completed"
        await self._save_state()

    async def _reminder_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.config.reminder_loop_interval)
                await self._process_reminders()
        except asyncio.CancelledError:
            pass

    async def _process_reminders(self) -> None:
        now = utc_now()
        to_remove = []
        for delegation_id, record in list(self.delegations.items()):
            if record.status != "waiting":
                continue
            if record.deadline and now >= record.deadline:
                logger.info(
                    "Delegation %s exceeded deadline; marking as timed out.",
                    delegation_id,
                )
                record.status = "timed_out"
                task_state = self.tasks.get(record.task_id)
                if task_state and task_state.waiting_on_delegation == delegation_id:
                    task_state.waiting_on_delegation = None
                    task_state.status = "failed"
                continue
            next_due = record.next_due()
            if next_due and now >= next_due:
                await self._send_reminder(record)
        for reminder_id, record in list(self.reminders.items()):
            if record.status != "scheduled":
                continue
            if now >= record.next_run:
                await self._fire_self_reminder(record)
        if to_remove:
            for delegation_id in to_remove:
                self.delegations.pop(delegation_id, None)
            await self._save_state()

    async def _send_reminder(self, record: DelegationRecord) -> None:
        logger.info(
            "Sending reminder to %s for task %s (attempt %s)",
            record.target_agent,
            record.task_id,
            record.reminder_attempts + 1,
        )
        reminder_content = (
            f"Reminder: awaiting your response for task {record.task_id}."
        )
        try:
            await self.functions_registry.execute(
                "send_message",
                parameters={
                    "recipient": record.target_agent,
                    "content": reminder_content,
                    "message_type": "CHAT",
                },
                context=ToolExecutionContext(agent=self.agent),
            )
        except Exception:
            logger.exception("Failed to send reminder via send_message tool")
        record.register_reminder()
        if record.reminder_attempts >= record.max_attempts:
            logger.info(
                "Delegation %s exceeded reminder attempts; marking as timed out.",
                record.delegation_id,
            )
            record.status = "timed_out"
            task_state = self.tasks.get(record.task_id)
            if task_state and task_state.waiting_on_delegation == record.delegation_id:
                task_state.waiting_on_delegation = None
                task_state.status = "failed"
        await self._save_state()

    async def schedule_reminder(
        self,
        *,
        content: str,
        recipient: Optional[str] = None,
        run_at: Optional[datetime] = None,
        delay_seconds: Optional[float] = None,
        repeat_seconds: Optional[float] = None,
        description: Optional[str] = None,
    ) -> ReminderRecord:
        if not content:
            raise ValueError("content is required")

        if run_at is None and delay_seconds is None and repeat_seconds is None:
            delay_seconds = self.config.reminder_interval_seconds

        if run_at is None:
            if delay_seconds is None:
                raise ValueError("Either run_at or delay_seconds must be provided")
            run_at = utc_now() + timedelta(seconds=delay_seconds)

        reminder = ReminderRecord(
            reminder_id=str(uuid.uuid4()),
            description=description or content[:80],
            content=content,
            recipient=recipient or self.agent.name,
            next_run=run_at,
            repeat_interval=repeat_seconds,
        )
        self.reminders[reminder.reminder_id] = reminder
        await self._save_state()
        return reminder

    async def cancel_reminder(self, reminder_id: str) -> bool:
        record = self.reminders.get(reminder_id)
        if not record:
            return False
        record.status = "cancelled"
        await self._save_state()
        return True

    async def _fire_self_reminder(self, record: ReminderRecord) -> None:
        logger.info(
            "Delivering reminder %s to %s", record.reminder_id, record.recipient
        )
        try:
            await self.functions_registry.execute(
                "send_message",
                parameters={
                    "recipient": record.recipient,
                    "content": record.content,
                    "message_type": "CHAT",
                },
                context=ToolExecutionContext(agent=self.agent),
            )
        except Exception:
            logger.exception("Failed to deliver reminder %s", record.reminder_id)

        if not record.reschedule():
            self.reminders.pop(record.reminder_id, None)
        await self._save_state()
