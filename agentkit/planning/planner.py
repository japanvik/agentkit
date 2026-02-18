from __future__ import annotations

import asyncio
import logging
import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

from networkkit.messages import Message, MessageType

from agentkit.functions.functions_registry import (
    DefaultFunctionsRegistry,
    ToolExecutionContext,
)
from agentkit.processor import extract_json, llm_chat
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
    task_generation_model: Optional[str] = None
    task_generation_api_config: Dict[str, Any] = field(default_factory=dict)
    max_generated_actions: int = 6


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
        default_model: Optional[str] = None
        default_api_config: Dict[str, Any] = {}
        if hasattr(self.agent, "config") and isinstance(self.agent.config, dict):
            default_model = self.agent.config.get("model")
            default_api_config = self.agent.config.get("api_config", {}) or {}
        self._task_generation_model = config.task_generation_model or default_model
        self._task_generation_api_config = (
            config.task_generation_api_config or default_api_config
        )

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
                    "actions": [
                        {
                            "action_id": action.action_id,
                            "description": action.description,
                            "status": action.status,
                        }
                        for action in task.actions
                    ],
                    "completion_criteria": task.metadata.get("completion_criteria", []),
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
                    "metadata": reminder.metadata,
                }
                for reminder_id, reminder in self.reminders.items()
            },
            "known_agents": self.known_agents,
        }

    def register_helo(
        self,
        *,
        agent_name: str,
        description: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        last_seen: Optional[datetime] = None,
    ) -> None:
        existing = self.known_agents.get(agent_name, {})
        entry = {
            "description": description or existing.get("description"),
            "capabilities": capabilities or existing.get("capabilities") or {},
            "last_seen": (last_seen or utc_now()).isoformat(),
        }
        self.known_agents[agent_name] = entry

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
            description: Optional[str] = None
            capabilities: Optional[Dict[str, Any]] = None
            if message.content:
                try:
                    payload = json.loads(message.content)
                    description = payload.get("description")
                    capabilities = payload.get("capabilities")
                except (json.JSONDecodeError, TypeError):
                    description = message.content
            identity_payload = {}
            if hasattr(self.agent, "_build_identity_payload"):
                identity_payload = self.agent._build_identity_payload()
            self.register_helo(
                agent_name=message.source,
                description=description,
                capabilities=capabilities if isinstance(capabilities, dict) else None,
            )
            task_state.status = "completed"
            await self._save_state()
            return {
                "action_type": "send_message",
                "tool_name": "send_message",
                "parameters": {
                    "recipient": message.source,
                    "content": json.dumps(identity_payload),
                    "message_type": "ACK",
                },
            }

        if message.message_type == MessageType.ACK:
            description: Optional[str] = None
            capabilities: Optional[Dict[str, Any]] = None
            if message.content:
                try:
                    payload = json.loads(message.content)
                    description = payload.get("description")
                    capabilities = payload.get("capabilities")
                except (json.JSONDecodeError, TypeError):
                    description = message.content
            self.register_helo(
                agent_name=message.source,
                description=description,
                capabilities=capabilities if isinstance(capabilities, dict) else None,
            )
            task_state.status = "completed"
            await self._save_state()
            return {
                "action_type": "noop",
                "tool_name": "",
                "parameters": {},
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

        if message.message_type in {MessageType.CHAT, MessageType.SYSTEM} and not task_state.actions:
            await self._generate_task_actions(task_state, message)

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

    async def _generate_task_actions(
        self,
        task_state: PlannerTaskState,
        message: Message,
    ) -> None:
        actions: List[Dict[str, Any]]
        completion_criteria: List[str]
        dependencies: List[List[int]]

        generated = await self._generate_task_plan_with_llm(message)
        if generated:
            actions, completion_criteria, dependencies = generated
        else:
            actions = [{"description": "Respond to the requester with a complete answer."}]
            completion_criteria = ["Requester receives a clear, final response."]
            dependencies = [[]]

        for action in actions[: self.config.max_generated_actions]:
            description = str(action.get("description", "")).strip()
            if not description:
                continue
            task_state.add_action(description)

        task_state.metadata["completion_criteria"] = completion_criteria
        task_state.metadata["action_dependencies"] = dependencies
        task_state.metadata["planning_model"] = self._task_generation_model
        task_state.updated_at = utc_now()
        await self._save_state()

    async def _generate_task_plan_with_llm(
        self,
        message: Message,
    ) -> Optional[tuple[List[Dict[str, Any]], List[str], List[List[int]]]]:
        if not self._task_generation_model:
            return None

        system_prompt = (
            "You decompose user requests into executable actions for an orchestrator. "
            "Return JSON with shape: "
            '{"actions":[{"description":"...","depends_on":[0]}],"completion_criteria":["..."]}. '
            "Keep actions concise, actionable, and limited to at most 6 items. "
            "Use depends_on as zero-based indices."
        )
        user_prompt = (
            f"Source: {message.source}\n"
            f"Type: {message.message_type}\n"
            f"Request:\n{message.content}"
        )

        api_base = self._task_generation_api_config.get("api_base")
        api_key = self._task_generation_api_config.get("api_key")

        try:
            raw = await llm_chat(
                llm_model=self._task_generation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_base=api_base,
                api_key=api_key,
                response_format={"type": "json_object"},
            )
            payload = extract_json(raw)
        except Exception:
            logger.exception(
                "Failed to generate planner action list with model %s",
                self._task_generation_model,
            )
            return None

        actions_raw = payload.get("actions")
        if not isinstance(actions_raw, list) or not actions_raw:
            return None

        normalized_actions: List[Dict[str, Any]] = []
        normalized_dependencies: List[List[int]] = []
        for idx, item in enumerate(actions_raw[: self.config.max_generated_actions]):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            depends_raw = item.get("depends_on")
            depends_on: List[int] = []
            if isinstance(depends_raw, list):
                for dep in depends_raw:
                    if isinstance(dep, int) and 0 <= dep < idx:
                        depends_on.append(dep)
            normalized_actions.append({"description": description})
            normalized_dependencies.append(depends_on)

        if not normalized_actions:
            return None

        criteria_raw = payload.get("completion_criteria")
        completion_criteria: List[str] = []
        if isinstance(criteria_raw, list):
            completion_criteria = [
                str(item).strip() for item in criteria_raw if str(item).strip()
            ][:3]

        if not completion_criteria:
            completion_criteria = ["Requester receives a complete and clear response."]

        return normalized_actions, completion_criteria, normalized_dependencies

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
        instructions: str = "",
        path: Optional[Sequence[str]] = None,
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
            instructions=instructions,
            path=list(path or []),
        )
        self.delegations[record.delegation_id] = record
        task_state = self.tasks.get(task_id)
        if task_state:
            task_state.waiting_on_delegation = record.delegation_id
            task_state.status = "waiting"
            task_state.metadata["delegation_path"] = record.path
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

    def choose_escalation_target(
        self,
        *,
        path: Sequence[str],
        current_agent: str,
        fallback: Optional[str] = None,
    ) -> str:
        """Choose the best escalation target walking the delegation path upstream."""

        candidates = [name for name in path if name]
        if current_agent in candidates:
            idx = candidates.index(current_agent)
            upstream = list(reversed(candidates[:idx]))
        else:
            upstream = list(reversed(candidates))

        for candidate in upstream:
            if self._is_human_agent(candidate):
                return candidate

        if upstream:
            return upstream[0]

        return fallback or current_agent

    def _is_human_agent(self, agent_name: str) -> bool:
        info = self.known_agents.get(agent_name, {}) if hasattr(self, "known_agents") else {}
        capabilities = info.get("capabilities") or {}
        if isinstance(capabilities, dict):
            role = str(
                capabilities.get("role")
                or capabilities.get("type")
                or capabilities.get("agent_type")
                or ""
            ).lower()
            if "human" in role:
                return True
        return "human" in agent_name.lower()

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
        reminder_content = record.instructions or (
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
        metadata: Optional[Dict[str, Any]] = None,
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
            metadata=metadata or {},
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
        metadata = record.metadata or {}
        action = metadata.get("action") or (
            "send_message" if metadata.get("target") else "plan"
        )
        if action == "send_message" and metadata.get("target"):
            target = metadata.get("target")
            message_text = metadata.get("message") or record.content
            prefix = metadata.get("prefix") or "Reminder: "
            meta_context = {
                "conversation_id": metadata.get("conversation_id"),
                "requested_by": metadata.get("requested_by"),
                "agent_name": self.agent.name,
                "delegation_path": metadata.get("delegation_path"),
            }
            try:
                await self.functions_registry.execute(
                    "send_message",
                    parameters={
                        "recipient": target,
                        "content": f"{prefix}{message_text}",
                        "message_type": "CHAT",
                    },
                    context=ToolExecutionContext(
                        agent=self.agent,
                        metadata=meta_context,
                    ),
                )
            except Exception:
                logger.exception("Failed to deliver reminder %s", record.reminder_id)
        else:
            instructions = metadata.get("message") or record.content
            synthetic_source = metadata.get("requested_by") or self.agent.name
            synthetic_message = Message(
                source=synthetic_source,
                to=self.agent.name,
                content=instructions,
                message_type=MessageType.CHAT,
            )
            try:
                await self.agent.handle_message(synthetic_message)
            except Exception:
                logger.exception("Failed to enqueue reminder task for %s", record.reminder_id)

        if not record.reschedule():
            self.reminders.pop(record.reminder_id, None)
        await self._save_state()
