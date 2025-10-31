from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


@dataclass
class PlannerActionState:
    action_id: str
    description: str
    created_at: datetime = field(default_factory=utc_now)
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Optional[Dict] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "PlannerActionState":
        copied = dict(data)
        copied["created_at"] = datetime.fromisoformat(copied["created_at"])
        return cls(**copied)


@dataclass
class ReminderRecord:
    reminder_id: str
    description: str
    content: str
    recipient: str
    next_run: datetime
    repeat_interval: Optional[float] = None
    created_at: datetime = field(default_factory=utc_now)
    status: str = "scheduled"  # scheduled, completed, cancelled

    def reschedule(self) -> bool:
        if self.repeat_interval is None:
            self.status = "completed"
            return False
        self.next_run = utc_now() + timedelta(seconds=self.repeat_interval)
        return True

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["next_run"] = self.next_run.isoformat()
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "ReminderRecord":
        copied = dict(data)
        copied["next_run"] = datetime.fromisoformat(copied["next_run"])
        copied["created_at"] = datetime.fromisoformat(copied["created_at"])
        return cls(**copied)


@dataclass
class DelegationRecord:
    delegation_id: str
    task_id: str
    target_agent: str
    created_at: datetime = field(default_factory=utc_now)
    last_reminder: Optional[datetime] = None
    reminder_interval: float = 60.0
    reminder_multiplier: float = 2.0
    max_reminder_interval: float = 3600.0
    reminder_attempts: int = 0
    max_attempts: int = 6
    deadline: Optional[datetime] = None
    status: str = "waiting"  # waiting, completed, cancelled, timed_out

    def next_due(self) -> Optional[datetime]:
        if self.status != "waiting":
            return None
        if self.reminder_attempts >= self.max_attempts:
            return None
        base_time = self.last_reminder or self.created_at
        return base_time + timedelta(seconds=self.reminder_interval)

    def register_reminder(self) -> None:
        self.last_reminder = utc_now()
        self.reminder_attempts += 1
        self.reminder_interval = min(
            self.reminder_interval * self.reminder_multiplier,
            self.max_reminder_interval,
        )

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_reminder"] = (
            self.last_reminder.isoformat() if self.last_reminder else None
        )
        data["deadline"] = self.deadline.isoformat() if self.deadline else None
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "DelegationRecord":
        copied = dict(data)
        copied["created_at"] = datetime.fromisoformat(copied["created_at"])
        if copied.get("last_reminder"):
            copied["last_reminder"] = datetime.fromisoformat(copied["last_reminder"])
        if copied.get("deadline"):
            copied["deadline"] = datetime.fromisoformat(copied["deadline"])
        return cls(**copied)


@dataclass
class PlannerTaskState:
    task_id: str
    conversation_id: str
    origin: str
    status: str = "pending"  # pending, running, waiting, completed, failed, cancelled
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: Dict = field(default_factory=dict)
    actions: List[PlannerActionState] = field(default_factory=list)
    waiting_on_delegation: Optional[str] = None

    def add_action(self, description: str) -> PlannerActionState:
        action = PlannerActionState(action_id=str(uuid.uuid4()), description=description)
        self.actions.append(action)
        self.updated_at = utc_now()
        return action

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["actions"] = [action.to_dict() for action in self.actions]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "PlannerTaskState":
        copied = dict(data)
        copied["created_at"] = datetime.fromisoformat(copied["created_at"])
        copied["updated_at"] = datetime.fromisoformat(copied["updated_at"])
        copied["actions"] = [
            PlannerActionState.from_dict(a) for a in copied.get("actions", [])
        ]
        return cls(**copied)


class PlannerStateStore:
    """
    Simple JSON-backed persistence for planner state.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Dict]:
        if not self.path.exists():
            return {
                "tasks": {},
                "delegations": {},
                "known_agents": {},
                "reminders": {},
            }
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        tasks = {
            task_id: PlannerTaskState.from_dict(data)
            for task_id, data in raw.get("tasks", {}).items()
        }
        delegations = {
            delegation_id: DelegationRecord.from_dict(data)
            for delegation_id, data in raw.get("delegations", {}).items()
        }
        known_agents = raw.get("known_agents", {})
        reminders = {
            reminder_id: ReminderRecord.from_dict(data)
            for reminder_id, data in raw.get("reminders", {}).items()
        }
        return {
            "tasks": tasks,
            "delegations": delegations,
            "known_agents": known_agents,
            "reminders": reminders,
        }

    def save(
        self,
        *,
        tasks: Dict[str, PlannerTaskState],
        delegations: Dict[str, DelegationRecord],
        known_agents: Dict[str, Dict],
        reminders: Dict[str, "ReminderRecord"],
    ) -> None:
        data = {
            "tasks": {task_id: task.to_dict() for task_id, task in tasks.items()},
            "delegations": {
                deleg_id: record.to_dict()
                for deleg_id, record in delegations.items()
            },
            "known_agents": known_agents,
            "reminders": {
                reminder_id: record.to_dict()
                for reminder_id, record in reminders.items()
            },
        }
        temp_path = self.path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except FileNotFoundError:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        temp_path.replace(self.path)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()
