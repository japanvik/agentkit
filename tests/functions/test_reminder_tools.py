import asyncio
from pathlib import Path

import pytest

from agentkit.functions.functions_registry import DefaultFunctionsRegistry, FunctionDescriptor, ParameterDescriptor, ToolExecutionContext
from agentkit.functions.reminder_tools import schedule_reminder_tool
from agentkit.planning import AgentPlanner, PlannerConfig


class DummyAgent:
    def __init__(self, tmp_path: Path):
        self.name = "ReminderAgent"
        self._tasks = []
        self.last_message = None
        self.functions_registry = DefaultFunctionsRegistry()
        self.planner = AgentPlanner(
            self,
            config=PlannerConfig(persistence_dir=tmp_path),
            functions_registry=self.functions_registry,
        )

    def create_background_task(self, coro, name=None):
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        return task


@pytest.fixture
def agent(tmp_path):
    agent = DummyAgent(tmp_path)

    async def fake_send_message(context, recipient: str, content: str, message_type: str = "CHAT"):
        context.agent.last_message = {
            "recipient": recipient,
            "content": content,
            "message_type": message_type,
        }
        return {"status": "sent"}

    agent.functions_registry.register_function(
        fake_send_message,
        FunctionDescriptor(
            name="send_message",
            description="Send message",
            parameters=[
                ParameterDescriptor(name="recipient", description="target", required=True),
                ParameterDescriptor(name="content", description="content", required=True),
                ParameterDescriptor(name="message_type", description="type", required=False),
            ],
        ),
        pass_context=True,
    )
    return agent


@pytest.mark.asyncio
async def test_schedule_reminder_tool(agent):
    context = ToolExecutionContext(agent=agent)
    result = await schedule_reminder_tool(
        context,
        content="Reminder message",
        delay_seconds=0.1,
        recipient="human",
    )
    assert result["status"] == "scheduled"
    reminder_id = result["reminder_id"]
    record = agent.planner.reminders[reminder_id]
    record.next_run = record.next_run.replace(year=record.next_run.year - 1)
    await agent.planner._process_reminders()
    assert agent.last_message["recipient"] == "human"
    assert agent.last_message["message_type"] == "CHAT"
    assert agent.last_message["content"].endswith("Reminder message")


@pytest.mark.asyncio
async def test_schedule_reminder_tool_accepts_z_suffix(agent):
    context = ToolExecutionContext(agent=agent)
    result = await schedule_reminder_tool(
        context,
        content="UTC reminder",
        run_at="2025-11-01T17:45:00Z",
        recipient="human",
        description="Test ISO 8601 with Z",
    )
    assert result["status"] == "scheduled"
    reminder_id = result["reminder_id"]
    record = agent.planner.reminders[reminder_id]
    assert record.next_run.tzinfo is None
    assert record.next_run.isoformat() == "2025-11-01T17:45:00"
