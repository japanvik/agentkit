import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from networkkit.messages import Message, MessageType

from agentkit.functions.functions_registry import DefaultFunctionsRegistry
from agentkit.functions.execution_tools import python_execution_tool
from agentkit.planning import AgentPlanner, PlannerConfig
from agentkit.functions.functions_registry import FunctionDescriptor, ParameterDescriptor


class DummyAgent:
    def __init__(self, name: str = "PlannerAgent"):
        self.name = name
        self.config = {}
        self._tasks = []
        self.last_message = None

    async def handle_message(self, message: Message):
        self.last_message = {
            "recipient": message.to,
            "content": message.content,
            "message_type": message.message_type.name
            if hasattr(message.message_type, "name")
            else str(message.message_type),
        }

    def create_background_task(self, coro, name=None):
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        return task


@pytest.fixture
def functions_registry():
    registry = DefaultFunctionsRegistry()

    async def fake_send_message(context, recipient: str, content: str, message_type: str = "CHAT"):
        context.agent.last_message = {
            "recipient": recipient,
            "content": content,
            "message_type": message_type,
        }
        return {"status": "sent"}

    registry.register_function(
        fake_send_message,
        FunctionDescriptor(
            name="send_message",
            description="Send a message",
            parameters=[
                ParameterDescriptor(name="recipient", description="target", required=True),
                ParameterDescriptor(name="content", description="text", required=True),
                ParameterDescriptor(name="message_type", description="type", required=False),
            ],
        ),
        pass_context=True,
    )
    registry.register_function(
        python_execution_tool,
        FunctionDescriptor(
            name="python_execute",
            description="Run python",
            parameters=[
                ParameterDescriptor(name="code", description="code", required=True),
            ],
        ),
        pass_context=True,
    )
    return registry


@pytest.fixture
def planner(tmp_path: Path, functions_registry):
    agent = DummyAgent()
    config = PlannerConfig(persistence_dir=tmp_path)
    planner = AgentPlanner(agent, config=config, functions_registry=functions_registry)
    return planner


@pytest.mark.asyncio
async def test_plan_defaults_to_brain(planner):
    message = Message(
        source="human",
        to="agent",
        content="Hello there",
        message_type=MessageType.CHAT,
    )
    action = await planner.plan_for_message(message, conversation_id="conv1")
    assert action["action_type"] == "use_brain"


@pytest.mark.asyncio
async def test_notify_tool_result_returns_send_message(planner):
    message = Message(
        source="human",
        to="agent",
        content="What time?",
        message_type=MessageType.CHAT,
    )
    await planner.plan_for_message(message, conversation_id="conv1")
    action = await planner.notify_tool_result(
        conversation_id="conv1",
        result={"stdout": "2025-10-30T12:00:00"},
        original_message=message,
    )
    assert action["tool_name"] == "send_message"
    assert "2025-10-30" in action["parameters"]["content"]


@pytest.mark.asyncio
async def test_delegation_reminder(planner):
    record = await planner.schedule_delegation(task_id="task1", target_agent="peer")
    assert record.delegation_id in planner.delegations
    await planner._send_reminder(record)
    updated = planner.delegations[record.delegation_id]
    assert updated.reminder_attempts == 1


@pytest.mark.asyncio
async def test_schedule_self_reminder(planner):
    reminder = await planner.schedule_reminder(
        content="Ping",
        recipient="human",
        delay_seconds=0.1,
    )
    record = planner.reminders[reminder.reminder_id]
    # Force trigger
    record.next_run = record.next_run - timedelta(seconds=1)
    await planner._process_reminders()
    assert planner.agent.last_message["content"] == "Ping"
    assert reminder.reminder_id not in planner.reminders


@pytest.mark.asyncio
async def test_schedule_recurring_reminder(planner):
    reminder = await planner.schedule_reminder(
        content="Repeat",
        recipient="human",
        delay_seconds=0.1,
        repeat_seconds=0.5,
    )
    record = planner.reminders[reminder.reminder_id]
    original_next = record.next_run
    record.next_run = record.next_run - timedelta(seconds=1)
    await planner._process_reminders()
    updated = planner.reminders[reminder.reminder_id]
    assert updated.next_run > original_next
    assert updated.status == "scheduled"


@pytest.mark.asyncio
async def test_plan_generates_actions_with_planner_model(tmp_path: Path, functions_registry, monkeypatch):
    agent = DummyAgent()
    config = PlannerConfig(
        persistence_dir=tmp_path,
        task_generation_model="planner-model",
    )
    planner = AgentPlanner(agent, config=config, functions_registry=functions_registry)

    async def fake_llm_chat(*, llm_model, messages, api_base=None, api_key=None, response_format=None):
        assert llm_model == "planner-model"
        return (
            '{"actions":[{"description":"Inspect request context","depends_on":[]},'
            '{"description":"Draft final response","depends_on":[0]}],'
            '"completion_criteria":["User gets complete answer"]}'
        )

    monkeypatch.setattr("agentkit.planning.planner.llm_chat", fake_llm_chat)

    message = Message(
        source="human",
        to="agent",
        content="Please summarize latest logs.",
        message_type=MessageType.CHAT,
    )
    action = await planner.plan_for_message(message, conversation_id="conv-plan")
    assert action["action_type"] == "use_brain"

    task_state = next(iter(planner.tasks.values()))
    assert [a.description for a in task_state.actions] == [
        "Inspect request context",
        "Draft final response",
    ]
    assert task_state.metadata["completion_criteria"] == ["User gets complete answer"]
    assert task_state.metadata["action_dependencies"] == [[], [0]]
    assert task_state.metadata["planning_model"] == "planner-model"


@pytest.mark.asyncio
async def test_plan_uses_fallback_action_without_planner_model(planner):
    message = Message(
        source="human",
        to="agent",
        content="Any update?",
        message_type=MessageType.CHAT,
    )
    await planner.plan_for_message(message, conversation_id="conv-fallback")
    task_state = next(iter(planner.tasks.values()))
    assert len(task_state.actions) == 1
    assert task_state.actions[0].description == "Respond to the requester with a complete answer."
