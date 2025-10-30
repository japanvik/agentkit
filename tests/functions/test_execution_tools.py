import asyncio
import os

import pytest

from agentkit.functions.execution_tools import (
    python_execution_tool,
    shell_command_tool,
)
from agentkit.functions.functions_registry import ToolExecutionContext


class DummyAgent:
    def __init__(self):
        self.name = "dummy"
        self.working_dir = os.getcwd()


@pytest.fixture
def tool_context():
    return ToolExecutionContext(agent=DummyAgent())


@pytest.mark.asyncio
async def test_python_execution_tool_success(tool_context):
    result = await python_execution_tool(
        tool_context,
        code='print("hello world")',
        timeout=5,
    )
    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]
    assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_python_execution_tool_timeout(tool_context):
    result = await python_execution_tool(
        tool_context,
        code="import time; time.sleep(1)",
        timeout=0.1,
    )
    assert result["status"] == "timeout"
    assert result["timed_out"] is True


@pytest.mark.asyncio
async def test_shell_command_tool_success(tool_context):
    result = await shell_command_tool(
        tool_context,
        command="echo 'shell tool'",
        timeout=5,
    )
    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert "shell tool" in result["stdout"]
