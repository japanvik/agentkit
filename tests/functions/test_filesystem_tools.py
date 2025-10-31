import asyncio
from pathlib import Path

import pytest

from agentkit.functions.filesystem_tools import (
    FilesystemAccessError,
    list_directory_tool,
    read_file_tool,
    write_file_tool,
)
from agentkit.functions.functions_registry import ToolExecutionContext


class DummyAgent:
    def __init__(self, root: Path):
        self.name = "dummy"
        self.config = {"filesystem_root": str(root)}
        self._filesystem_root = root


@pytest.fixture
def fs_root(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_list_directory(fs_root):
    agent = DummyAgent(fs_root)
    context = ToolExecutionContext(agent=agent)
    result = await list_directory_tool(context, path="", recursive=False)
    names = {entry["name"] for entry in result["entries"]}
    assert "hello.txt" in names
    assert "subdir" in names


@pytest.mark.asyncio
async def test_read_file(fs_root):
    agent = DummyAgent(fs_root)
    context = ToolExecutionContext(agent=agent)
    result = await read_file_tool(context, path="hello.txt")
    assert result["content"] == "hello world"


@pytest.mark.asyncio
async def test_write_file(fs_root):
    agent = DummyAgent(fs_root)
    context = ToolExecutionContext(agent=agent)
    await write_file_tool(context, path="output.txt", content="data")
    assert (fs_root / "output.txt").read_text(encoding="utf-8") == "data"
    await write_file_tool(context, path="output.txt", content="+more", mode="append")
    assert (fs_root / "output.txt").read_text(encoding="utf-8") == "data+more"


@pytest.mark.asyncio
async def test_access_outside_root_raises(fs_root):
    agent = DummyAgent(fs_root)
    context = ToolExecutionContext(agent=agent)
    with pytest.raises(FilesystemAccessError):
        await read_file_tool(context, path="../outside.txt")
