import pytest

from agentkit.functions.functions_registry import DefaultFunctionsRegistry, ToolExecutionContext
from agentkit.mcp.config import MCPServerConfig
from agentkit.mcp import manager as manager_module
from agentkit.mcp.manager import MCPClientManager


class StubSession:
    def __init__(self, read_stream=None, write_stream=None, *, client_info=None, **_kwargs):
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.client_info = client_info
        self.initialized = False
        self.closed = False
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return {
            "tools": [
                {
                    "name": "ping",
                    "description": "Echo the provided message",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message to echo back",
                            }
                        },
                        "required": ["message"],
                    },
                }
            ]
        }

    async def call_tool(self, name, arguments=None):
        arguments = arguments or {}
        message = arguments.get("message", "")
        return {
            "content": [
                {"type": "text", "text": f"{name}:{message}"},
            ]
        }


class StubStdioContext:
    def __init__(self, params):
        self.params = params
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        # Return dummy read/write streams
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb):
        self.entered = False


class DummyAgent:
    def __init__(self):
        self.name = "Tester"
        self.config = {}


@pytest.fixture(autouse=True)
def patch_stdio_client(monkeypatch):
    monkeypatch.setattr(manager_module, "stdio_client", lambda params: StubStdioContext(params))
    monkeypatch.setattr(manager_module, "ClientSession", StubSession)


@pytest.mark.asyncio
async def test_manager_registers_and_invokes_tools():
    registry = DefaultFunctionsRegistry()
    config = MCPServerConfig(name="demo", command="stub-command")
    manager = MCPClientManager(agent_name="Tester", functions_registry=registry, server_configs=[config])

    await manager.start()

    assert "demo::ping" in registry.function_map

    agent = DummyAgent()
    context = ToolExecutionContext(agent=agent, metadata={"conversation_id": "conv-1"})
    result = await registry.execute("demo::ping", {"message": "hello"}, context=context)

    assert result["status"] == "completed"
    assert result["outputs"][0]["text"] == "ping:hello"
    assert result["server"] == "demo"

    await manager.stop()
