"""
Runtime management for MCP server connections.
"""
import asyncio
import logging
from contextlib import AsyncExitStack
from functools import partial
from typing import Any, Dict, List, Optional

from agentkit.functions.functions_registry import (
    FunctionDescriptor,
    ParameterDescriptor,
    ToolExecutionContext,
)
from agentkit.mcp.config import MCPServerConfig, parse_server_configs
from mcp import Implementation, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPIntegrationError(RuntimeError):
    """Raised when an MCP server cannot be connected."""


class MCPClientManager:
    """
    Handles the lifecycle of MCP server connections for an agent.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        functions_registry,
        server_configs: Optional[List[MCPServerConfig]] = None,
    ) -> None:
        self.agent_name = agent_name
        self.functions_registry = functions_registry
        self.server_configs = server_configs or []
        self._connections: Dict[str, MCPServerConnection] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, agent, functions_registry, raw_configs: Optional[List[Dict[str, Any]]] = None):
        configs = parse_server_configs(raw_configs)
        if not configs:
            return None
        return cls(agent_name=agent.name, functions_registry=functions_registry, server_configs=configs)

    async def start(self) -> None:
        async with self._lock:
            if not self.server_configs:
                return
            if self._connections:
                return

            for config in self.server_configs:
                connection = MCPServerConnection(functions_registry=self.functions_registry, config=config)
                await connection.connect(agent_name=self.agent_name)
                self._connections[config.name] = connection

    async def stop(self) -> None:
        async with self._lock:
            if not self._connections:
                return

            await asyncio.gather(*(conn.close() for conn in self._connections.values()), return_exceptions=True)
            self._connections.clear()

    @property
    def connections(self) -> Dict[str, "MCPServerConnection"]:
        return dict(self._connections)


class MCPServerConnection:
    """
    Represents a single MCP server session and the tools it exposes.
    """

    def __init__(self, *, functions_registry, config: MCPServerConfig) -> None:
        self.config = config
        self.functions_registry = functions_registry
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._registered_tools: List[str] = []

    async def connect(self, *, agent_name: str) -> None:
        if self._session:
            return

        if self.config.transport != "stdio":
            raise MCPIntegrationError(
                f"Unsupported MCP transport '{self.config.transport}' for server '{self.config.name}'"
            )

        params = StdioServerParameters(
            command=self.config.command or "",
            args=self.config.args,
            env=self.config.env or None,
        )
        client_context = stdio_client(params)

        exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream = await exit_stack.enter_async_context(client_context)
            client_info = Implementation(name=agent_name, version="0.0.0", title=f"{agent_name} MCP Client")
            session = ClientSession(read_stream, write_stream, client_info=client_info)
            session = await exit_stack.enter_async_context(session)
            await session.initialize()
        except Exception as exc:
            await exit_stack.aclose()
            raise MCPIntegrationError(f"Failed to start MCP stdio client '{self.config.name}': {exc}") from exc

        try:
            await self._register_tools(session)
        except Exception:
            await exit_stack.aclose()
            raise

        self._session = session
        self._exit_stack = exit_stack

    async def close(self) -> None:
        if not self._session or not self._exit_stack:
            return
        try:
            await self._exit_stack.aclose()
        except Exception:  # pragma: no cover
            logger.exception("Error while shutting down MCP server '%s'", self.config.name)
        self._session = None
        self._exit_stack = None

    async def _register_tools(self, session: ClientSession) -> None:
        listing = await session.list_tools()
        tools = getattr(listing, "tools", None)
        if tools is None:
            tools = listing.get("tools", []) if isinstance(listing, dict) else listing or []
        namespace = self.config.namespace or self.config.name

        for tool in tools:
            tool_name = _get(tool, "name")
            if not tool_name:
                continue

            exposed_name = f"{namespace}::{tool_name}"
            if self.functions_registry.has_function(exposed_name):
                logger.warning(
                    "Skipping MCP tool '%s' from server '%s' because a function with the same name already exists",
                    exposed_name,
                    self.config.name,
                )
                continue

            descriptor = self._build_function_descriptor(namespace=namespace, tool=tool)
            handler = partial(self._invoke_tool, session=session, tool=tool, exposed_name=exposed_name)

            try:
                self.functions_registry.register_function(handler, descriptor, pass_context=True)
            except Exception:
                logger.exception("Failed to register MCP tool '%s' from server '%s'", tool_name, self.config.name)
                continue

            self._registered_tools.append(exposed_name)
            logger.info(
                "Registered MCP tool '%s' from server '%s' as '%s'",
                tool_name,
                self.config.name,
                exposed_name,
            )

    def _build_function_descriptor(self, *, namespace: str, tool) -> FunctionDescriptor:
        tool_name = _get(tool, "name") or "unknown"
        description = _get(tool, "description") or f"MCP tool '{tool_name}' from server '{self.config.name}'"
        parameters = self._build_parameters(tool)
        return FunctionDescriptor(
            name=f"{namespace}::{tool_name}",
            description=description,
            parameters=parameters,
            categories=["mcp", namespace],
        )

    def _build_parameters(self, tool) -> List[ParameterDescriptor]:
        schema = (
            _get(tool, "input_schema")
            or _get(tool, "inputSchema")
            or {}
        )
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        if not isinstance(schema, dict):
            return []

        properties = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        params: List[ParameterDescriptor] = []
        for name, prop in properties.items():
            description = ""
            if isinstance(prop, dict):
                description = prop.get("description") or prop.get("title") or ""
                if not description and "type" in prop:
                    description = f"{prop['type']} value"
            params.append(
                ParameterDescriptor(
                    name=name,
                    description=description or "No description provided",
                    required=name in required,
                )
            )
        return params

    async def _invoke_tool(
        self,
        context: ToolExecutionContext,
        *,
        session: ClientSession,
        tool,
        exposed_name: str,
        **arguments: Any,
    ) -> Dict[str, Any]:
        tool_name = _get(tool, "name") or exposed_name

        metadata = {
            "server": self.config.name,
            "tool": tool_name,
            "namespace": self.config.namespace or self.config.name,
            "agent": context.agent.name,
        }
        metadata.update(self.config.metadata or {})
        if context.metadata:
            metadata["request_metadata"] = context.metadata

        payload = {k: v for k, v in arguments.items() if v is not None}

        result = await session.call_tool(tool_name, payload)
        normalized = self._normalize_result(result, tool_name=tool_name)
        normalized.setdefault("metadata", {})
        normalized["metadata"].update(metadata)
        return normalized

    def _normalize_result(self, result: Any, *, tool_name: str) -> Dict[str, Any]:
        if result is None:
            return {
                "status": "completed",
                "tool": tool_name,
                "server": self.config.name,
                "outputs": [],
            }

        data = result
        if hasattr(result, "model_dump"):
            try:
                data = result.model_dump()
            except Exception:  # pragma: no cover
                data = result
        elif hasattr(result, "__dict__") and not isinstance(result, dict):
            data = result.__dict__

        outputs = []
        content = None
        if isinstance(data, dict):
            content = data.get("content")

        if isinstance(content, list):
            for item in content:
                if hasattr(item, "model_dump"):
                    try:
                        outputs.append(item.model_dump())
                        continue
                    except Exception:  # pragma: no cover
                        pass
                if isinstance(item, dict):
                    outputs.append(item)
                elif hasattr(item, "__dict__"):
                    outputs.append(item.__dict__)
                else:
                    outputs.append({"type": "text", "text": str(item)})

        return {
            "status": "completed",
            "tool": tool_name,
            "server": self.config.name,
            "outputs": outputs,
            "raw": data,
        }


def _get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
