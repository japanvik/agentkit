"""
Configuration models for MCP server connections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MCPServerConfig:
    """
    Declarative configuration for a single MCP server.

    Attributes:
        name: Logical name for the server. Used as a namespace when exposing tools.
        transport: Transport implementation to use. Currently only ``stdio`` is
            supported, but additional transports (websocket, http) can be added
            in the future.
        command: Executable that should be launched for stdio transports.
        args: Optional command arguments.
        env: Optional environment variables that should be added/overridden when
            launching the MCP server process.
        namespace: Optional namespace prefix for published tools. If omitted the
            server ``name`` is used.
        metadata: Extra user-defined metadata available to tool execution
            handlers.
    """

    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MCPServerConfig":
        if "name" not in payload:
            raise ValueError("MCP server configuration requires a 'name' field")

        transport = payload.get("transport", "stdio")
        command = payload.get("command")
        if transport == "stdio" and not command:
            raise ValueError(
                f"MCP server '{payload['name']}' uses stdio transport but no 'command' was provided"
            )

        args = list(payload.get("args", []) or [])
        env = dict(payload.get("env", {}) or {})
        namespace = payload.get("namespace")
        metadata = dict(payload.get("metadata", {}) or {})

        return cls(
            name=payload["name"],
            transport=transport,
            command=command,
            args=args,
            env=env,
            namespace=namespace,
            metadata=metadata,
        )


def parse_server_configs(raw: Optional[Iterable[Dict[str, Any]]]) -> List[MCPServerConfig]:
    """
    Safely parse a collection of configuration dictionaries.
    """
    configs: List[MCPServerConfig] = []
    if not raw:
        return configs

    for entry in raw:
        configs.append(MCPServerConfig.from_dict(entry))
    return configs
