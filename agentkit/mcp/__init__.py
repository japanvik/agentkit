"""
Model Context Protocol (MCP) integration utilities.

These helpers allow AgentKit agents to connect to MCP servers, discover the
tools they expose, and surface those tools through the AgentKit functions
registry so they can be invoked by LLM-driven planners.
"""

from .config import MCPServerConfig
from .manager import MCPClientManager

__all__ = ["MCPServerConfig", "MCPClientManager"]
