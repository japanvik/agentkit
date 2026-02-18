"""
Utility modules for AgentKit.

This package contains utility classes and functions that simplify
working with AgentKit components.
"""

from agentkit.utils.agent_home import apply_agent_home_convention

__all__ = ["AgentRunner", "apply_agent_home_convention"]


def __getattr__(name):
    if name == "AgentRunner":
        from agentkit.utils.agent_runner import AgentRunner

        return AgentRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
