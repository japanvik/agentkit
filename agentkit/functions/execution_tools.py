"""
Execution-related built-in tools.

This module provides tools that allow agents to execute shell commands or run
Python code in a controlled, timeout-aware manner.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from typing import Any, Dict, List, Optional

from agentkit.functions.functions_registry import ToolExecutionContext

DEFAULT_TIMEOUT_SECONDS = 15.0


async def _run_subprocess(
    command: List[str],
    *,
    timeout: float,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Helper to run a subprocess with a timeout and capture output."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        status = "completed"
        timed_out = False
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        stdout, stderr = b"", b""
        status = "timeout"
        timed_out = True

    return {
        "status": status,
        "timed_out": timed_out,
        "exit_code": process.returncode if not timed_out else None,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


async def python_execution_tool(
    context: ToolExecutionContext,
    code: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    python_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a snippet of Python code in a subprocess.

    Args:
        context: Tool execution context (currently unused, reserved for future policy checks).
        code: The Python code to execute.
        timeout: Maximum execution time in seconds before the process is killed.
        python_path: Optional path to the python interpreter. Defaults to sys.executable.

    Returns:
        Dictionary containing execution status, stdout, stderr, and exit_code.
    """
    interpreter = python_path or sys.executable
    return await _run_subprocess(
        [interpreter, "-c", code],
        timeout=timeout,
    )


async def shell_command_tool(
    context: ToolExecutionContext,
    command: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a shell command using bash -lc.

    Args:
        context: Tool execution context (currently unused, reserved for policy / auditing).
        command: The shell command to execute.
        timeout: Maximum execution time in seconds before the process is killed.
        working_dir: Optional working directory for the command.

    Returns:
        Dictionary containing execution status, stdout, stderr, and exit_code.
    """
    cwd = working_dir or getattr(context.agent, "working_dir", None) or os.getcwd()
    return await _run_subprocess(
        ["bash", "-lc", command],
        timeout=timeout,
        cwd=cwd,
    )
