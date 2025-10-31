"""Filesystem-related tools for agents."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import Dict, List, Optional

from agentkit.functions.functions_registry import ToolExecutionContext


class FilesystemAccessError(Exception):
    """Raised when a filesystem operation is not permitted."""


def _resolve_path(base: Optional[Path], requested: Optional[str]) -> Path:
    if base is None:
        target = Path(requested).expanduser() if requested else Path.cwd()
    else:
        base = base.resolve()
        target = Path(requested).expanduser() if requested else base
        if not target.is_absolute():
            target = (base / target).resolve()
        else:
            target = target.resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise FilesystemAccessError("Path is outside allowed filesystem root") from exc
    return target


async def list_directory_tool(
    context: ToolExecutionContext,
    path: Optional[str] = None,
    pattern: Optional[str] = None,
    recursive: bool = False,
) -> Dict[str, object]:
    agent = context.agent
    root = getattr(agent, "_filesystem_root", None)
    target = _resolve_path(root, path)

    if not target.exists():
        raise FilesystemAccessError(f"Directory does not exist: {target}")
    if not target.is_dir():
        raise FilesystemAccessError(f"Path is not a directory: {target}")

    loop = asyncio.get_running_loop()

    def _collect() -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        iterator = target.rglob(pattern or "*") if recursive and pattern else (
            target.rglob("*") if recursive else target.iterdir()
        )
        for entry in iterator:
            if pattern and not recursive and not fnmatch.fnmatch(entry.name, pattern):
                continue
            stat = entry.stat()
            entries.append(
                {
                    "path": str(entry.relative_to(root)),
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        return entries

    entries = await loop.run_in_executor(None, _collect)
    return {
        "status": "ok",
        "root": str(root) if root else None,
        "directory": str(target if root is None else target.relative_to(root)),
        "entries": entries,
    }


async def read_file_tool(
    context: ToolExecutionContext,
    path: str,
    max_bytes: Optional[int] = None,
    encoding: str = "utf-8",
) -> Dict[str, object]:
    if not path:
        raise ValueError("Path is required")

    agent = context.agent
    root = getattr(agent, "_filesystem_root", None)
    target = _resolve_path(root, path)

    if not target.exists() or not target.is_file():
        raise FilesystemAccessError("File does not exist")

    loop = asyncio.get_running_loop()

    def _read() -> str:
        with target.open("r", encoding=encoding) as fh:
            if max_bytes is not None:
                return fh.read(max_bytes)
            return fh.read()

    content = await loop.run_in_executor(None, _read)
    return {
        "status": "ok",
        "path": str(target if root is None else target.relative_to(root)),
        "content": content,
    }


async def write_file_tool(
    context: ToolExecutionContext,
    path: str,
    content: str,
    mode: str = "overwrite",
    encoding: str = "utf-8",
) -> Dict[str, object]:
    if not path:
        raise ValueError("Path is required")

    agent = context.agent
    root = getattr(agent, "_filesystem_root", None)
    target = _resolve_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be either 'overwrite' or 'append'")

    loop = asyncio.get_running_loop()

    def _write() -> None:
        write_mode = "a" if mode == "append" else "w"
        with target.open(write_mode, encoding=encoding) as fh:
            fh.write(content)

    await loop.run_in_executor(None, _write)
    return {
        "status": "ok",
        "path": str(target if root is None else target.relative_to(root)),
        "mode": mode,
        "bytes_written": len(content.encode(encoding)),
    }
