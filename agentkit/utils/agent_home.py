from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def resolve_agent_home(agent_home: str, config_dir: Optional[Path] = None) -> Path:
    home = Path(agent_home).expanduser()
    if not home.is_absolute():
        base = config_dir or Path.cwd()
        home = (base / home).resolve()
    return home


def initialize_agent_home(agent_home: Path) -> None:
    agent_home.mkdir(parents=True, exist_ok=True)
    for relative in ("state", "tasks", "workspace", "logs"):
        (agent_home / relative).mkdir(parents=True, exist_ok=True)


def load_agent_system_prompt(agent_home: Path) -> str:
    prompt_path = agent_home / "AGENTS.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing AGENTS.md for agent home: {agent_home}"
        )
    return prompt_path.read_text(encoding="utf-8").strip()


def apply_agent_home_convention(
    agent_config: Dict[str, Any],
    *,
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if "agent_home" not in agent_config:
        raise ValueError(
            f"Agent '{agent_config.get('name', 'unknown')}' is missing required field 'agent_home'"
        )

    configured = dict(agent_config)
    home = resolve_agent_home(str(configured["agent_home"]), config_dir=config_dir)
    initialize_agent_home(home)

    configured["agent_home"] = str(home)
    configured["system_prompt"] = load_agent_system_prompt(home)
    configured.setdefault("planner_state_dir", str(home / "state"))
    configured.setdefault("workspace_dir", str(home / "workspace"))
    configured.setdefault("tasks_dir", str(home / "tasks"))
    configured.setdefault("logs_dir", str(home / "logs"))
    return configured
