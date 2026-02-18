import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from networkkit.network import ZMQMessageReceiver

from agentkit.agents.task_aware_agent import TaskAwareAgent
from agentkit.brains.tool_brain import ToolBrain
from agentkit.memory.threaded_memory import ThreadedMemory
from agentkit.functions.functions_registry import DefaultFunctionsRegistry
from agentkit.utils.agent_home import apply_agent_home_convention


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return apply_agent_home_convention(raw, config_dir=path.parent.resolve())


async def create_agent(config: Dict[str, Any]) -> TaskAwareAgent:
    name = config["name"]
    description = config.get("description", "")
    model = config.get("model", "ollama/qwen3")
    system_prompt = config.get("system_prompt", "")
    user_prompt = config.get("user_prompt", "")
    api_config = config.get("api_config", {})

    memory = ThreadedMemory()
    functions_registry = DefaultFunctionsRegistry()

    brain = ToolBrain(
        name=name,
        description=description,
        model=model,
        memory_manager=memory,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_config=api_config,
        functions_registry=functions_registry,
    )

    agent = TaskAwareAgent(
        name=name,
        config=config,
        brain=brain,
        memory=memory,
        functions_registry=functions_registry,
    )
    return agent


async def run_sophia(config_path: Path, log_level: str) -> None:
    config = load_config(config_path)

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")

    agent = await create_agent(config)

    bus_ip = config.get("bus_ip", "127.0.0.1")
    receiver = ZMQMessageReceiver(subscribe_address=f"tcp://{bus_ip}:5555")
    receiver.register_subscriber(agent)

    shutdown_event = asyncio.Event()

    def _signal_handler(sig):
        logging.info("Received exit signal %s -- shutting down", sig.name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s))
        except NotImplementedError:
            logging.warning("Signal handlers not supported on this platform")

    receiver_task = asyncio.create_task(receiver.start(), name="SophiaReceiver")

    try:
        await agent.start()
        logging.info("Sophia is online and listening.")
        await shutdown_event.wait()
    finally:
        logging.info("Stopping Sophia...")
        await receiver.stop()
        if not receiver_task.done():
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
        await agent.stop()
        logging.info("Sophia has shut down cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sophia task-aware agent.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "sophia_task_agent.json"),
        help="Path to the agent configuration file",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_sophia(Path(args.config), args.loglevel))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
