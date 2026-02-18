# AgentKit

## Overview

`AgentKit` is a simple framework designed to facilitate the networked communication between agents in distributed systems. It enables the creation, management, and coordination of intelligent agents over a network, fostering complex data-driven decision processes and automated tasks with minimal human intervention. This toolkit, built with Python, focuses on ease-of-use, efficiency, and scalability, making it ideal for applications beyond conversational agents, but also in IoT, smart home systems, and automation tasks.

## Installation

Install `AgentKit` using pip:

```bash
pip install agentkit
```

Ensure you have Python 3.10 or higher installed to meet the compatibility requirements.

## Features

### Simplified Agent Framework

- **Easy to Create and Manage Agents**: The framework facilitates lightweight and straightforward implementation of agents, with a simple interface for sending and receiving messages. AgentKit is suitable for developers seeking to build systems with complex interactions but straightforward code.

- **Asynchronous Operation Supported**: The architecture supports asynchronous operations, essential for maintaining high performance in distributed systems where blocking operations can create bottlenecks.

- **Built-in Logging and Monitoring**: Integrated logging tracks message flows and system activity, providing valuable insights during development and troubleshooting phases.

### Model Context Protocol (MCP) Integration

- **Per-agent MCP tool catalogs**: Each agent can declare its own list of MCP servers. Tools discovered from those servers are automatically surfaced through the AgentKit functions registry, allowing LLM-driven planners to call them like any other function.
- **Namespace isolation**: Tools are exposed using a `<server>::<tool>` naming convention so that specialized agents can connect to different MCP stacks without name collisions.
- **Powered by the official SDK**: AgentKit now depends on the `mcp` package, ensuring first-class support for the Model Context Protocol standard and future enhancements.

Sample snippet from an agent configuration:

```json
"mcp_servers": [
  {
    "name": "filesystem",
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "mcp_servers.fs"],
    "namespace": "fs"
  }
]
```

## Why AgentKit?

AgentKit simplifies the creation and management of networked agent systems, making it an invaluable tool for a variety of innovative applications:

### Real-Time Data Processing

Enables agents to collect, process, and react to real-time data across various domains. This is crucial for environments where timely data analysis and decision-making are key.

### Automated and Intelligent Decision Making

Facilitates autonomous systems that learn from data to make decisions independently of human input, optimizing processes through dynamic response to environmental variables.

### Enhanced Connectivity and Flexibility

Allows for the deployment of agents across diverse and geographically dispersed environments, making it possible to manage complex systems from centralized or various locations.

### Research and Development

Aids in the simulation and modeling of complex systems to facilitate research and theoretical analysis without physical constraints.

## Getting Started

Refer to the `examples` directory to see a simple chat agent implementation. Comprehensive documentation will be provided soon!

### Using the AgentRunner Utility

For simplified agent creation and lifecycle management, AgentKit provides the `AgentRunner` utility:

```python
import asyncio
from pathlib import Path
from agentkit.utils import AgentRunner

async def main():
    runner = AgentRunner(
        name="Julia",
        description="A friendly AI assistant",
        model="ollama/qwen3",
        agent_home=str(Path("agentkit/examples/agent_homes/Julia").resolve())
    )
    
    await runner.run()

if __name__ == "__main__":
    asyncio.run(main())
```

This utility handles:
- Agent creation with sensible defaults
- Message receiver setup and registration
- Signal handling for graceful shutdown
- Proper resource cleanup

TODO: Add more examples of AgentRunner configuration options

### Agent Home Convention

AgentKit now uses a per-agent home directory for durable state, working files, and prompt instructions.

Each configured agent must provide `agent_home`, and that directory must contain an `AGENTS.md` file. The content of `AGENTS.md` is loaded as the system prompt.

Expected layout:

```text
<agent_home>/
  AGENTS.md
  state/
  tasks/
  workspace/
  logs/
```

The runtime ensures these subdirectories exist when the agent is loaded.

Example `sophia_task_agent.json`:

```json
{
  "name": "Sophia",
  "description": "A thoughtful task-aware assistant who can plan, delegate, and execute tools.",
  "agent_home": "../agent_homes/Sophia",
  "model": "ollama/qwen3-coder:480b-cloud",
  "planner_model": "ollama/qwen3-coder:480b-cloud",
  "user_prompt": "Conversation history:\n{context}\nSophia:",
  "bus_ip": "127.0.0.1"
}
```

## Documentation

Still in the works. Refer to the source docstrings for documentation for now.

### Quick Start Guide

#### Running the Data Bus

The [Networkkit](https://github.com/japanvik/networkkit) project which spun off from this project has a Data bus implementation that handles the communication aspects of the agents. To start the Data Bus, use the following command:

```bash
python -m networkkit.databus
```

This command initializes the Data Hub, which listens for incoming messages from agents and handles their distribution across the network using the Pub-Sub model. Networkkit should be installed as part of a dependancy of Agentkit.

#### Running the Example Chat Bot Agent

To run a chat bot agent that utilizes the Data Bub for sending and receiving messages, follow these steps:

1. **Download or Clone the Repository**:
   Ensure you have the latest version of the example files from the `examples` directory.

2. **Configure the Agent**:
   Examine and modify the configuration file (`simple_chat.json`) to fit your setup. By default, the agent is configured to interact with [Ollama](https://ollama.com/) running on localhost. Ensure each agent has an `agent_home` path and that `AGENTS.md` exists there.

   Update the configuration settings for the model and identity as necessary. For using other LLMs like OpenAI's models, refer to the [Litellm documentation](https://docs.litellm.ai/docs/) for details on specifying API keys and model settings.

3. **Run the Chat Bot**:
   Execute the following command to start the chat bot:

   ```bash
   python ./config_agent.py --config ./config/simple_chat.json
   ```

   This script will initiate the chat bot that communicates through the Data Hub using the specified LLM settings.

#### Running the Example Human Agent

To interact with the LLM-powered chat bot above, launch the agent with the human_agent.json as the configuration file so you can send it chat messages and interact with it:

   ```bash
   python ./config_agent.py --config ./config/human_agent.json --loglevel WARN
   ```

   Once the prompt appears, you can use a multi-agent chat-room style interface:

   - Plain text sends to the current channel (broadcast as `ALL`)
   - `/join <channel>` switches channels (default `#general`)
   - `/dm <agent> <message>` sends a direct message
   - `/reply <message>` replies to the most recent sender
   - `/agents` lists discovered agents
   - `/history [N]` shows recent timeline entries
   - `/help` shows all commands

   Japanese input/output is supported out of the box (UTF-8).

## Further Assistance

Should you need further assistance, consult the source code or [submit an issue](https://github.com/yourusername/agentkit/issues) on GitHub for support.

## Contributing

Contributions are welcome!

 Please review our issues section or send a PR on GitHub if you're interested in improving `AgentKit`.

## License

`AgentKit` is licensed under the MIT License. See the LICENSE file for more details.
