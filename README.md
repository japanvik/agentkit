# AgentKit

## Overview

`AgentKit` is a simple framework designed to facilitate the networked communication between agents in distributed systems. It enables the creation, management, and coordination of intelligent agents over a network, fostering complex data-driven decision processes and automated tasks with minimal human intervention. This toolkit, built with Python, focuses on ease-of-use, efficiency, and scalability, making it ideal for applications beyond conversational agents, but also in IoT, smart home systems, and automation tasks.

## Installation

Install `AgentKit` using pip:

```bash
pip install agentkit
```

Ensure you have Python 3.8 or higher installed to meet the compatibility requirements.

## Features

### Simplified Agent Framework

- **Easy to Create and Manage Agents**: The framework facilitates lightweight and straightforward implementation of agents, with a simple interface for sending and receiving messages. AgentKit is suitable for developers seeking to build systems with complex interactions but straightforward code.

- **Asynchronous Operation Supported**: The architecture supports asynchronous operations, essential for maintaining high performance in distributed systems where blocking operations can create bottlenecks.

- **Built-in Logging and Monitoring**: Integrated logging tracks message flows and system activity, providing valuable insights during development and troubleshooting phases.

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
from agentkit.utils import AgentRunner

async def main():
    runner = AgentRunner(
        name="Julia",
        description="A friendly AI assistant",
        model="ollama/qwen3",
        system_prompt="You are a friendly and helpful AI agent."
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
   Examine and modify the configuration file (`simple_chat.json`) to fit your setup. By default, the agent is configured to interact with [Ollama](https://ollama.com/) running on localhost.

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

   Once the command line prompt appears, begin typing your messages. The chat bot will respond according to the capabilities defined in its configuration.

## Further Assistance

Should you need further assistance, consult the source code or [submit an issue](https://github.com/yourusername/agentkit/issues) on GitHub for support.

## Contributing

Contributions are welcome!

 Please review our issues section or send a PR on GitHub if you're interested in improving `AgentKit`.

## License

`AgentKit` is licensed under the MIT License. See the LICENSE file for more details.
