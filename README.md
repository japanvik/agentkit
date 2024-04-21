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

### Data Hub-Centric Communication

- **Unified Messaging Endpoint**: The Data Hub, a robust message handling endpoint, listens to incoming messages from any agent over HTTP. This centralized approach captures and manages all messages efficiently before broadcasting, using a Pub-Sub network for agents to pick up and act upon.

- **Pub-Sub Messaging Protocol**: Using ZeroMQ to create a publish-subscribe messaging system, the Data Hub not only receives messages but also broadcasts them across the network. This setup allows for decoupled, asynchronous, and scalable communication patterns, ideal for complex distributed systems.

- **Dynamic Agent Interaction via ZMQ**:
  - **ZMQMessageReceiver**: Agents subscribe to messages through the `ZMQMessageReceiver`, the subscriber endpoint within the pub-sub model. This component is crucial for agents that need to process broadcasted messages relevant to their roles.
  - **Real-time Message Distribution**: The Data Hub, upon receiving a message, immediately publishes it to the subscribed agents, enabling real-time response and interaction among agents. This capability is particularly useful where timing and speed of information dissemination are critical.

- **Scalable and Efficient**: ZeroMQ allows the system to handle high volumes of messages with minimal latency, enabling scalability to large networks of interacting agents.

- **Flexible and Configurable**: Agents can dynamically register or deregister with the `ZMQMessageReceiver`, allowing for flexible network configurations and adaptation to changing system conditions.

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

## Documentation

Still in the works. Refer to the source docstrings for documentation for now.

### Quick Start Guide

#### Running the Data Hub

The Data Hub is central to the AgentKit framework, facilitating communication between different agents. To start the Data Hub, use the following command:

```bash
python -m agentkit.datahub
```

This command initializes the Data Hub, which listens for incoming messages from agents and handles their distribution across the network using the Pub-Sub model. Ensure all required dependencies are installed and the Python environment is correctly set up to run this module.

#### Running the Example Chat Bot Agent

To run a chat bot agent that utilizes the Data Hub for sending and receiving messages, follow these steps:

1. **Download or Clone the Repository**:
   Ensure you have the latest version of the example files from the `examples` directory.

2. **Configure the Agent**:
   Examine and modify the configuration file (`simple_chat.json`) to fit your setup. By default, the agent is configured to interact with [Ollama](https://ollama.com/) running on localhost:

   ```bash
   python ./simple_chat_agent.py --config ./config/simple_chat.json
   ```

   Update the configuration settings for the model and identity as necessary. For using other LLMs like OpenAI's models, refer to the [Litellm documentation](https://docs.litellm.ai/docs/) for details on specifying API keys and model settings.

3. **Run the Chat Bot**:
   Execute the following command to start the chat bot:

   ```bash
   python ./simple_chat_agent.py --config ./config/simple_chat.json
   ```

   This script will initiate the chat bot that communicates through the Data Hub using the specified LLM settings.

#### Running the Example Human Agent

To interact with the LLM-powered chat bot, launch the human agent script:

1. **Start the Human Agent**:
   Run the following command, replacing `"Your Name"` with your desired agent name:

   ```bash
   python ./human_agent.py --name "Your Name"
   ```

2. **Interact with the Chat Bot**:
   Once the command line prompt appears, begin typing your messages. The chat bot will respond according to the capabilities defined in its configuration.

## Further Assistance

Should you need further assistance, consult the source code or [submit an issue](https://github.com/yourusername/agentkit/issues) on GitHub for support.

## Contributing

Contributions are welcome!

 Please review our issues section or send a PR on GitHub if you're interested in improving `AgentKit`.

## License

`AgentKit` is licensed under the MIT License. See the LICENSE file for more details.
