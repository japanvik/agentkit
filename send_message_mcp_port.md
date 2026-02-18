# Porting the AgentKit `send_message` Tool to a NetworkKit MCP Server

## 1. Goals
- Expose AgentKit’s built-in `send_message` capability as an MCP server owned by NetworkKit.
- Preserve the existing agent-facing contract (parameters, defaults, result shape) so AgentKit planners and reminders continue to function with minimal changes.
- Centralize transport-specific logic (HTTP/ZMQ bus) inside NetworkKit, reducing AgentKit’s direct dependency surface.

## 2. Current AgentKit Implementation (for reference)
- **Tool handler** – `agentkit/functions/built_in_tools.py:17` defines `send_message_tool(context, recipient, content, message_type="CHAT")`.
- **Dependencies** – Uses `networkkit.messages.Message`, `MessageType`, and `networkkit.network.HTTPMessageSender`; obtains the calling agent through `ToolExecutionContext`.
- **Message sender bootstrap** – Fallback creates an `HTTPMessageSender` using `agent.config["bus_publish_address"]` or `bus_ip` when `_message_sender` is missing (`agentkit/functions/built_in_tools.py:38-58`).
- **Dispatch path** – Builds a `Message`, then delegates to `agent._internal_send_message(message)` which relays via `_message_sender` if present (`agentkit/agents/base_agent.py:200-218`).
- **Return payload** – Returns `{status="sent", message_id=<uuid>, recipient, message_type}` to satisfy planners/reminders (`agentkit/functions/built_in_tools.py:73-78`).

### Why this matters
- Tool semantics (inputs/outputs) set expectations for planners, reminders, and task flows.
- AgentKit currently owns transport bootstrap; moving this into NetworkKit simplifies AgentKit.

## 3. System Touchpoints Inside AgentKit
| Component | Dependency on `send_message` |
|-----------|------------------------------|
| `BaseAgent.register_tools` (`agentkit/agents/base_agent.py:220-282`) | Registers `send_message` and ensures MCP registry initialization. |
| `TaskAwareAgent` (`agentkit/agents/task_aware_agent.py:55-78`) | Auto-injects an `HTTPMessageSender` so tools and fallbacks share transport. |
| Planner (`agentkit/planning/planner.py:300-308`) | Auto-schedules follow-up `send_message` actions after tool executions. |
| Reminders (`agentkit/functions/reminder_tools.py:27-70`) | Reminder triggers default to `send_message` actions. |

Maintaining the same contract keeps these flows intact.

## 4. NetworkKit Building Blocks
- **Transport** – `networkkit.network.HTTPMessageSender` handles async POSTs to the bus (`networkkit/network.py`).
- **Message model** – `networkkit.messages.Message` / `MessageType` remain the schema of record.
- **Databus** – `python -m networkkit.databus` runs the ZeroMQ hub that agents publish to (see NetworkKit README and module docs).

These should be reused within the MCP server for parity.

## 5. Target MCP Server Design (NetworkKit)
### 5.1 Package layout suggestion
```
networkkit/
  mcp/
    __init__.py
    send_message_server.py   # new MCP server entrypoint
    schemas.py               # optional shared models
```
Expose `python -m networkkit.mcp.send_message_server` so AgentKit can launch it via MCP stdio transport.

### 5.2 Tool contract
- **Name**: `send_message`
- **Description**: “Send a message to another agent or entity on the NetworkKit bus.”
- **Input schema (JSON)**:
  ```json
  {
    "type": "object",
    "properties": {
      "recipient": { "type": "string", "description": "Target agent or broadcast alias" },
      "content": { "type": "string", "description": "Message body" },
      "message_type": {
        "type": "string",
        "enum": ["HELO", "ACK", "CHAT", "SYSTEM", "SENSOR", "ERROR", "INFO"],
        "default": "CHAT"
      }
    },
    "required": ["recipient", "content"]
  }
  ```
- **Output schema (JSON)**:
  ```json
  {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "message_id": { "type": "string" },
      "recipient": { "type": "string" },
      "message_type": { "type": "string" },
      "metadata": { "type": "object" }
    }
  }
  ```

### 5.3 Server lifecycle sketch
```python
# networkkit/mcp/send_message_server.py
import asyncio
import os
import uuid
from mcp.server import Server
from networkkit.messages import Message, MessageType
from networkkit.network import HTTPMessageSender

server = Server("networkkit-send-message")

def _publish_address():
    return os.getenv("NETWORKKIT_BUS_PUBLISH_ADDRESS", "http://127.0.0.1:8000")

def _source_identity(context_agent_name: str | None = None):
    return os.getenv("NETWORKKIT_AGENT_NAME") or context_agent_name or "networkkit"

@server.tool()
async def send_message(recipient: str, content: str, message_type: str = "CHAT") -> dict:
    sender = server.state.setdefault(
        "message_sender",
        HTTPMessageSender(publish_address=_publish_address()),
    )

    message = Message(
        source=_source_identity(server.state.get("agent_name")),
        to=recipient,
        content=content,
        message_type=MessageType(message_type),
    )
    await sender.send_message(message)

    return {
        "status": "sent",
        "message_id": str(uuid.uuid4()),
        "recipient": recipient,
        "message_type": message_type,
    }

async def main():
    await server.serve_stdio()

if __name__ == "__main__":
    asyncio.run(main())
```

Populate `server.state["agent_name"]` during initialization using metadata provided by the MCP client (AgentKit can pass the agent’s configured `name`).

### 5.4 Metadata handling
- Accept `metadata` from the MCP launch config to override publish address or identity if needed.
- Return useful metadata (publish address, timestamps) alongside the response if AgentKit needs it for logging.

## 6. Porting Steps
1. **Implement MCP server module** inside NetworkKit using the sketch above; add cleanup to close `HTTPMessageSender`’s `aiohttp` session when the server shuts down.
2. **Update NetworkKit packaging** (`pyproject.toml`) to include the `mcp` dependency and a console entry point like `networkkit-mcp-send-message = networkkit.mcp.send_message_server:main`.
3. **AgentKit integration**:
   - Add an MCP server config to agent JSON (e.g. `agentkit/examples/config/sophia_task_agent.json`) pointing to the new entry point.
   - Behind a feature flag, prefer the MCP-provided tool over the built-in `send_message`.
4. **Migration strategy**:
   - Temporarily keep the built-in tool for compatibility; allow switching via config.
   - Once MCP version is stable, remove the legacy fallback.

## 7. Testing & Validation
- **NetworkKit unit tests** – mock the bus endpoint to validate payload formation, message type enum handling, and error propagation.
- **AgentKit integration tests** – extend existing send-message tests to run against the MCP-backed tool by injecting the MCP config.
- **Manual smoke** – run `python -m networkkit.databus`, start two configured AgentKit agents with the MCP server enabled, and verify round-trip messaging.

## 8. Open Questions
- Should the MCP server expose additional bus operations (broadcast, history, attachments)?
- Does AgentKit need stronger delivery guarantees (ACK handling) before returning `status="sent"`?
- How should multiple agents share a single MCP server process versus spawning per agent?

## 9. Next Actions Checklist
- [ ] Implement `networkkit.mcp.send_message_server` with proper cleanup.
- [ ] Add MCP dependency and entry point to NetworkKit packaging.
- [ ] Wire AgentKit configs/tests to launch the MCP server.
- [ ] Deprecate the AgentKit built-in `send_message` tool once MCP integration is default.
