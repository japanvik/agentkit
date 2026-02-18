"""
Tests for built-in tools.

This module contains tests for the built-in tools provided by the AgentKit framework.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from networkkit.messages import Message, MessageType

from agentkit.agents.base_agent import BaseAgent
from agentkit.functions.built_in_tools import send_message_tool
from agentkit.functions.functions_registry import (
    DefaultFunctionsRegistry,
    FunctionDescriptor,
    ParameterDescriptor,
    ToolExecutionContext,
)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = "test_agent"
    agent._internal_send_message = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_send_message_tool(mock_agent):
    """Test that the send_message tool sends a message correctly."""
    # Call the send_message tool
    result = await send_message_tool(
        ToolExecutionContext(agent=mock_agent),
        recipient="recipient",
        content="Hello, world!",
        message_type="CHAT"
    )
    
    # Check that the agent's _internal_send_message method was called
    mock_agent._internal_send_message.assert_called_once()
    
    # Get the message that was sent
    message = mock_agent._internal_send_message.call_args[0][0]
    
    # Check that the message has the correct properties
    assert message.source == "test_agent"
    assert message.to == "recipient"
    assert message.content == "Hello, world!"
    assert message.message_type == MessageType.CHAT
    
    # Check that the result has the expected structure
    assert "status" in result
    assert result["status"] == "sent"
    assert "message_id" in result
    assert "recipient" in result
    assert result["recipient"] == "recipient"
    assert "message_type" in result
    assert result["message_type"] == "CHAT"


@pytest.mark.asyncio
async def test_send_message_tool_prefers_mcp_when_enabled(mock_agent):
    """Test that send_message uses configured MCP function when enabled."""
    mock_agent.config = {
        "send_message_backend": "mcp",
        "mcp_send_message_function": "networkkit::send_message",
    }
    mock_agent.functions_registry = MagicMock()
    mock_agent.functions_registry.has_function.return_value = True
    mock_agent.functions_registry.execute = AsyncMock(
        return_value={
            "raw": {
                "status": "sent",
                "message_id": "mcp-123",
                "recipient": "recipient",
                "message_type": "CHAT",
            }
        }
    )

    result = await send_message_tool(
        ToolExecutionContext(agent=mock_agent),
        recipient="recipient",
        content="Hello over MCP",
        message_type="CHAT",
    )

    mock_agent.functions_registry.execute.assert_awaited_once()
    mock_agent._internal_send_message.assert_not_awaited()
    assert result["status"] == "sent"
    assert result["message_id"] == "mcp-123"
    assert result["recipient"] == "recipient"
    assert result["message_type"] == "CHAT"


@pytest.mark.asyncio
async def test_send_message_tool_falls_back_when_mcp_call_fails(mock_agent):
    """Test that send_message falls back to direct send when MCP invocation fails."""
    mock_agent.config = {
        "send_message_backend": "mcp",
        "mcp_send_message_function": "networkkit::send_message",
    }
    mock_agent.functions_registry = MagicMock()
    mock_agent.functions_registry.has_function.return_value = True
    mock_agent.functions_registry.execute = AsyncMock(side_effect=RuntimeError("boom"))

    result = await send_message_tool(
        ToolExecutionContext(agent=mock_agent),
        recipient="recipient",
        content="Fallback please",
        message_type="CHAT",
    )

    mock_agent.functions_registry.execute.assert_awaited_once()
    mock_agent._internal_send_message.assert_awaited_once()
    assert result["status"] == "sent"
    assert result["recipient"] == "recipient"
    assert result["message_type"] == "CHAT"


@pytest.mark.asyncio
async def test_register_tools():
    """Test that the register_tools method registers the send_message tool."""
    # Create a functions registry
    registry = DefaultFunctionsRegistry()
    
    # Call the actual register_tools method from BaseAgent
    with patch('agentkit.functions.built_in_tools.send_message_tool') as mock_send_message_tool:
        # Create a real BaseAgent instance
        agent = BaseAgent(name="test_agent", config={})
        
        # Register tools
        agent.register_tools(registry)
    
    # Check that the send_message tool was registered
    assert "send_message" in registry.function_map
    assert "shell_command" in registry.function_map
    assert "schedule_reminder" in registry.function_map
    assert "list_directory" in registry.function_map
    assert "read_file" in registry.function_map
    assert "write_file" in registry.function_map
    
    # Check that the function descriptor has the expected properties
    descriptor = registry.function_registry["send_message"]
    assert descriptor.name == "send_message"
    assert "Send a message" in descriptor.description
    
    # Check that the parameters are correct
    assert len(descriptor.parameters) == 3
    
    # Check recipient parameter
    recipient_param = next((p for p in descriptor.parameters if p.name == "recipient"), None)
    assert recipient_param is not None
    assert recipient_param.required is True
    
    # Check content parameter
    content_param = next((p for p in descriptor.parameters if p.name == "content"), None)
    assert content_param is not None
    assert content_param.required is True
    
    # Check message_type parameter
    message_type_param = next((p for p in descriptor.parameters if p.name == "message_type"), None)
    assert message_type_param is not None
    assert message_type_param.required is False


@pytest.mark.asyncio
async def test_task_aware_agent_uses_send_message_tool():
    """Test that the TaskAwareAgent uses the send_message tool."""
    from agentkit.agents.task_aware_agent import TaskAwareAgent
    
    # Create a TaskAwareAgent
    agent = TaskAwareAgent(name="test_agent", config={})
    
    # Check that the functions registry has the send_message tool
    assert hasattr(agent, "functions_registry")
    assert "send_message" in agent.functions_registry.function_map
    
    # Mock the execute method of the functions registry
    agent.functions_registry.execute = AsyncMock()
    
    # Create a mock action
    action = {
        "action_type": "send_message",
        "tool_name": "send_message",
        "parameters": {
            "recipient": "recipient",
            "content": "Hello, world!",
            "message_type": "CHAT"
        }
    }
    
    # Call the _execute_action method
    source_message = Message(
        source="recipient",
        to="test_agent",
        content="Hello",
        message_type=MessageType.CHAT,
    )
    await agent._execute_action(
        action,
        conversation_id="conv1",
        source_message=source_message,
    )
    
    # Check that the functions registry's execute method was called with the correct arguments
    from unittest.mock import ANY

    agent.functions_registry.execute.assert_called_once_with(
        function="send_message",
        parameters={
            "recipient": "recipient",
            "content": "Hello, world!",
            "message_type": "CHAT"
        },
        context=ANY,
    )
