"""Tool-based brain implementation."""
from typing import Dict, Any, Optional
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import logging
import json

from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.memory_protocol import Memory
from agentkit.processor import llm_chat, extract_json
from agentkit.functions.functions_registry import DefaultFunctionsRegistry, ToolExecutionContext
from networkkit.messages import Message
from agentkit.constants import FUNCTION_SYSTEM_TEMPLATE

class ToolBrain(SimpleBrain):
    """
    Tool-based brain for agent interaction, utilizing LLM for message generation and function calling.

    This class extends SimpleBrain to add support for function calling through a functions registry.
    Instead of directly calling send_message, it uses the functions registry to call the send_message_tool.
    
    The brain operates based on pre-defined system and user prompts that can be formatted with placeholders like agent name,
    description, context, and target recipient. These prompts are used to guide the LLM in response generation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        memory_manager: Memory,
        system_prompt: str = "",
        user_prompt: str = "",
        api_config: Dict[str, Any] = None,
        functions_registry: Optional[DefaultFunctionsRegistry] = None
    ) -> None:
        """
        Initialize the tool brain with name, description, model, memory manager, and functions registry.
        
        Args:
            name: The name of the agent this brain belongs to
            description: A description of the agent's purpose or capabilities
            model: The name of the LLM model to be used (e.g., "gpt-4")
            memory_manager: An instance of a memory component for storing conversation history
            system_prompt: The system prompt template for the LLM
            user_prompt: The user prompt template for the LLM
            api_config: Configuration dictionary for the LLM API (e.g., temperature, max_tokens)
            functions_registry: The functions registry to use for function calling
        """
        super().__init__(
            name=name,
            description=description,
            model=model,
            memory_manager=memory_manager,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_config=api_config
        )
        self.functions_registry = functions_registry or DefaultFunctionsRegistry()
    
    def set_config(self, config) -> None:
        """
        Set the component configuration and register tools.
        
        This method is called by the agent to provide the brain with access to
        agent information and capabilities through the ComponentConfig object.
        It also registers the send_message tool with the functions registry.
        
        Args:
            config: The component configuration containing agent information and capabilities
        """
        super().set_config(config)
        
        # Register the send_message tool with the functions registry
        if hasattr(config.message_sender, 'register_tools'):
            config.message_sender.register_tools(self.functions_registry)
        
    async def handle_chat_message(self, message: Message) -> None:
        """
        Handle incoming chat messages directed to the agent.

        This method is called whenever a chat message is received by the agent.
        It performs the following actions:

        1. Stores the received message in the memory manager.
        2. Checks if the message source is different from the agent itself (i.e., an incoming message).
        3. If it's an incoming message, updates the agent's attention to the message source.
        4. Generates a response using the LLM.
        5. Processes the response as a function call using the functions registry.

        Args:
            message (Message): The received message object.
        """
        if not self.component_config:
            logging.error("No config set - brain operations require configuration")
            return

        self.memory_manager.remember(message)
        
        if message.source != self.name:
            # Reply to a chat message from someone
            self.component_config.message_sender.attention = message.source
            max_attempts = 3
            extra_prompt = ""
            last_response = None

            for attempt in range(max_attempts):
                response = await self.generate_chat_response(extra_prompt)
                last_response = response

                try:
                    function_data = extract_json(response.content)
                except Exception as exc:
                    logging.warning(
                        "Failed to parse function call attempt %s: %s",
                        attempt + 1,
                        exc,
                    )
                    if attempt == max_attempts - 1:
                        if last_response:
                            tools = ", ".join(sorted(self.functions_registry.function_map.keys()))
                            content = last_response.content.strip()
                            if content:
                                content += "\n\n"
                            content += (
                                "(For reference, available tools: "
                                f"{tools}.)"
                            )
                            last_response.content = content
                            await self.component_config.message_sender.send_message(last_response)
                        return

                    extra_prompt = (
                        "Reminder: respond with exactly one JSON object describing a function call "
                        "using the format Function: {\"function\": \"name\", \"parameters\": {...}}."
                    )
                    continue

                if "function" in function_data:
                    function_name = function_data["function"]
                    parameters = function_data.get("parameters", {})

                    tool_context = ToolExecutionContext(
                        agent=self.component_config.message_sender
                    )

                    if function_name == "send_message":
                        parameters["recipient"] = message.source
                        await self.functions_registry.execute(
                            "send_message",
                            parameters,
                            context=tool_context,
                        )
                    elif self.functions_registry.has_function(function_name):
                        await self.functions_registry.execute(
                            function_name,
                            parameters,
                            context=tool_context,
                        )
                    else:
                        logging.warning(
                            "Function '%s' not found in registry. Sending explanation.",
                            function_name,
                        )
                        tools = ", ".join(sorted(self.functions_registry.function_map.keys()))
                        content = (
                            f"I tried to use a function called '{function_name}', but it's not available.\n"
                            f"Available tools: {tools}."
                        )
                        new_response = Message(
                            source=self.name,
                            to=message.source,
                            content=content,
                            message_type=MessageType.CHAT
                        )
                        await self.component_config.message_sender.send_message(new_response)
                    return

                await self.component_config.message_sender.send_message(response)
                return
            # Should not reach here, but ensure last response is delivered
            if last_response:
                await self.component_config.message_sender.send_message(last_response)

    async def generate_chat_response(self, extra_system_prompt: str = "") -> Message:
        """
        Generate a chat response based on the current context.
        
        Returns:
            Message: The generated response message.
        """
        # Format the system prompt to include function calling instructions
        context = self.get_context()
        target = self.component_config.message_sender.attention
        
        # Combine the original system prompt with function calling instructions
        functions_prompt = self.functions_registry.prompt()
        capability_summary = f"Current tools: {', '.join(sorted(self.functions_registry.function_map.keys()))}."
        function_instructions = FUNCTION_SYSTEM_TEMPLATE.format(functions=functions_prompt)

        sections = []
        if self.system_prompt:
            sections.append(
                self.system_prompt.format(
                    name=self.name,
                    description=self.description,
                    context=context,
                    target=target,
                )
            )
        sections.append(capability_summary)
        sections.append(f"Current datetime (UTC): {datetime.utcnow().isoformat()}")
        if extra_system_prompt:
            sections.append(extra_system_prompt)
        sections.append(function_instructions)

        combined_system_prompt = "\n\n".join(sections)
        
        messages = self.create_chat_messages_prompt(combined_system_prompt)
        
        # Extract API configuration
        api_base = self.api_config.get('api_base')
        api_key = self.api_config.get('api_key')
        
        reply = await llm_chat(
            llm_model=self.model,
            messages=messages,
            api_base=api_base,
            api_key=api_key
        )

        # Create a response message with the raw LLM output
        return self.format_response(reply)
