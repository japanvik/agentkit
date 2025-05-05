"""Tool-based brain implementation."""
from typing import Dict, Any, Optional

import logging
import json

from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.memory_protocol import Memory
from agentkit.processor import llm_chat, extract_json
from agentkit.functions.functions_registry import DefaultFunctionsRegistry
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
            response = await self.generate_chat_response()
            
            # Instead of directly sending the message, process it as a function call
            try:
                # Extract the function call from the response content
                function_data = extract_json(response.content)
                
                # Execute the function call
                if "function" in function_data:
                    function_name = function_data["function"]
                    parameters = function_data.get("parameters", {})
                    
                    if function_name == "send_message":
                        # Always set the recipient to the message source to ensure correct routing
                        parameters["recipient"] = message.source
                        await self.functions_registry.execute("send_message", parameters)
                    elif self.functions_registry.has_function(function_name):
                        # Execute other registered functions
                        await self.functions_registry.execute(function_name, parameters)
                    else:
                        # Function doesn't exist, convert to a regular message
                        logging.warning(f"Function '{function_name}' not found in registry. Converting to regular message.")
                        # Create a new message with the original content (without the function call)
                        content = f"I tried to use a function called '{function_name}' but it's not available. Here's my response instead:\n\n"
                        if "content" in parameters:
                            content += parameters["content"]
                        else:
                            content += "I'd like to help you with that, but I don't have the capability you're asking for."
                        
                        new_response = Message(
                            source=self.name,
                            to=message.source,
                            content=content,
                            message_type=MessageType.CHAT
                        )
                        await self.component_config.message_sender.send_message(new_response)
                else:
                    # No function call found, send the original response
                    await self.component_config.message_sender.send_message(response)
            except Exception as e:
                logging.error(f"Error processing function call: {e}")
                # Fallback to direct send_message
                await self.component_config.message_sender.send_message(response)

    async def generate_chat_response(self) -> Message:
        """
        Generate a chat response based on the current context.
        
        Returns:
            Message: The generated response message.
        """
        # Format the system prompt to include function calling instructions
        context = self.get_context()
        target = self.component_config.message_sender.attention
        
        # Combine the original system prompt with function calling instructions
        function_instructions = FUNCTION_SYSTEM_TEMPLATE.format(functions=self.functions_registry.prompt())
        combined_system_prompt = f"{self.system_prompt.format(name=self.name, description=self.description, context=context, target=target)}\n\n{function_instructions}"
        
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
