from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentkit.constants import (
    DEFAULT_LLM_MODEL,
    FUNCTION_SYSTEM_TEMPLATE,
    FUNCTION_USER_TEMPLATE,
)
from agentkit.processor import JSONParseError, extract_json, llm_chat
from networkkit.messages import Message, MessageType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentkit.agents.base_agent import BaseAgent


class ParameterDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    required: bool

    def prompt(self) -> str:
        return f"{self.name} ({'required' if self.required else 'optional'}): {self.description}"


class FunctionDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: List[ParameterDescriptor] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    requires_confirmation: bool = False

    def prompt(self) -> str:
        s = f"{self.name}: {self.description}\n"
        if self.parameters:
            s += "Parameters:\n"
            for param in self.parameters:
                s += f" - {param.prompt()}\n"
        else:
            s += "Parameters: None\n"
        return s


@dataclass
class ToolExecutionContext:
    """Context information supplied to tool implementations."""

    agent: "BaseAgent"
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredTool:
    """Runtime information for a registered tool."""

    handler: Callable[..., Any]
    descriptor: FunctionDescriptor
    pass_context: bool = False


class FunctionsRegistry(Protocol):
    def register_function(
        self, fn: Callable, descriptor: FunctionDescriptor, *, pass_context: bool = False
    ) -> None:
        ...

    def set_function_map(self, map: Dict[str, Callable]) -> None:
        ...

    def prompt(self) -> str:
        ...

    async def execute(
        self,
        function: str,
        parameters: Optional[dict],
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        ...

    def has_function(self, function: str) -> bool:
        ...


class FunctionsRegistryError(ValueError):
    pass


class DefaultFunctionsRegistry:
    def __init__(self):
        self.function_registry: Dict[str, FunctionDescriptor] = {}
        self.function_map: Dict[str, Callable] = {}
        self._registered_tools: Dict[str, RegisteredTool] = {}

    def register_function(
        self, fn: Callable, descriptor: FunctionDescriptor, *, pass_context: bool = False
    ) -> None:
        if not isinstance(descriptor, FunctionDescriptor):
            raise TypeError("Descriptor must be an instance of FunctionDescriptor")
        if descriptor.name in self.function_registry:
            raise FunctionsRegistryError(f"Function {descriptor.name} is already registered")

        self.function_registry[descriptor.name] = descriptor
        self.function_map[descriptor.name] = fn
        self._registered_tools[descriptor.name] = RegisteredTool(
            handler=fn, descriptor=descriptor, pass_context=pass_context
        )

    def set_function_map(self, map: Dict[str, Callable]) -> None:
        self.function_map = map
        # When the function map is replaced, keep existing descriptors but update handlers
        for name, handler in map.items():
            existing = self._registered_tools.get(name)
            if existing:
                self._registered_tools[name] = RegisteredTool(
                    handler=handler,
                    descriptor=existing.descriptor,
                    pass_context=existing.pass_context,
                )

    def prompt(self) -> str:
        prompt = "List of available functions and their parameters:\n"
        for _, descriptor in self.function_registry.items():
            prompt += descriptor.prompt() + "\n"
        return prompt

    async def generate_function_request(self, state: str) -> str:
        system_prompt = FUNCTION_SYSTEM_TEMPLATE.format(functions=self.prompt())
        user_prompt = FUNCTION_USER_TEMPLATE.format(state=state)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await llm_chat(
            llm_model=DEFAULT_LLM_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )

    async def execute(
        self,
        function: str,
        parameters: Optional[dict] = None,
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        if function not in self.function_map:
            raise FunctionsRegistryError(f"Function {function} is not registered")

        tool = self._registered_tools.get(function)
        if not tool:
            raise FunctionsRegistryError(f"No runtime metadata registered for {function}")

        params = parameters or {}

        # Optional: Validate that required parameters are provided
        descriptor = tool.descriptor
        for param in descriptor.parameters:
            if param.required and param.name not in params:
                raise FunctionsRegistryError(
                    f"Missing required parameter: {param.name}. "
                    f"for function {descriptor.name}. make sure to add this in your JSON."
                )

        handler = tool.handler
        actual_func = handler.func if isinstance(handler, partial) else handler
        call_args = params.copy()

        if tool.pass_context:
            if context is None:
                raise FunctionsRegistryError(
                    f"Tool '{descriptor.name}' requires an execution context"
                )
            if inspect.iscoroutinefunction(actual_func):
                return await handler(context, **call_args)
            return handler(context, **call_args)

        if inspect.iscoroutinefunction(actual_func):
            return await handler(**call_args)
        return handler(**call_args)

    def read_function_definitions(self, file_path: str) -> Dict:
        with open(file_path, "r") as file:
            return json.load(file)

    def register_functions_from_json(self, json_file: str) -> None:
        function_definitions = self.read_function_definitions(json_file)
        for func_def in function_definitions:
            func = self.function_map.get(func_def["function_name"])
            if func:
                descriptor = FunctionDescriptor(
                    name=func_def["function_name"],
                    description=func_def.get("description", ""),
                    parameters=func_def.get("parameters", []),
                )
                existing = self._registered_tools.get(descriptor.name)
                if existing:
                    self.function_registry[descriptor.name] = descriptor
                    self._registered_tools[descriptor.name] = RegisteredTool(
                        handler=existing.handler,
                        descriptor=descriptor,
                        pass_context=existing.pass_context,
                    )
                else:
                    self.register_function(func, descriptor)

    async def execute_function_from_message(self, message: Message) -> Message:
        if message.message_type in (MessageType.ERROR, "ERROR"):
            return message
        function_request = await self.generate_function_request(state=message.content)
        try:
            result_message = await self.process_function_request(function_request)
        except (JSONParseError, FunctionsRegistryError, TypeError, ValidationError) as exc:
            logger.exception("Failed to execute function from message")
            result_message = Message(
                source=message.to,
                to=message.source,
                content=str(exc),
                message_type=MessageType.ERROR,
            )
        return result_message

    async def process_function_request(self, function_request: str) -> Any:
        action = extract_json(function_request)
        return await self.execute(**action)

    def has_function(self, function: str) -> bool:
        """
        Check if a function is registered in the registry.

        Args:
            function: The name of the function to check

        Returns:
            bool: True if the function is registered, False otherwise
        """
        return function in self.function_map
