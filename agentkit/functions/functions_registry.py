from typing import Any, Dict, Protocol, Callable, List, Optional
from pydantic import BaseModel, ValidationError, ConfigDict
import inspect
import asyncio
import json
from functools import partial
from agentkit.constants import DEFAULT_LLM_MODEL, FUNCTION_SYSTEM_TEMPLATE, FUNCTION_USER_TEMPLATE
from agentkit.processor import llm_processor
from agentkit.processor import JSONParseError, extract_json
from networkkit.messages import Message

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
    parameters: Optional[List[ParameterDescriptor]] = []

    def prompt(self) -> str:
        s = f"{self.name}: {self.description}\n"
        if self.parameters:
            s += "Parameters:\n"
            for param in self.parameters:
                s += f" - {param.prompt()}\n"
        else:
            s += "Parameters: None\n"
        return s            

class FunctionsRegistry(Protocol):
    def register_function(self, fn: Callable, descriptor: FunctionDescriptor) -> None:
        ...

    def set_function_map(self, map: Dict[str, Callable]) -> None:
        ...
        
    def prompt(self) -> str:
        ...

    def execute(self, function: str, parameters: Optional[dict]) -> Any:
        ...


class FunctionsRegistryError(ValueError):
    pass


class DefaultFunctionsRegistry:
    def __init__(self):
        self.function_registry: Dict[str, FunctionDescriptor] = {}
        self.function_map: Dict[str, Callable] = {}
    
    def register_function(self, fn: Callable, descriptor: FunctionDescriptor) -> None:
        if not isinstance(descriptor, FunctionDescriptor):
            raise TypeError("Descriptor must be an instance of FunctionDescriptor")
        self.function_registry[descriptor.name] = descriptor
        self.function_map[descriptor.name] = fn
        
    def set_function_map(self, map: Dict[str, Callable]) -> None:
        self.function_map = map

    def prompt(self) -> str:
        prompt = "List of available functions and their parameters:\n"
        for _, descriptor in self.function_registry.items():
            prompt += descriptor.prompt() + "\n"
        return prompt
    
    async def generate_function_request(self, state: str) -> str:
        system_prompt = FUNCTION_SYSTEM_TEMPLATE.format( functions=self.prompt())
        user_prompt = FUNCTION_USER_TEMPLATE.format(state=state)
        return await llm_processor(
            llm_model=DEFAULT_LLM_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
 
    async def execute(self, function: str, parameters: Optional[dict]={}) -> Any:
        if function not in self.function_map:
            raise FunctionsRegistryError(f"Function {function} is not registered")

        func = self.function_map[function]
        func_descriptor = self.function_registry.get(function)

        # Optional: Validate that required parameters are provided
        if func_descriptor:
            for param in func_descriptor.parameters:
                if param.required and param.name not in parameters:
                    raise FunctionsRegistryError(f"Missing required parameter: {param.name}. for function {func_descriptor.name}. make sure to add this in your JSON.")

        # Check if the function is a coroutine
        # If func is a partial, unwrap it to get the actual function
        actual_func = func
        if isinstance(func, partial):
            actual_func = func.func

        # Check if the actual function is a coroutine
        if inspect.iscoroutinefunction(actual_func):
            return await func(**parameters)
        else:
            return func(**parameters)

    def read_function_definitions(self, file_path: str) -> Dict:
        with open(file_path, 'r') as file:
            return json.load(file)

    def register_functions_from_json(self, json_file: str) -> None:
        function_definitions = self.read_function_definitions(json_file)
        for func_def in function_definitions:
            func = self.function_map.get(func_def['function_name'])
            if func:
                descriptor = FunctionDescriptor(
                    name=func_def['function_name'],
                    description=func_def.get('description', ''),
                    parameters=func_def.get('parameters', [])
                )
                self.register_function(func, descriptor)
    
    async def execute_function_from_message(self, message:Message) -> Message:
        if message.message_type == "ERROR":
            return message
        function_request = await self.generate_function_request(state=message.content)
        try:
            m = await self.process_function_request(function_request)
        except (JSONParseError, FunctionsRegistryError, TypeError, ValidationError) as e:
            m = Message(source=message.to, to=message.source, content=str(e), message_type="ERROR")
        return m

    async def process_function_request(self, function_request: str) -> Any:
        action = extract_json(function_request)
        return await self.execute(**action)
