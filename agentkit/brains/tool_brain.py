"""Tool-based brain implementation."""
from typing import Dict, Any, Optional, List
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
from networkkit.messages import Message, MessageType
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
        self._active_conversation_id: Optional[str] = None
        self._active_source_message: Optional[Message] = None
        self._active_task_id: Optional[str] = None
        self._active_delegation_path: List[str] = []
        self.max_tool_iterations = 8
    
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

        if message.source == self.name:
            return

        original_sender = message.source
        self.component_config.message_sender.attention = original_sender
        delegation_path = list(self._active_delegation_path)
        if original_sender not in delegation_path:
            delegation_path.append(original_sender)
        if not delegation_path or delegation_path[-1] != self.name:
            delegation_path.append(self.name)
        self._active_delegation_path = delegation_path
        tool_context = ToolExecutionContext(
            agent=self.component_config.message_sender,
            session_id=self._active_conversation_id,
            metadata={
                "conversation_id": self._active_conversation_id,
                "requester": original_sender,
                "source": original_sender,
                "target": original_sender,
                "last_message_content": message.content,
                "agent_name": self.name,
                "delegation_path": delegation_path,
                "task_id": self._active_task_id,
            },
        )

        max_iterations = getattr(self, "max_tool_iterations", 6)
        last_tool_name: Optional[str] = None
        last_tool_parameters: Optional[Dict[str, Any]] = None
        last_tool_result: Any = None
        followup_prompt: str = ""

        verification_attempted = False
        iterations_used = 0

        for iteration in range(max_iterations):
            iterations_used = iteration + 1
            response = await self.generate_chat_response(extra_system_prompt=followup_prompt)
            followup_prompt = ""
            self.memory_manager.remember(response)

            raw_content = (response.content or "").strip()
            if not raw_content:
                logging.warning("LLM returned empty content on iteration %s", iteration)
                if iteration < max_iterations - 1:
                    followup_prompt = self._build_retry_prompt(
                        reason="the previous reply was empty",
                        last_tool_name=last_tool_name,
                        last_tool_parameters=last_tool_parameters,
                        last_tool_result=last_tool_result,
                        user_request=message.content,
                    )
                    continue
                break

            try:
                function_data = extract_json(response.content)
            except Exception as exc:
                logging.warning("Failed to parse function call: %s", exc)
                if iteration < max_iterations - 1:
                    followup_prompt = self._build_retry_prompt(
                        reason="I could not parse your previous reply as JSON",
                        last_tool_name=last_tool_name,
                        last_tool_parameters=last_tool_parameters,
                        last_tool_result=last_tool_result,
                        user_request=message.content,
                    )
                    continue
                break

            function_name = function_data.get("function")
            if not function_name:
                logging.warning("No function specified in LLM response")
                if iteration < max_iterations - 1:
                    followup_prompt = self._build_retry_prompt(
                        reason="the previous reply did not specify a function name",
                        last_tool_name=last_tool_name,
                        last_tool_parameters=last_tool_parameters,
                        last_tool_result=last_tool_result,
                        user_request=message.content,
                    )
                    continue
                break

            parameters = function_data.get("parameters", {}) or {}

            if function_name == "schedule_reminder" and "recipient" not in parameters:
                parameters["recipient"] = original_sender

            if function_name == "send_message":
                parameters["recipient"] = original_sender
                await self.functions_registry.execute(
                    "send_message",
                    parameters,
                    context=tool_context,
                )
                return

            if not self.functions_registry.has_function(function_name):
                logging.warning("Function '%s' not found in registry", function_name)
                if iteration < max_iterations - 1:
                    followup_prompt = self._build_retry_prompt(
                        reason=f"'{function_name}' is not an available tool",
                        last_tool_name=last_tool_name,
                        last_tool_parameters=last_tool_parameters,
                        last_tool_result=last_tool_result,
                        user_request=message.content,
                    )
                    continue
                break

            last_tool_name = function_name
            last_tool_parameters = parameters
            last_tool_result = await self.functions_registry.execute(
                function_name,
                parameters,
                context=tool_context,
            )
            self._record_tool_observation(function_name, parameters, last_tool_result)
            followup_prompt = self._build_followup_prompt(
                function_name,
                parameters,
                last_tool_result,
                user_request=message.content,
            )

        # If we exit the loop without sending a message, provide a graceful fallback.
        fallback_content = "I ran out of planning steps before I could send a reply."
        if last_tool_name is not None:
            summary = self._summarize_tool_result(last_tool_name, last_tool_result)
            issue_hint = self._tool_issue_hint(last_tool_result)
            if not issue_hint:
                if await self._attempt_final_response(
                    summary=summary,
                    recipient=original_sender,
                    user_request=message.content,
                    tool_name=last_tool_name,
                    tool_parameters=last_tool_parameters or {},
                ):
                    return
                if not verification_attempted:
                    decision = await self._verify_task_completion(
                        user_request=message.content,
                        summary=summary,
                        last_tool_name=last_tool_name,
                        last_tool_parameters=last_tool_parameters or {},
                    )
                    verification_attempted = True
                    if decision:
                        action = decision.get("action")
                        if action == "send_message":
                            content = decision.get("content", "").strip() or summary
                            await self._send_message_to_recipient(
                                content,
                                original_sender,
                                tool_context=tool_context,
                            )
                            return
                        if (
                            action == "call_tool"
                            and iterations_used < max_iterations + 1
                        ):
                            next_function = decision.get("function")
                            parameters = decision.get("parameters") or {}
                            if next_function and self.functions_registry.has_function(next_function):
                                try:
                                    next_result = await self.functions_registry.execute(
                                        next_function,
                                        parameters,
                                        context=tool_context,
                                    )
                                except Exception:
                                    logging.exception(
                                        "Verification-triggered tool '%s' failed",
                                        next_function,
                                    )
                                else:
                                    self._record_tool_observation(
                                        next_function,
                                        parameters,
                                        next_result,
                                    )
                                    summary = self._summarize_tool_result(next_function, next_result)
                                    if await self._attempt_final_response(
                                        summary=summary,
                                        recipient=original_sender,
                                        user_request=message.content,
                                        tool_name=next_function,
                                        tool_parameters=parameters,
                                    ):
                                        return
                                    last_tool_name = next_function
                                    last_tool_parameters = parameters
                                    last_tool_result = next_result
                                    issue_hint = self._tool_issue_hint(next_result)
                                    iterations_used += 1
                                    if not issue_hint:
                                        synthesized = await self._synthesize_fallback_response(
                                            user_request=message.content,
                                            summary=summary,
                                            tool_name=next_function,
                                            tool_result=next_result,
                                        )
                                        if synthesized:
                                            await self._send_message_to_recipient(
                                                synthesized,
                                                original_sender,
                                                tool_context=tool_context,
                                            )
                                            return
                                        fallback_content = (
                                            "Here's what I found from the most recent tool run:\n"
                                            f"{summary}\n\n"
                                            "Let me know if you'd like me to dig deeper, retry with a different approach, or format the result differently."
                                        )
                                        # Proceed to send fallback below using updated summary
                                    else:
                                        summary = self._summarize_tool_result(next_function, next_result)
                                    # fall through to fallback message if needed
                            else:
                                logging.warning("Verification suggested unavailable tool '%s'", next_function)
            synthesized = await self._synthesize_fallback_response(
                user_request=message.content,
                summary=summary,
                tool_name=last_tool_name,
                tool_result=last_tool_result,
            )
            if synthesized:
                await self._send_message_to_recipient(
                    synthesized,
                    original_sender,
                    tool_context=tool_context,
                )
                return
            if summary:
                fallback_content = (
                    "Here's what I found from the most recent tool run:\n"
                    f"{summary}\n\n"
                    "Let me know if you'd like me to dig deeper, retry with a different approach, or format the result differently."
                )
            else:
                synthesized = await self._synthesize_failure_response(
                    user_request=message.content,
                    reason="No tools were executed successfully before planning halted.",
                )
                if synthesized:
                    await self._send_message_to_recipient(
                        synthesized,
                        original_sender,
                        tool_context=tool_context,
                    )
                    return
                fallback_content = (
                    "I wasn't able to make progress on that request. "
                    "If you can share a specific source or more details, I can try again."
                )
        await self._send_message_to_recipient(
            fallback_content,
            original_sender,
            tool_context=tool_context,
        )

    async def _send_fallback_response(
        self,
        response: Message,
        recipient: str,
        *,
        tool_context: ToolExecutionContext,
    ) -> None:
        if not self.component_config:
            logging.error("No config set - cannot send fallback response")
            return

        if response and response.content:
            logging.debug("LLM raw response: %s", response.content)
            content = response.content.strip()
            if content:
                content += (
                    "\n\n(Note: I couldn't interpret this as a tool action, "
                    "so I'm responding directly.)"
                )
            else:
                content = (
                    "I'm having trouble interpreting the response right now. "
                    "Please try again."
                )
        else:
            content = (
                "I'm having trouble interpreting the response right now. "
                "Please try again."
            )

        await self._send_message_to_recipient(
            content,
            recipient,
            tool_context=tool_context,
        )

    async def _send_message_to_recipient(
        self,
        content: str,
        recipient: str,
        *,
        tool_context: ToolExecutionContext,
    ) -> None:
        if not self.component_config:
            logging.error("Cannot send message without component config")
            return

        try:
            if self.functions_registry.has_function("send_message"):
                await self.functions_registry.execute(
                    "send_message",
                    {
                        "recipient": recipient,
                        "content": content,
                        "message_type": "CHAT",
                    },
                    context=tool_context,
                )
                return
        except Exception:
            logging.exception("Failed to send message via send_message tool; falling back to direct send")

        reply = Message(
            source=self.name,
            to=recipient,
            content=content,
            message_type=MessageType.CHAT,
        )
        await self.component_config.message_sender.send_message(reply)
        try:
            self.memory_manager.remember(reply)
        except Exception:
            logging.debug("Failed to store outbound message in memory", exc_info=True)

    def _record_tool_observation(
        self,
        function_name: str,
        parameters: Dict[str, Any],
        result: Any,
    ) -> None:
        try:
            payload = {
                "function": function_name,
                "parameters": parameters,
                "result": result,
            }
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        except TypeError:
            serialized = f"{function_name}(parameters={parameters}) -> {result}"

        truncated = self._truncate(serialized, limit=2000)
        observation_message = Message(
            source=self.name,
            to=self.name,
            content=f"[tool {function_name}] observation:\n{truncated}",
            message_type=MessageType.SYSTEM,
        )
        try:
            self.memory_manager.remember(observation_message)
        except Exception:
            logging.debug("Failed to record tool observation", exc_info=True)

    def _summarize_tool_result(self, function_name: str, result: Any) -> str:
        if function_name == "schedule_reminder" and isinstance(result, dict):
            return self._summarize_schedule_reminder(result)

        if isinstance(result, dict):
            parts = []
            exit_code = result.get("exit_code")
            if exit_code is not None:
                parts.append(f"exit code {exit_code}")
            if result.get("timed_out"):
                parts.append("command timed out")
            status = result.get("status")
            if status and status not in {"completed", "ok"}:
                parts.append(f"status: {status}")

            header = ""
            if parts:
                header = "Result: " + ", ".join(parts) + "."

            stdout = (result.get("stdout") or "").strip()
            stderr = (result.get("stderr") or "").strip()

            snippets = []
            friendly_lines = []
            if stdout:
                stdout_lines = stdout.splitlines()
                friendly_lines.extend(self._interpret_stdout(stdout_lines))
                snippet_lines = stdout_lines[:5]
                snippet = "\n".join(snippet_lines)
                if len(stdout_lines) > 5:
                    snippet += "\n…"
                snippets.append(f"stdout:\n{snippet}")

            if stderr:
                stderr_lines = stderr.splitlines()
                snippet_lines = stderr_lines[:3]
                snippet = "\n".join(stderr_lines[:3])
                if len(stderr_lines) > 3:
                    snippet += "\n…"
                snippets.append(f"stderr:\n{snippet}")

            body = "\n".join(snippets) if snippets else ""
            friendly_text = "\n".join(friendly_lines).strip()
            summary = "\n".join(filter(None, [header, friendly_text, body])).strip()
            if summary:
                return self._truncate(summary, limit=1000)

        try:
            serialized = json.dumps(result, ensure_ascii=False, indent=2)
        except TypeError:
            serialized = str(result)
        return self._truncate(serialized, limit=1000)

    def _build_followup_prompt(
        self,
        function_name: str,
        parameters: Dict[str, Any],
        result: Any,
        *,
        user_request: str,
    ) -> str:
        summary = self._summarize_tool_result(function_name, result)
        request_lower = user_request.lower() if user_request else ""
        user_wants_json = "json" in request_lower or "raw" in request_lower
        issue_hint = self._tool_issue_hint(result)
        response_guidance = (
            "If the user specifically asked for JSON or raw output, include it verbatim. "
            "Otherwise, craft a concise, human-friendly explanation that highlights the most relevant facts, "
            "including filenames, timestamps, numbers, or status values as appropriate. "
            "Avoid dumping long command output unless it was requested."
        )
        if user_wants_json:
            response_guidance = (
                "The user explicitly asked for JSON or raw output. Provide the data they requested, "
                "and optionally add one short sentence explaining what it contains."
            )
        evaluation_prompt = (
            "Carefully evaluate the tool output against the user's latest request. "
            "If—and only if—the output fully answers the request, craft a final send_message response that clearly states the result. "
            "If the output is incomplete, inaccurate, ambiguous, or unrelated, you must launch another tool call (with corrected arguments if necessary) before replying. "
            "When stderr contains an error or unsupported option, you must adjust the command and try again—do not send the answer yet. "
            "Always confirm the answer yourself; do not pass raw tool output to the user unless you are certain it satisfies the request."
        )
        return (
            "Tool execution completed. "
            f"Tool name: {function_name} with parameters {json.dumps(parameters, ensure_ascii=False)}.\n"
            f"Tool result summary:\n{summary}\n\n"
            f"{issue_hint}{evaluation_prompt} {response_guidance} "
            "If the task is complete, call the send_message tool with your final answer. "
            "If another tool call is needed, issue it now."
        )

    @staticmethod
    def _truncate(value: str, *, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + "\n... (truncated)"
    @staticmethod
    def _localize_timestamp(iso_string: str) -> Optional[tuple[str, str, str]]:
        if not iso_string:
            return None
        try:
            dt = datetime.fromisoformat(iso_string)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        utc_str = dt.isoformat().replace("+00:00", "Z")
        local_dt = dt.astimezone()
        tz_name = local_dt.tzname() or "local time"
        local_str = local_dt.strftime("%Y-%m-%d %H:%M:%S ")
        local_str = f"{local_str}{tz_name}"
        return local_str, tz_name, utc_str

    def _summarize_schedule_reminder(self, result: Dict[str, Any]) -> str:
        parts: list[str] = []
        status = result.get("status") or "scheduled"
        recipient = result.get("recipient") or "self"
        description = result.get("description")
        next_run = result.get("next_run")
        localization = self._localize_timestamp(next_run) if next_run else None
        if localization:
            local_str, tz_name, utc_str = localization
            parts.append(
                f"Reminder {status} for {recipient}. Next run: {local_str} ({tz_name}), which is {utc_str}."
            )
        else:
            if next_run:
                parts.append(
                    f"Reminder {status} for {recipient}. Next run at {next_run} (timezone not detected)."
                )
            else:
                parts.append(f"Reminder {status} for {recipient}.")
        if description:
            parts.append(f"Description: {description}")
        metadata = result.get("metadata") or {}
        if isinstance(metadata, dict):
            action = metadata.get("action")
            target = metadata.get("target")
            if action or target:
                detail = ", ".join(
                    filter(
                        None,
                        [
                            f"action={action}" if action else "",
                            f"target={target}" if target else "",
                        ],
                    )
                )
                if detail:
                    parts.append(f"Delivery: {detail}")
        return "\n".join(parts).strip()

    def _tool_issue_hint(self, result: Any) -> str:
        if not isinstance(result, dict):
            return ""

        issues = []
        status = (result.get("status") or "").lower()
        exit_code = result.get("exit_code")
        timed_out = result.get("timed_out")
        stderr = (result.get("stderr") or "").strip()
        stdout = (result.get("stdout") or "").strip()

        if timed_out:
            issues.append("the command timed out")
        if exit_code not in (None, 0):
            issues.append(f"exit code {exit_code}")
        if status and status not in {"completed", "ok", "success"}:
            issues.append(f"status reported as '{status}'")
        if stderr:
            stderr_line = stderr.splitlines()[0]
            issues.append(f"stderr: {stderr_line}")
        if not stdout and not stderr:
            issues.append("no output was produced")

        if not issues:
            return ""

        issue_text = (
            "Important: the previous tool call may not have satisfied the request because "
            + ", ".join(issues)
            + ". You must correct the problem by running another tool call with a different command or parameters before using send_message. "
        )
        return issue_text

    def _interpret_stdout(self, stdout_lines: Optional[list[str]]) -> list[str]:
        interpretations: list[str] = []
        if not stdout_lines:
            return interpretations

        # Attempt to interpret common command outputs (ls, find timestamps, etc.)
        first_data_line: Optional[str] = None
        for line in stdout_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("total "):
                continue
            first_data_line = stripped
            break

        if first_data_line:
            tokens = first_data_line.split()
            # Handle BSD/macOS ls -lt output
            if len(tokens) >= 8 and tokens[-4].isalpha() and tokens[-3].isdigit():
                month = tokens[-4]
                day = tokens[-3]
                time_or_year = tokens[-2]
                name = " ".join(tokens[8:]) if len(tokens) > 8 else tokens[-1]
                perms = tokens[0]
                entry_type = "directory" if perms.startswith("d") else "file"
                interpretations.append(
                    f"Most recent {entry_type}: {name} (modified {month} {day} {time_or_year})."
                )
            # Handle GNU find -printf output: "<epoch> <path>"
            elif len(tokens) >= 2 and self._looks_like_epoch(tokens[0]):
                try:
                    epoch = float(tokens[0])
                    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    path = " ".join(tokens[1:])
                    interpretations.append(
                        f"Most recent file: {path} (modified {dt.isoformat(timespec='seconds')})."
                    )
                except ValueError:
                    pass

        return interpretations

    @staticmethod
    def _looks_like_epoch(token: str) -> bool:
        try:
            float(token)
            return True
        except (TypeError, ValueError):
            return False

    async def _attempt_final_response(
        self,
        *,
        summary: str,
        recipient: str,
        user_request: str,
        tool_name: str,
        tool_parameters: Dict[str, Any],
    ) -> bool:
        """Give the LLM one last chance to craft a proper send_message reply using the gathered summary."""
        if not self.component_config:
            return False

        instructions = (
            "You have gathered information using tools, but ran out of planning steps. "
            "Provide a final reply to the human based solely on the summary below. "
            "If the summary does not fully answer the request, clearly explain what is missing. "
            "Respond by emitting exactly one JSON object describing a send_message tool call."
        )
        summary_block = (
            f"User request: {user_request}\n"
            f"Last tool used: {tool_name} with parameters {json.dumps(tool_parameters, ensure_ascii=False)}\n"
            f"Tool summary:\n{summary}"
        )

        messages = [
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": summary_block,
            },
        ]

        try:
            response = await llm_chat(
                llm_model=self.model,
                messages=messages,
                api_base=self.api_config.get("api_base"),
                api_key=self.api_config.get("api_key"),
                response_format={"type": "json_object"},
            )
        except Exception:
            logging.exception("Final response attempt failed due to LLM error")
            return False

        try:
            function_data = extract_json(response)
        except Exception:
            logging.warning("Final response attempt did not produce valid JSON")
            return False

        if function_data.get("function") != "send_message":
            logging.warning("Final response attempt produced non send_message function")
            return False

        parameters = function_data.get("parameters", {}) or {}
        parameters["recipient"] = recipient

        try:
            await self.functions_registry.execute(
                "send_message",
                parameters,
                context=ToolExecutionContext(
                    agent=self.component_config.message_sender,
                    session_id=self._active_conversation_id,
                    metadata={
                        "conversation_id": self._active_conversation_id,
                        "requester": recipient,
                        "agent_name": self.name,
                        "delegation_path": list(self._active_delegation_path),
                        "task_id": self._active_task_id,
                    },
                ),
            )
            return True
        except Exception:
            logging.exception("Failed to execute final send_message response")
            return False

    async def _verify_task_completion(
        self,
        *,
        user_request: str,
        summary: str,
        last_tool_name: str,
        last_tool_parameters: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Ask the LLM to verify whether the user's request has been satisfied.

        Returns a structured directive:
            {"action": "send_message", "content": "..."} or
            {"action": "call_tool", "function": "...", "parameters": {...}, "reason": "..."}
        """
        if not self.component_config:
            return None

        available_tools = ", ".join(sorted(self.functions_registry.function_map.keys()))
        verifier_system_prompt = (
            "You are a verification assistant ensuring that an agent has satisfied the user's latest request. "
            "Review the summary of the most recent tool results and decide the next step. "
            "Respond with a JSON object in one of two forms:\n"
            '{ "action": "send_message", "content": "<final reply to user>" }\n'
            'or\n'
            '{ "action": "call_tool", "function": "<tool name>", "parameters": { ... }, "reason": "<why more work is needed>" }.\n'
            "If you choose send_message, ensure the content fully answers the user. "
            "Only choose call_tool if additional steps are required and specify valid parameters."
        )
        verifier_user_prompt = (
            f"User request:\n{user_request}\n\n"
            f"Latest tool used: {last_tool_name} with parameters {json.dumps(last_tool_parameters, ensure_ascii=False)}\n"
            f"Summary of tool output:\n{summary}\n\n"
            f"Available tools: {available_tools}"
        )

        try:
            response = await llm_chat(
                llm_model=self.model,
                messages=[
                    {"role": "system", "content": verifier_system_prompt},
                    {"role": "user", "content": verifier_user_prompt},
                ],
                api_base=self.api_config.get("api_base"),
                api_key=self.api_config.get("api_key"),
                response_format={"type": "json_object"},
            )
        except Exception:
            logging.exception("Verification LLM call failed")
            return None

        try:
            decision = json.loads(response)
        except Exception:
            try:
                decision = extract_json(response)
            except Exception:
                logging.warning("Verification step did not return valid JSON")
                return None

        if not isinstance(decision, dict):
            return None

        action = decision.get("action")
        if action == "send_message":
            content = decision.get("content")
            if content:
                return {"action": "send_message", "content": content}
            return None
        if action == "call_tool":
            function = decision.get("function")
            parameters = decision.get("parameters") or {}
            reason = decision.get("reason")
            payload = {"action": "call_tool", "function": function, "parameters": parameters}
            if reason:
                payload["reason"] = reason
            return payload
        return None

    async def _synthesize_fallback_response(
        self,
        *,
        user_request: str,
        summary: str,
        tool_name: str,
        tool_result: Any,
    ) -> Optional[str]:
        """Generate a concise human-friendly reply from the gathered tool output."""
        if not summary and not tool_result:
            return None

        try:
            if isinstance(tool_result, (dict, list)):
                raw_result = json.dumps(tool_result, ensure_ascii=False)
            else:
                raw_result = str(tool_result)
        except Exception:
            raw_result = str(tool_result)

        raw_excerpt = self._truncate(raw_result, limit=4000)
        synthesis_prompt = (
            "Summarize the retrieved information for the user. Provide a clear, concise response "
            "covering the main points relevant to their request. If information is missing, state what could not be found."
        )
        user_payload = (
            f"User request:\n{user_request}\n\n"
            f"Tool used: {tool_name}\n"
            f"Tool summary:\n{summary}\n\n"
            f"Raw result excerpt:\n{raw_excerpt}"
        )

        try:
            response = await llm_chat(
                llm_model=self.model,
                messages=[
                    {"role": "system", "content": synthesis_prompt},
                    {"role": "user", "content": user_payload},
                ],
                api_base=self.api_config.get("api_base"),
                api_key=self.api_config.get("api_key"),
            )
        except Exception:
            logging.exception("Failed to synthesize fallback response")
            return None

        cleaned = (response or "").strip()
        return cleaned or None

    async def _synthesize_failure_response(
        self,
        *,
        user_request: str,
        reason: str,
    ) -> Optional[str]:
        """Craft a helpful response when no tools succeeded."""
        prompt = (
            "You are an assistant who must acknowledge that automated tools could not satisfy the user's request. "
            "Explain the limitation clearly, offer constructive next steps, and ask for any extra details that could help."
        )
        user_payload = (
            f"User request:\n{user_request}\n\n"
            f"Why the automation failed:\n{reason}"
        )
        try:
            response = await llm_chat(
                llm_model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_payload},
                ],
                api_base=self.api_config.get("api_base"),
                api_key=self.api_config.get("api_key"),
            )
        except Exception:
            logging.exception("Failed to synthesize failure response")
            return None

        cleaned = (response or "").strip()
        return cleaned or None

    def _build_retry_prompt(
        self,
        *,
        reason: str,
        last_tool_name: Optional[str],
        last_tool_parameters: Optional[Dict[str, Any]],
        last_tool_result: Any,
        user_request: str,
    ) -> str:
        details = f"The user request is: {user_request!r}."
        tool_info = ""
        if last_tool_name:
            summary = self._summarize_tool_result(last_tool_name, last_tool_result)
            tool_info = (
                f" The last tool you executed was '{last_tool_name}' with parameters "
                f"{json.dumps(last_tool_parameters or {}, ensure_ascii=False)} and it produced:\n{summary}\n"
            )
        user_wants_json = "json" in user_request.lower() if user_request else False
        default_format_guidance = (
            "Unless the user explicitly asked for JSON/raw data, respond with polished natural language that highlights the key findings."
        )
        if user_wants_json:
            default_format_guidance = (
                "The user asked for JSON/raw data, so you may return it verbatim, optionally with a brief explanation."
            )
        return (
            f"The previous reply could not be used because {reason}. "
            f"{details}{tool_info}"
            "You must respond with a single JSON object describing EXACTLY one function call "
            "from the available tool list. Do not include explanatory text outside the JSON. "
            f"If you are ready to answer the human, call the send_message tool with a concise explanation. {default_format_guidance}"
        )

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
        agent = getattr(self.component_config, "message_sender", None) if self.component_config else None
        if agent and hasattr(agent, "planner") and getattr(agent.planner, "known_agents", None):
            known_entries = []
            for agent_name, info in agent.planner.known_agents.items():
                description = info.get("description") or "unknown"
                last_seen = info.get("last_seen", "unknown time")
                capabilities = info.get("capabilities") or {}
                if isinstance(capabilities, dict) and capabilities:
                    capability_list = ", ".join(sorted(str(k) for k in capabilities.keys()))
                else:
                    capability_list = "unspecified"
                known_entries.append(
                    f"- {agent_name}: {description} (last seen {last_seen}; capabilities: {capability_list})"
                )
            if not known_entries:
                known_entries.append("- None recorded")
            sections.append("Known agents observed via HELO/ACK:\n" + "\n".join(known_entries))
        local_now = datetime.now().astimezone()
        sections.append(f"Current datetime (local): {local_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        sections.append(f"Current datetime (UTC): {datetime.utcnow().isoformat()}Z")
        if extra_system_prompt:
            sections.append(extra_system_prompt)
        sections.append(function_instructions)

        combined_system_prompt = "\n\n".join(sections)
        
        messages = self.create_chat_messages_prompt(combined_system_prompt)
        
        # Extract API configuration
        api_base = self.api_config.get('api_base')
        api_key = self.api_config.get('api_key')
        
        try:
            reply = await llm_chat(
                llm_model=self.model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                response_format={"type": "json_object"},
            )
            return self.format_response(reply)
        except Exception as exc:
            logging.error("LLM call failed: %s", exc)
            fallback = Message(
                source=self.name,
                to=self.component_config.message_sender.attention,
                content="I'm having trouble reaching my language model right now. Please try again later.",
                message_type=MessageType.CHAT,
            )
            return fallback
    def set_active_context(
        self,
        conversation_id: Optional[str],
        source_message: Optional[Message],
        task_id: Optional[str] = None,
        delegation_path: Optional[List[str]] = None,
    ) -> None:
        self._active_conversation_id = conversation_id
        self._active_source_message = source_message
        self._active_task_id = task_id
        self._active_delegation_path = list(delegation_path or [])

    def clear_active_context(self) -> None:
        self._active_conversation_id = None
        self._active_source_message = None
        self._active_task_id = None
        self._active_delegation_path = []
