from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from .client import LLMClient, ToolCall
from .compression import CompressionService, CompressionStatus, hard_truncate_to_limit, estimate_messages_tokens
from .config import Config, build_system_prompt
from .logger import APILogger, create_logger
from .memory_log import MemoryLogger
from .multi_agent import agent_execution_context
from .session import SessionRecordingService, ConversationRecord
from .telemetry import SessionMetrics, TokenUsageStats
from .tools import ToolRegistry, WritePlanTool, default_registry
from .tools.plan import PLAN_READY_SENTINEL
from .tools.profiles import get_profile, infer_profile

_PLAN_SYSTEM_ADDENDUM = (
    "\n\nIMPORTANT: For this task you MUST call `write_plan` with a list of steps "
    "BEFORE using any other tool. You may use read-only tools (read_file, glob, grep, "
    "list_dir, shell with read-only commands) to explore first, then submit your plan."
)


@dataclass
class TurnEvent:
    """Events emitted during a single agent turn."""
    type: str   # "text"|"tool_start"|"tool_end"|"plan_ready"|"error"|"done"|"usage"|"compressed"
    data: Any = None


@dataclass
class AgentState:
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: list[ToolCall] | None = None) -> None:
        # Use None instead of "" when content is empty — the API rejects empty text blocks
        msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _dump(tc.arguments)},
                }
                for tc in tool_calls
            ]
        self.messages.append(msg)  # type: ignore[arg-type]

    def add_tool_result(self, tool_call_id: str, name: str, result: str) -> None:
        self.messages.append({           # type: ignore[arg-type]
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": result,
        })

    def reset(self) -> None:
        """Keep only the system prompt."""
        system = [m for m in self.messages if m["role"] == "system"]
        self.messages = system


class Agent:
    """
    Core agent loop.

    Basic usage:
        for event in agent.run("list all python files"):
            ...

    Plan-then-execute usage:
        for event in agent.generate_plan("refactor all functions"):
            if event.type == "plan_ready":
                # show plan, ask approval
                ...
        if approved:
            for event in agent.execute():
                ...
    """

    def __init__(
        self,
        config: Config | None = None,
        registry: ToolRegistry | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        agent_depth: int = 0,
        logs_dir: str | None = None,
        memory_log_dir: str | None = None,
    ):
        self.config = config or Config()
        self.agent_id = agent_id
        self.agent_depth = agent_depth

        if registry is not None:
            self.registry = registry
        else:
            profile_name = self.config.tool_profile
            if profile_name == "auto":
                profile_name = infer_profile(self.config.model)
            self.profile_name = profile_name
            self.registry = get_profile(profile_name).build_registry()

        self.logger = create_logger(logs_dir=logs_dir or "api_logs")
        self.client = LLMClient(self.config, logger=self.logger)
        self.state = AgentState()
        self._plan_tool = WritePlanTool()
        self._compression = CompressionService()
        self._has_failed_compression = False
        self._memory_logger = MemoryLogger(log_dir=memory_log_dir or "memory_logs")
        self.metrics = SessionMetrics()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.logger.start_session(self.session_id, self.config.model)
        self.recorder = SessionRecordingService()
        self.recorder.create_session(self.session_id)
        self._init_system_prompt()

    # ------------------------------------------------------------------
    # Public API — session management
    # ------------------------------------------------------------------

    def resume_session(self, session_id: str) -> ConversationRecord | None:
        """
        Resume a previous session by restoring its messages into state.

        How it works (mirroring the template_proj approach):
        1. Load the ConversationRecord from the session JSON file on disk
        2. Reset current agent state
        3. Re-inject the system prompt
        4. Replay all saved messages back into agent state using the
           same format the OpenAI API expects:
             - user messages     → state.add_user()
             - assistant messages → state.add_assistant() (with tool_calls)
             - tool messages      → state.add_tool_result()
        5. Point the recorder at the resumed session file so new
           messages continue appending to the same file.

        The LLM then sees the full conversation history on the next
        request, exactly as if the session had never been interrupted.
        """
        record = self.recorder.resume_session(session_id)
        if record is None:
            return None

        self.session_id = record.session_id

        # Reset state and re-init system prompt
        self.state = AgentState()
        self._init_system_prompt()

        # Replay saved messages into agent state
        for msg in record.messages:
            if msg.role == "system":
                continue  # Already set above
            elif msg.role == "user":
                self.state.add_user(msg.content)
            elif msg.role == "assistant":
                tool_calls_objs = None
                if msg.tool_calls:
                    tool_calls_objs = [
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc.get("arguments", {}),
                        )
                        for tc in msg.tool_calls
                    ]
                self.state.add_assistant(msg.content, tool_calls_objs)
            elif msg.role == "tool":
                # Tool result messages have tool_call_id and name
                self.state.add_tool_result(
                    tool_call_id=msg.tool_call_id or "",
                    name=msg.name or "",
                    result=msg.content,
                )

        return record

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions."""
        return self.recorder.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        return self.recorder.delete_session(session_id)

    # ------------------------------------------------------------------
    # Public API — normal mode
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> Iterator[TurnEvent]:
        """Process one user message with full tool access."""
        self.state.add_user(user_input)
        self.recorder.save_message("user", user_input)
        yield from self._agent_loop(self.registry)
        yield TurnEvent(type="done")

    # ------------------------------------------------------------------
    # Public API — plan-then-execute mode
    # ------------------------------------------------------------------

    def generate_plan(self, user_input: str) -> Iterator[TurnEvent]:
        """
        Phase 1: explore and produce a plan.
        Stops as soon as write_plan is called and yields a plan_ready event.
        """
        plan_registry = ToolRegistry()
        for schema in self.registry.schemas():
            tool = self.registry.get(schema["function"]["name"])
            if tool:
                plan_registry.register(tool)
        plan_registry.register(self._plan_tool)

        self._push_planning_prompt()
        self.state.add_user(user_input)
        self.recorder.save_message("user", user_input)

        turn_start = time.time()
        cumulative_usage = TokenUsageStats()

        for iteration in range(self.config.max_tool_iterations):
            for event in self._llm_turn(plan_registry):
                if event.type == "usage":
                    cumulative_usage += event.data
                yield event

            tool_calls = self._last_tool_calls()
            if not tool_calls:
                cumulative_usage.latency_ms = (time.time() - turn_start) * 1000
                yield TurnEvent(type="usage", data=cumulative_usage) 
                yield TurnEvent(type="error", data="[agent] no plan was submitted.")
                yield TurnEvent(type="done")
                return

            plan_submitted = False
            for tc in tool_calls:
                yield TurnEvent(type="tool_start", data={"name": tc.name, "arguments": tc.arguments})
                tool_start_time = time.time()
                result = plan_registry.execute(tc.name, tc.arguments)
                tool_duration_ms = (time.time() - tool_start_time) * 1000
                success = not result.startswith("[error]") if isinstance(result, str) else True
                self.metrics.add_tool_call(tc.name, success, tool_duration_ms)
                self.logger.log_tool_exec(
                    tc.name, tc.arguments, result, success, tool_duration_ms,
                )
                yield TurnEvent(type="tool_end", data={"name": tc.name, "result": result})
                self.state.add_tool_result(tc.id, tc.name, result)
                self.recorder.save_message(
                    "tool", result, tool_call_id=tc.id, name=tc.name,
                )

                if result == PLAN_READY_SENTINEL and self._plan_tool.pending:
                    plan_submitted = True

            if plan_submitted:
                cumulative_usage.latency_ms = (time.time() - turn_start) * 1000
                yield TurnEvent(type="usage", data=cumulative_usage) 
                yield TurnEvent(type="plan_ready", data=self._plan_tool.pending)
                self._pop_planning_prompt()
                return

        cumulative_usage.latency_ms = (time.time() - turn_start) * 1000
        yield TurnEvent(type="usage", data=cumulative_usage)
        yield TurnEvent(type="error", data="[agent] hit max iterations without submitting a plan.")
        self._pop_planning_prompt()
        yield TurnEvent(type="done")

    def execute(self) -> Iterator[TurnEvent]:
        """
        Phase 2: execute the approved plan.
        Call this after generate_plan() yields a plan_ready event and user approves.
        """
        plan = self._plan_tool.pending
        if not plan:
            yield TurnEvent(type="error", data="[agent] no plan to execute.")
            yield TurnEvent(type="done")
            return

        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan["steps"]))
        exec_msg = f"Plan approved. Execute it now step by step:\n{steps_text}"
        self.state.add_user(exec_msg)
        self.recorder.save_message("user", exec_msg)

        yield from self._agent_loop(self.registry)
        self._plan_tool.pending = None
        yield TurnEvent(type="done")

    def reset(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self.state.reset()
        self._plan_tool.pending = None

    def save_session(self) -> None:
        """Save current metrics to the session file."""
        self.recorder.save_metrics(self.metrics)

    @property
    def history(self) -> list[ChatCompletionMessageParam]:
        return self.state.messages

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_system_prompt(self) -> None:
        # Build the system prompt dynamically from the actual registered tools
        # so it always matches the tools available for the current model/profile.
        prompt = self.config.system_prompt
        if not prompt:
            tool_names = [s["function"]["name"] for s in self.registry.schemas()]
            prompt = build_system_prompt(tool_names)
        self.state.messages.insert(0, {
            "role": "system",
            "content": prompt,
        })

    def _push_planning_prompt(self) -> None:
        msg = self.state.messages[0]
        self.state.messages[0] = {
            "role": "system",
            "content": str(msg["content"]) + _PLAN_SYSTEM_ADDENDUM,
        }

    def _pop_planning_prompt(self) -> None:
        msg = self.state.messages[0]
        content = str(msg["content"]).replace(_PLAN_SYSTEM_ADDENDUM, "")
        self.state.messages[0] = {"role": "system", "content": content}

    def _try_compress(self) -> TurnEvent | None:
        """Run compression check; returns a 'compressed' event if history was compacted."""
        messages_before = list(self.state.messages)  # snapshot before any mutation

        result = self._compression.maybe_compress(
            self.state.messages,
            self.config,
            self.client,
            self._has_failed_compression,
        )
        if result.status == CompressionStatus.FAILED_INFLATED:
            self._has_failed_compression = True
            self._memory_logger.log(
                session_id=self.session_id,
                status=result.status.value,
                original_tokens=result.original_tokens,
                new_tokens=result.new_tokens,
                messages_before=messages_before,
                messages_after=None,
            )
            return None
        if result.status in (CompressionStatus.COMPRESSED, CompressionStatus.CONTENT_TRUNCATED):
            if result.new_messages is not None:
                self.state.messages = result.new_messages
                if result.status == CompressionStatus.COMPRESSED:
                    self._has_failed_compression = False
            self._memory_logger.log(
                session_id=self.session_id,
                status=result.status.value,
                original_tokens=result.original_tokens,
                new_tokens=result.new_tokens,
                messages_before=messages_before,
                messages_after=result.new_messages,
            )
            return TurnEvent(
                type="compressed",
                data={
                    "status": result.status.value,
                    "original_tokens": result.original_tokens,
                    "new_tokens": result.new_tokens,
                },
            )

        # Hard safety net: enforce the absolute context_limit even when soft
        # compression did not fire (NOOP) or previously failed.
        truncated_msgs, was_truncated = hard_truncate_to_limit(
            self.state.messages, self.config.context_limit
        )
        if was_truncated:
            original_tokens = estimate_messages_tokens(self.state.messages)
            new_tokens = estimate_messages_tokens(truncated_msgs)
            self.state.messages = truncated_msgs
            self._memory_logger.log(
                session_id=self.session_id,
                status="hard_truncated",
                original_tokens=original_tokens,
                new_tokens=new_tokens,
                messages_before=messages_before,
                messages_after=truncated_msgs,
            )
            return TurnEvent(
                type="compressed",
                data={
                    "status": "hard_truncated",
                    "original_tokens": original_tokens,
                    "new_tokens": new_tokens,
                },
            )

        return None

    def _agent_loop(self, registry: ToolRegistry) -> Iterator[TurnEvent]:
        """Core loop: call LLM, run tools, repeat until no tool calls."""
        turn_start = time.time()
        cumulative_usage = TokenUsageStats()

        for _ in range(self.config.max_tool_iterations):
            # Compress before every LLM call so tool results added in previous
            # iterations don't push the payload over the context limit.
            compress_event = self._try_compress()
            if compress_event:
                yield compress_event

            for event in self._llm_turn(registry):
                if event.type == "usage":
                    cumulative_usage += event.data
                yield event

            tool_calls = self._last_tool_calls()
            if not tool_calls:
                cumulative_usage.latency_ms = (time.time() - turn_start) * 1000
                yield TurnEvent(type="usage", data=cumulative_usage)
                return

            for tc in tool_calls:
                yield TurnEvent(type="tool_start", data={"name": tc.name, "arguments": tc.arguments})
                tool_start_time = time.time()
                with agent_execution_context(
                    config=self.config,
                    registry=registry,
                    agent_id=self.agent_id,
                    depth=self.agent_depth,
                ):
                    result = registry.execute(tc.name, tc.arguments)
                tool_duration_ms = (time.time() - tool_start_time) * 1000
                success = not result.startswith("[error]") if isinstance(result, str) else True
                self.metrics.add_tool_call(tc.name, success, tool_duration_ms)
                self.logger.log_tool_exec(
                    tc.name, tc.arguments, result, success, tool_duration_ms,
                )
                yield TurnEvent(type="tool_end", data={"name": tc.name, "result": result})
                self.state.add_tool_result(tc.id, tc.name, result)
                # Record tool result to session file
                self.recorder.save_message(
                    "tool", result, tool_call_id=tc.id, name=tc.name,
                )

        # Final usage even if we hit max iterations
        cumulative_usage.latency_ms = (time.time() - turn_start) * 1000
        yield TurnEvent(type="usage", data=cumulative_usage)
        yield TurnEvent(type="error", data=f"[agent] hit max tool iterations ({self.config.max_tool_iterations})")

    def _llm_turn(self, registry: ToolRegistry) -> Iterator[TurnEvent]:
        """Send current messages to LLM, stream back text, record response."""
        turn_start = time.time()
        response = self.client.chat(
            messages=self.state.messages,
            tools=registry.schemas() if len(registry) > 0 else None,
        )
        for token in response.text_chunks():
            if token:
                yield TurnEvent(type="text", data=token)

        latency_ms = (time.time() - turn_start) * 1000

        # Yield usage event immediately after generation finishes
        if response.usage:
            response.usage.latency_ms = latency_ms
            yield TurnEvent(type="usage", data=response.usage) 

        # Record token usage from this API call
        self.metrics.add_api_response(
            model_name=self.config.model,
            tokens=response.usage,
            latency_ms=latency_ms,
        )

        self.state.add_assistant(response.content, response.tool_calls or None)

        # Record the assistant message to the session file
        tool_calls_data = None
        if response.tool_calls:
            tool_calls_data = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]
        self.recorder.save_message("assistant", response.content, tool_calls_data)

    def _last_tool_calls(self) -> list[ToolCall]:
        """Return tool calls from the most recent assistant message."""
        for msg in reversed(self.state.messages):
            if msg.get("role") == "assistant":
                raw = msg.get("tool_calls", [])
                if not raw:
                    return []
                calls = []
                for tc in raw:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
                return calls
        return []


def _dump(obj: Any) -> str:
    return json.dumps(obj) if isinstance(obj, dict) else str(obj)
