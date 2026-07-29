import asyncio
import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Literal

import jsonlines
import yaml
from jinja2 import Template
from jinja2.runtime import StrictUndefined
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ToolCall,
)
from pydantic import BaseModel

from deeppresenter.agents.env import AgentEnv
from deeppresenter.utils.config import (
    LLM,
    DeepPresenterConfig,
    get_json_from_response,
)
from deeppresenter.utils.constants import (
    AGENT_PROMPT,
    TOOLCALL_LIMIT_PROMPT,
    CONTEXT_MODE_PROMPT,
    CONTINUE_MSG,
    HALF_BUDGET_NOTICE_MSG,
    HIST_LOST_MSG,
    LAST_ITER_MSG,
    MA_RESEACHER_PROMPT,
    MA_RRESENTER_PROMPT,
    MAX_LOGGING_LENGTH,
    MAX_TOOLCALL_PER_TURN,
    MEMORY_COMPACT_MSG,
    OFFLINE_PROMPT,
    PACKAGE_DIR,
    URGENT_BUDGET_NOTICE_MSG,
)
from deeppresenter.utils.log import (
    debug,
    info,
    timer,
    sanitize_log_text,
)
from deeppresenter.utils.typings import (
    ChatMessage,
    Cost,
    InputRequest,
    Role,
    RoleConfig,
)


class Agent:
    def __init__(
        self,
        config: DeepPresenterConfig,
        agent_env: AgentEnv,
        workspace: Path,
        language: Literal["zh", "en"],
        config_file: str | Path | None = None,
        keep_reasoning: bool = True,
        max_turns: int | None = None,
    ):
        self.name = self.__class__.__name__
        self.config = config
        self.cost = Cost()
        self.context_length = 0
        self.context_warning = 0
        self.workspace = workspace
        self.agent_env = agent_env
        self.language = language
        self.keep_reasoning = keep_reasoning
        self.context_window = config.context_window
        self.max_context_turns = config.max_context_folds
        self.max_turns = max_turns
        self.turn_count = 0
        self._edit_failures: dict[str, int] = {}
        self._patch_html_failures: dict[str, int] = {}
        role_config_file = (
            Path(config_file)
            if config_file
            else PACKAGE_DIR / "roles" / f"{self.name}.yaml"
        )
        if not role_config_file.exists():
            raise FileNotFoundError(
                f"Cannot found role config file at: {role_config_file} "
            )

        # Setting basic context
        workspace.mkdir(parents=True, exist_ok=True)
        with open(role_config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        self.role_config = RoleConfig(**config_data)
        self.llm: LLM = config[self.role_config.use_model]
        self.model = self.llm.model_name
        self._setup_toolset()
        if self.tools:
            self.llm.require_capabilities("tools")
        if language not in self.role_config.system:
            raise ValueError(f"Language '{language}' not found in system prompts")
        self.error_history: list[ToolCall | ChatMessage] = []
        self.research_iter = 0
        if config.context_folding:
            self.context_warning = -1

        # Setting tools and interative context
        self.system = self.role_config.system[language]
        self.prompt: Template = Template(
            self.role_config.instruction, undefined=StrictUndefined
        )
        if any(t["function"]["name"] == "run_command" for t in self.tools):
            self.system += AGENT_PROMPT.format(
                workspace=self.workspace,
                cutoff_len=self.agent_env.cutoff_len,
                time=datetime.now().strftime("%Y-%m-%d"),
                max_toolcall_per_turn=MAX_TOOLCALL_PER_TURN,
            )
        else:
            # Agents without `run_command` (e.g. Design) never see the full
            # AGENT_PROMPT block above, which is where the toolcall limit is
            # normally mentioned. Still tell them the hard per-turn cap so an
            # agent that fans work out across many files (one write_file per
            # slide) knows to batch across turns instead of firing them all
            # in one turn and having the whole turn rejected.
            self.system += TOOLCALL_LIMIT_PROMPT.format(
                max_toolcall_per_turn=MAX_TOOLCALL_PER_TURN
            )

        if any(t["function"]["name"] == "delegate_subagent" for t in self.tools):
            if self.name == "Research":
                self.system += MA_RESEACHER_PROMPT
            elif self.name == "Design":
                self.system += MA_RRESENTER_PROMPT

        if config.offline_mode:
            self.system += OFFLINE_PROMPT

        if config.context_folding:
            self.system += CONTEXT_MODE_PROMPT

        self.chat_history: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system)
        ]
        available_tools = [tool["function"]["name"] for tool in self.tools]
        debug(
            f"{self.name} Agent got {len(self.tools)} tools: {', '.join(available_tools)}"
        )

    def _setup_toolset(self):
        toolset = self.role_config.toolset
        if toolset.include_tool_servers == "all":
            toolset.include_tool_servers = list(self.agent_env._server_tools)
        missing_servers = [
            server
            for server in toolset.include_tool_servers
            if server not in self.agent_env._server_tools
        ]
        if missing_servers:
            raise ValueError(
                f"Role {self.name} references unavailable tool servers: "
                f"{', '.join(missing_servers)}"
            )
        self.tools = []
        added_tool_names: set[str] = set()
        for server in toolset.include_tool_servers:
            if server not in toolset.exclude_tool_servers:
                for tool in self.agent_env._server_tools[server]:
                    if tool not in toolset.exclude_tools and tool not in added_tool_names:
                        self.tools.append(self.agent_env._tools_dict[tool])
                        added_tool_names.add(tool)

        missing_tools = set(toolset.include_tools) - self.agent_env._tools_dict.keys()
        if missing_tools:
            raise ValueError(
                f"Role {self.name} references unavailable tools: "
                f"{', '.join(sorted(missing_tools))}"
            )
        # `include_tools` may legitimately re-list a tool already pulled in
        # via `include_tool_servers` (e.g. to document intent), so skip
        # anything already added instead of appending a duplicate schema.
        # Some providers (e.g. kimi-k3) reject tool lists with a repeated
        # function name outright, so this dedupe is required for
        # correctness, not just tidiness.
        for tool_name, tool in self.agent_env._tools_dict.items():
            if tool_name in toolset.include_tools and tool_name not in added_tool_names:
                self.tools.append(tool)
                added_tool_names.add(tool_name)

    async def chat(
        self,
        message: ChatMessage,
        response_format: type[BaseModel] | None = None,
        **chat_kwargs,
    ) -> ChatMessage:
        if len(self.chat_history) == 1:
            self.chat_history.append(
                ChatMessage(role=Role.USER, content=self.prompt.render(**chat_kwargs))
            )
            self.log_message(self.chat_history[-1])
        self.chat_history.append(message)
        self.log_message(self.chat_history[-1])
        with timer(f"{self.name} Agent LLM chat"):
            response = await self.llm.run(
                messages=self.chat_history,
                response_format=response_format,
            )
            if response.usage is not None:
                self.cost += response.usage
                self.context_length = response.usage.total_tokens
            self.chat_history.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    content=response.choices[0].message.content,
                    cost=response.usage,
                    reasoning=getattr(response.choices[0].message, "reasoning", None)
                    if self.keep_reasoning
                    else None,
                )
            )
            self.log_message(self.chat_history[-1])
            return self.chat_history[-1]

    async def action(
        self,
        **chat_kwargs,
    ):
        """Tool calling interface"""
        self.turn_count += 1
        if self.max_turns is not None:
            if self.turn_count > self.max_turns:
                raise RuntimeError(
                    f"{self.name} exceeded max turns: {self.turn_count - 1}/{self.max_turns}"
                )
            if self.max_turns - self.turn_count < 2 and self.chat_history:
                self.chat_history[-1].content.append(
                    {
                        "type": "text",
                        "text": f"You have only {self.max_turns - self.turn_count} turn left. Finish the remaining work and call `finalize` immediately.",
                    }
                )

        if len(self.chat_history) == 1:
            self.chat_history.append(
                ChatMessage(
                    role=Role.USER,
                    content=self.prompt.render(**chat_kwargs),
                )
            )
            self.log_message(self.chat_history[-1])

        with timer(f"{self.name} Agent LLM call"):
            response = await self.llm.run(
                messages=self.chat_history,
                tools=self.tools,
            )
            if response.usage is not None:
                self.cost += response.usage
                self.context_length = response.usage.total_tokens
            agent_message: ChatCompletionMessage = response.choices[0].message
        self.chat_history.append(
            ChatMessage(
                role=Role.ASSISTANT,
                content=agent_message.content,
                cost=response.usage,
                tool_calls=agent_message.tool_calls,
                reasoning=getattr(agent_message, "reasoning", None)
                if self.keep_reasoning
                else None,
            )
        )
        self.log_message(self.chat_history[-1])
        return self.chat_history[-1]

    @abstractmethod
    def loop(
        self, req: InputRequest, *args, **kwargs
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """
        Loop interface, return the message or the outcome filepath of the agent.
        """

    @staticmethod
    def _tool_signature(tool_call: ToolCall) -> str:
        """Return a canonical signature for retry-sensitive tool calls."""
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = tool_call.function.arguments
        return json.dumps(
            {"tool": tool_call.function.name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _edit_failure_observation(
        self, tool_call: ToolCall, observation: ChatMessage
    ) -> ChatMessage:
        """Open a circuit for repeated brittle repair calls and prescribe IDs."""
        tool_name = tool_call.function.name
        if tool_name == "edit_file":
            if "Expected exactly one match" not in observation.text:
                return observation
            failures = self._edit_failures
            label = "EDIT_FILE_TERMINAL"
            condition = "the same exact-text replacement failed twice"
        elif tool_name == "patch_html":
            if not observation.is_error:
                return observation
            failures = self._patch_html_failures
            label = "PATCH_HTML_CIRCUIT_OPEN"
            condition = "the same selector patch failed twice"
        else:
            return observation

        signature = self._tool_signature(tool_call)
        count = failures.get(signature, 0) + 1
        failures[signature] = count
        if count < 2:
            return observation
        observation.content = [
            {
                "type": "text",
                "text": (
                    f"{label}: {condition}. This exact call is now blocked for the "
                    "rest of this run; do not retry it. Call inspect_slide_element(path) "
                    "to obtain the real data-slidex-id index, inspect the target ID, then "
                    "use patch_slide_element for a targeted repair. patch_html is reserved "
                    "for a single shared body/.slide-content/class-level container."
                ),
            }
        ]
        observation.is_error = False
        return observation

    def _circuit_open_observation(self, tool_call: ToolCall) -> ChatMessage | None:
        """Reject a patch_html retry after its failure circuit has opened."""
        if tool_call.function.name != "patch_html":
            return None
        if self._patch_html_failures.get(self._tool_signature(tool_call), 0) < 2:
            return None
        return ChatMessage(
            role=Role.TOOL,
            content=(
                "PATCH_HTML_CIRCUIT_OPEN: this exact failed patch_html call is blocked. "
                "Do not retry it. Enumerate inspect_slide_element(path), select a real "
                "data-slidex-id, inspect that element, and use patch_slide_element; only "
                "use patch_html for one shared body/.slide-content/class-level container."
            ),
            tool_call_id=tool_call.id,
            is_error=False,
        )

    async def execute(self, tool_calls: list[ToolCall]) -> str | list[ChatMessage]:
        coros = []
        observations: list[ChatMessage] = []
        used_tools = set()
        executable_calls = tool_calls
        if len(tool_calls) > MAX_TOOLCALL_PER_TURN:
            info(
                "%s Agent executes %s tool calls in bounded batches of %s",
                self.name, len(tool_calls), MAX_TOOLCALL_PER_TURN,
            )
        finish_id = None
        outcome = None
        for t in executable_calls:
            circuit_observation = self._circuit_open_observation(t)
            if circuit_observation is not None:
                observations.append(circuit_observation)
                continue
            arguments = t.function.arguments
            if len(arguments) == 0:
                arguments = None
            else:
                try:
                    arguments = get_json_from_response(t.function.arguments)
                    assert isinstance(arguments, dict), (
                        f"Tool call arguments must be a dict or empty, while {arguments} is given"
                    )
                    if t.function.name == "finalize":
                        arguments["agent_name"] = self.name
                        finish_id = t.id
                        assert "outcome" in arguments, (
                            "Finalize tool call must have an outcome"
                        )
                        outcome_path = Path(arguments["outcome"])
                        if outcome_path.is_absolute() and outcome_path.parts[:2] == (
                            "/",
                            "workspace",
                        ):
                            outcome_path = self.workspace.joinpath(
                                *outcome_path.parts[2:]
                            )
                            arguments["outcome"] = str(outcome_path)
                        outcome = arguments["outcome"]
                    t.function.arguments = json.dumps(arguments, ensure_ascii=False)
                except AssertionError as e:
                    observations.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=str(e),
                            tool_call_id=t.id,
                            is_error=True,
                        )
                    )
                    info(f"Tool call `{t.function}` encountered error: {e}")
                    continue
            used_tools.add(t.function.name)
            info(f"{self.name} Agent calling tool `{t.function.name}`")
            coros.append(self.agent_env.tool_execute(t))

        for start in range(0, len(coros), MAX_TOOLCALL_PER_TURN):
            observations.extend(
                await asyncio.gather(*coros[start : start + MAX_TOOLCALL_PER_TURN])
            )
        tool_call_map = {t.id: t for t in executable_calls}
        observations = [
            self._edit_failure_observation(tool_call_map[obs.tool_call_id], obs)
            if obs.tool_call_id in tool_call_map
            else obs
            for obs in observations
        ]
        for obs in observations:
            if obs.has_image:
                if "gemini" in self.model.lower() or "qwen" in self.model.lower():
                    obs.role = Role.USER
                if "claude" in self.model.lower():
                    oai_b64 = obs.content[0]["image_url"]["url"]
                    obs.content = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": oai_b64.split(";")[0].split(":")[1],
                                "data": oai_b64.split(",")[1],
                            },
                        }
                    ]

        self.chat_history.extend(observations)

        for o in observations:
            if o.is_error:
                t = tool_call_map[o.tool_call_id]
                self.error_history.append(t)
                self.error_history.append(o)

        if finish_id is not None:
            for obs in observations:
                if obs.tool_call_id == finish_id and obs.text == outcome:
                    info(f"{self.name} Agent finished with result: {obs.text}")
                    return obs.text

        if (
            self.context_warning == 0
            and self.context_length > self.context_window * 0.5
        ):
            self.context_warning += 1
            observations[0].content.insert(0, HALF_BUDGET_NOTICE_MSG)
        elif (
            self.context_warning == 1
            and self.context_length > self.context_window * 0.8
        ):
            observations[0].content.insert(0, URGENT_BUDGET_NOTICE_MSG)
            self.context_warning = 2

        for obs in observations:
            self.log_message(obs)

        if self.context_length > self.context_window:
            if self.context_warning == -1:
                await self.compact_history()
            else:
                raise RuntimeError(
                    f"{self.name} agent exceeded context window: {self.context_length}/{self.context_window}"
                )
        return observations

    def log_message(self, msg: ChatMessage) -> None:
        """Log bounded metadata and sanitized text, never inline binary content."""
        text = sanitize_log_text(msg.text)
        preview = text[:MAX_LOGGING_LENGTH]
        suffix = "..." if len(text) > MAX_LOGGING_LENGTH else ""
        debug(
            "%s role=%s chars=%d content=%s%s",
            self.name,
            msg.role.value,
            len(text),
            preview,
            suffix,
        )

    async def compact_history(self, keep_head: int = 10, keep_tail: int = 4):
        """Summarize the history."""
        # ? it's 10 = system + user + (thinking, read, design, write)*2
        if keep_head + keep_tail > len(self.chat_history):
            return

        if self.research_iter == self.max_context_turns:
            return

        self.save_history(message_only=True)
        self.research_iter += 1
        head, tail = self._split_history(keep_head, keep_tail)
        summary_ask = ChatMessage(
            role=Role.USER, content=MEMORY_COMPACT_MSG.format(language=self.language)
        )
        response = await self.llm.run(
            self.chat_history + [summary_ask],
            tools=self.tools,
        )
        agent_message = response.choices[0].message
        summary_message = ChatMessage(
            id=f"context_fold_{uuid.uuid4().hex[:8]}",
            role=agent_message.role,
            content=agent_message.content,
            tool_calls=agent_message.tool_calls,
            reasoning=getattr(agent_message, "reasoning", None)
            if self.keep_reasoning
            else None,
        )
        debug(
            f"Summary of Resarch Iter {self.research_iter:02d}: \n"
            + summary_message.text
        )
        tasks = [
            self.agent_env.tool_execute(tc) for tc in summary_message.tool_calls or []
        ]
        observations = await asyncio.gather(*tasks)
        observations[-1].content.append(CONTINUE_MSG)
        if self.research_iter == self.max_context_turns:
            observations[-1].content.append(LAST_ITER_MSG)
        new_tail = [
            summary_ask,
            summary_message,
            *observations,
        ]
        self.chat_history = head + tail + new_tail

    def _split_history(self, keep_head, keep_tail):
        # ensure the left context window contains the paired tool call and tool call result
        head = []
        for msg in self.chat_history:
            if len(head) < keep_head or msg.role == Role.TOOL:
                head.append(msg)
            else:
                break
        head[-1].content.append(HIST_LOST_MSG)

        tail = self.chat_history[-keep_tail:]
        for i, m in enumerate(tail):
            if m.role == Role.ASSISTANT and m not in head:
                tail = tail[i:]
                break
        else:
            tail = []

        return head, tail

    def save_history(self, hist_dir: Path | None = None, message_only: bool = False):
        hist_dir = hist_dir or self.workspace / ".history"
        hist_dir.mkdir(parents=True, exist_ok=True)

        history_file = hist_dir / f"{self.name}-history.jsonl"
        if self.research_iter >= 0:
            history_file = (
                hist_dir / f"{self.name}-{self.research_iter:02d}-history.jsonl"
            )
        with jsonlines.open(history_file, mode="w") as writer:
            for message in self.chat_history:
                writer.write(message.model_dump())

        if message_only:
            return

        config_file = hist_dir / f"{self.name}-config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "model": self.model,
                    "context_window": self.context_length,
                    "cost": self.cost.model_dump(),
                    "tools": self.tools,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if self.error_history:
            error_file = hist_dir / f"{self.name}-errors.jsonl"
            with jsonlines.open(error_file, mode="w") as writer:
                for msg in self.error_history:
                    writer.write(msg.model_dump())

        debug(
            f"{self.name} done | cost:{self.cost} ctx:{self.context_length} | history:{history_file.name} config:{config_file.name}"
        )
