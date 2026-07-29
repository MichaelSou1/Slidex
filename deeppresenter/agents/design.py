import json

from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ToolCall,
)

from deeppresenter.agents.agent import Agent
from deeppresenter.slidex.authoring import authoring_skill
from deeppresenter.utils.constants import SLIDE_QUOTA_EXCEEDED_MSG_TEMPLATE
from deeppresenter.utils.log import info
from deeppresenter.utils.typings import ChatMessage, InputRequest, Role

# Tool argument key that names the slide file each tool operates on.
_SLIDE_FILE_ARG_BY_TOOL = {
    "write_file": "path",
    "edit_file": "path",
    "inspect_slide": "html_file",
    "patch_html": "path",
    "patch_slide_element": "path",
}


def _slide_file_from_tool_call(tool_call: ToolCall) -> str | None:
    """Return the ``*.html`` slide file a write_file/edit_file/inspect_slide call targets."""
    arg_key = _SLIDE_FILE_ARG_BY_TOOL.get(tool_call.function.name)
    if arg_key is None:
        return None
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return None
    slide_file = arguments.get(arg_key)
    if not isinstance(slide_file, str) or not slide_file.endswith(".html"):
        return None
    return slide_file


class Design(Agent):
    """Design agent: writes and inspects one HTML slide file per iteration.

    Enforces a per-slide turn quota (13.7 E2E hardening) on top of the
    shared ``max_episode_steps`` budget: if the agent spends more than
    ``config.slidex.max_turns_per_slide`` consecutive turns editing or
    inspecting the same slide file, it is nudged to move on so one hard
    slide cannot exhaust the whole episode's turn budget and leave later
    slides unwritten. This only reorders effort across slides; every slide
    must still pass `inspect_slide` before finalize, since the export gate
    (deeppresenter/main.py) rejects any deck with uninspected slides.
    """

    async def loop(self, req: InputRequest, markdown_file: str):
        (self.workspace / "slides").mkdir(exist_ok=True)
        streak_file: str | None = None
        streak_count = 0
        quota = self.config.slidex.max_turns_per_slide
        prompt = f"{req.designagent_prompt}\n\n{authoring_skill()}"
        repair_requires_inspection = False
        while True:
            agent_message = await self.action(markdown_file=markdown_file, prompt=prompt)
            yield agent_message
            calls = self.chat_history[-1].tool_calls or []
            repair_gate = _repair_inspection_gate(
                calls,
                requires_inspection=repair_requires_inspection,
                enabled=getattr(self, "require_repair_inspection", False),
            )
            if repair_gate is not None:
                gate_observations = _repair_gate_observations(calls, repair_gate.text)
                self.chat_history.extend(gate_observations)
                for observation in gate_observations:
                    yield observation
                continue
            repaired = _has_source_repair(calls)
            inspected = _has_slide_inspection(calls)
            streak_file, streak_count = _update_streak(
                streak_file, streak_count, calls
            )
            outcome = await self.execute(calls)
            if repaired:
                repair_requires_inspection = True
            elif inspected:
                repair_requires_inspection = False
            if isinstance(outcome, list):
                if streak_count > quota:
                    info(
                        f"Design Agent exceeded per-slide quota "
                        f"({streak_count}/{quota}) on `{streak_file}`, nudging to move on"
                    )
                    outcome[-1].content.append(
                        {
                            "type": "text",
                            "text": SLIDE_QUOTA_EXCEEDED_MSG_TEMPLATE.format(
                                turns=streak_count, slide=streak_file
                            ),
                        }
                    )
                    streak_file, streak_count = None, 0
                for item in outcome:
                    yield item
            else:
                break

        yield outcome


def _update_streak(
    streak_file: str | None,
    streak_count: int,
    tool_calls: list[ToolCall] | None,
) -> tuple[str | None, int]:
    """Track consecutive turns spent on a single slide file.

    A turn counts toward the streak only when it names exactly one slide
    file; turns that touch zero or multiple slide files (e.g. reading the
    manuscript, or a batched multi-slide call) reset the streak, since no
    single slide is being fixated on.
    """
    slide_files = {
        slide_file
        for call in tool_calls or []
        if (slide_file := _slide_file_from_tool_call(call)) is not None
    }
    if len(slide_files) != 1:
        return None, 0
    slide_file = next(iter(slide_files))
    if slide_file == streak_file:
        return slide_file, streak_count + 1
    return slide_file, 1


def _has_source_repair(tool_calls: list[ToolCall]) -> bool:
    """Whether a turn mutates existing HTML during a repair-only run."""
    return any(
        call.function.name in {"patch_html", "patch_slide_element", "edit_file", "apply_repair"}
        for call in tool_calls
    )


def _has_slide_inspection(tool_calls: list[ToolCall]) -> bool:
    """Whether a turn requests the rendered inspection that closes a repair step."""
    return any(call.function.name == "inspect_slide" for call in tool_calls)


def _repair_inspection_gate(
    tool_calls: list[ToolCall], *, requires_inspection: bool, enabled: bool
):
    """Prevent unverified repair chains from consuming the episode budget.

    A source patch is only meaningful once the rendered critic has seen it.
    In repair-only E2E, allow exactly one mutation turn, then require a
    separate inspect_slide turn before another mutation. Keeping inspection in
    a separate LLM turn avoids a parallel tool batch inspecting the pre-patch
    DOM snapshot.
    """
    if not enabled or not _has_source_repair(tool_calls):
        return None
    if requires_inspection or _has_slide_inspection(tool_calls):
        return ChatMessage(
            role=Role.TOOL,
            content=(
                "REPAIR_INSPECTION_REQUIRED: source changes are blocked until a separate "
                "inspect_slide call verifies the previous patch. Do not combine mutation "
                "tools and inspect_slide in one turn: they execute concurrently. Call "
                "inspect_slide for the affected slide now; inspect its hard findings and "
                "repair_actions before choosing one different targeted action."
            ),
            is_error=False,
        )
    return None


def _repair_gate_observations(tool_calls: list[ToolCall], message: str) -> list[ChatMessage]:
    """Return one synthetic tool response per blocked tool call.

    OpenAI-compatible APIs require every tool_call_id in an assistant message
    to be acknowledged before the next model turn, even when the harness
    blocks the complete batch before executing any tool.
    """
    return [
        ChatMessage(
            role=Role.TOOL,
            content=message if index == 0 else "REPAIR_INSPECTION_REQUIRED: batch blocked; call inspect_slide next.",
            tool_call_id=call.id,
            is_error=False,
        )
        for index, call in enumerate(tool_calls)
    ]
