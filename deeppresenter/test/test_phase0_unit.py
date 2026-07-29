import json
from pathlib import Path

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ToolCall,
)

from deeppresenter.agents.agent import Agent
from deeppresenter.agents.env import AgentEnv
from deeppresenter.agents.subagent import SubAgent
from deeppresenter.tools.filesystem import WorkspaceTools
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.typings import InputRequest


def _write_config(tmp_path: Path) -> DeepPresenterConfig:
    mcp_file = tmp_path / "mcp.json"
    mcp_file.write_text("[]", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
offline_mode: true
mcp_config_file: MCP_FILE
research_agent: &model
  base_url: http://localhost:1/v1
  model: test-model
  api_key: test
design_agent: *model
long_context_model: *model
""".replace("MCP_FILE", str(mcp_file)),
        encoding="utf-8",
    )
    return DeepPresenterConfig.load_from_file(str(config_file))


@pytest.mark.unit
def test_config_loads_without_network(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    assert config.offline_mode is True
    assert config.research_agent.model_name == "test-model"
    assert config.mcp_config_file == str(tmp_path / "mcp.json")


@pytest.mark.unit
def test_copy_file_and_directory_to_workspace(tmp_path: Path) -> None:
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello", encoding="utf-8")
    source_dir = tmp_path / "assets"
    source_dir.mkdir()
    (source_dir / "image.txt").write_text("image", encoding="utf-8")
    workspace = tmp_path / "workspace"

    request = InputRequest(
        instruction="test", attachments=[str(source_file), str(source_dir)]
    )
    request.copy_to_workspace(workspace)

    assert (workspace / "attachments" / "source.txt").read_text() == "hello"
    assert (workspace / "attachments" / "assets" / "image.txt").read_text() == "image"
    assert all(Path(item).is_relative_to(workspace) for item in request.attachments)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_env_runs_sync_and_async_local_tools(tmp_path: Path) -> None:
    env = AgentEnv(tmp_path / "workspace", _write_config(tmp_path))

    def sync_tool(value: int) -> int:
        """Double a value."""
        return value * 2

    async def async_tool(value: int) -> int:
        """Increment a value."""
        return value + 1

    env.register_tool(sync_tool)
    env.register_tool(async_tool)
    sync_result = await env.tool_execute(
        ToolCall(
            id="sync",
            type="function",
            function={"name": "sync_tool", "arguments": '{"value": 2}'},
        )
    )
    async_result = await env.tool_execute(
        ToolCall(
            id="async",
            type="function",
            function={"name": "async_tool", "arguments": '{"value": 2}'},
        )
    )

    assert sync_result.text == "4"
    assert async_result.text == "3"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_env_connects_local_servers_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    async with AgentEnv(workspace, _write_config(tmp_path)) as env:
        WorkspaceTools(workspace).register(env)
        assert "local" in env._server_tools
        assert "write_file" in env._server_tools["local"]
        assert "inspect_slide_element" in env._server_tools["local"]
        assert "patch_slide_element" in env._server_tools["local"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delegate_missing_context_does_not_consume_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = AgentEnv(workspace, _write_config(tmp_path))
    delegate = SubAgent.delegate(_write_config(tmp_path), env, workspace, "en")

    with pytest.raises(FileNotFoundError, match="Context file missing.txt"):
        await delegate("research_01", "research", "missing.txt")
    assert not (workspace / "subagents" / "research_01").exists()

    context = workspace / "context.txt"
    context.write_text("context", encoding="utf-8")

    async def fake_loop(self, task: str, content: str) -> str:
        assert task == "research"
        assert content == "context"
        return "result.md"

    monkeypatch.setattr(SubAgent, "loop", fake_loop)
    monkeypatch.setattr(SubAgent, "save_history", lambda self: None)
    assert await delegate("research_01", "research", "context.txt") == "result.md"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalize_accepts_workspace_alias(tmp_path: Path) -> None:
    env = AgentEnv(tmp_path / "workspace", _write_config(tmp_path))
    env.workspace.mkdir(parents=True)

    def finalize(outcome: str, agent_name: str = "") -> str:
        """Finalize an agent artifact."""
        assert agent_name == "Research"
        assert Path(outcome).exists()
        return outcome

    env.register_tool(finalize)
    agent = object.__new__(Agent)
    agent.name = "Research"
    agent.workspace = env.workspace
    agent.agent_env = env
    agent.chat_history = []
    agent.error_history = []
    artifact = env.workspace / "result.md"
    artifact.write_text("result", encoding="utf-8")
    outcome = await agent.execute(
        [
            ToolCall(
                id="finalize",
                type="function",
                function={
                    "name": "finalize",
                    "arguments": '{"outcome": "/workspace/result.md"}',
                },
            )
        ]
    )
    assert outcome == str(artifact)


@pytest.mark.unit
def test_workspace_tools_are_scoped_and_record_command_output(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path / "workspace")
    tools.write_file("nested/a.txt", "before")
    tools.edit_file("nested/a.txt", "before", "after")

    assert tools.read_file("nested/a.txt") == "after"
    tools.write_file("lines.txt", "zero\none\ntwo\nthree\n")
    assert tools.read_file("lines.txt", offset=1, limit=2) == "one\ntwo\n"
    tools.edit_file(html_file="nested/a.txt", old="after", new="aliased")
    assert tools.read_file("nested/a.txt") == "aliased"
    tools.write_file(
        "slide.html",
        '<html><body><p data-slidex-id="body">Before</p></body></html>',
    )
    patch = json.loads(
        tools.patch_html(
            "slide.html", '[data-slidex-id="body"]', "set_style", "color", "red"
        )
    )
    assert patch["operation"] == "set_style"
    assert 'style="color: red"' in tools.read_file("slide.html")
    tools.write_file("/workspace/legacy.txt", "mapped")
    assert tools.read_file("legacy.txt") == "mapped"
    assert json.loads(tools.list_files(pattern="**/*.txt")) == [
        "legacy.txt",
        "lines.txt",
        "nested/a.txt",
    ]
    result = json.loads(tools.run_command("printf command-ok"))
    assert result == {"exit_code": 0, "stdout": "command-ok", "stderr": ""}
    with pytest.raises(ValueError, match="escapes workspace"):
        tools.read_file("../outside.txt")


@pytest.mark.unit
def test_write_file_autofixes_double_escaped_html(tmp_path: Path) -> None:
    """Guards against a real gpt-4o-mini failure mode: JSON-encoding an HTML
    tool-call payload twice leaves literal ``\\n``/``\\"`` sequences instead of
    real newlines/quotes. Rejecting-and-asking-the-model-to-retry does not
    reliably break this loop in practice -- gpt-4o-mini was observed adding
    *another* layer of escaping on retry instead of removing one (observed
    during 13.7 E2E smoke testing). Since the transformation is deterministic
    and reversible, we auto-correct it instead: the file must end up with
    real newlines/quotes, matching what the model actually meant to write.
    """
    tools = WorkspaceTools(tmp_path / "workspace")
    double_escaped = (
        '<!DOCTYPE html>\\n<html lang=\\"en\\" style=\\"width: 1280px;\\">'
        "\\n<body>\\n</body>\\n</html>"
    )
    tools.write_file("slides/slide_01.html", double_escaped)
    written = tools.read_file("slides/slide_01.html")
    assert written == '<!DOCTYPE html>\n<html lang="en" style="width: 1280px;">\n<body>\n</body>\n</html>'
    assert chr(92) + "n" not in written

    # A normal, correctly-encoded HTML file must still write fine.
    tools.write_file("slides/slide_01.html", "<!DOCTYPE html>\n<html>\n<body></body>\n</html>\n")
    assert "<!DOCTYPE html>" in tools.read_file("slides/slide_01.html")


def test_write_file_rejects_unrecoverable_multi_layer_escaping(tmp_path: Path) -> None:
    """A pathological multi-layer escaping case (observed compounding across
    retries, e.g. gpt-4o-mini re-escaping an already double-escaped string)
    that our bounded auto-unescape cannot cleanly resolve must still be
    rejected rather than silently writing corrupted content.
    """
    tools = WorkspaceTools(tmp_path / "workspace")
    backslash = chr(92)
    # A 64-backslash run before each "n" halves in length on every unescape
    # pass (each pass turns a \\ pair into one \), so it takes 7 passes to
    # fully resolve -- well beyond _MAX_UNESCAPE_PASSES -- and must still be
    # flagged as double-escaped after the pass budget is exhausted.
    run = backslash * 64
    mangled = run + "n" + "<html>" + run + "n" + run + "n" + run + "n"
    with pytest.raises(ValueError, match="double-escaped"):
        tools.write_file("slides/slide_02.html", mangled)
    assert not (tmp_path / "workspace" / "slides" / "slide_02.html").exists()

    # edit_file must also reject a replacement that produces unrecoverable
    # double-escaped output, leaving the original file untouched.
    tools.write_file("slides/slide_01.html", "<!DOCTYPE html>\n<html>\n<body></body>\n</html>\n")
    with pytest.raises(ValueError, match="double-escaped"):
        tools.edit_file(
            "slides/slide_01.html",
            old="<body></body>",
            new=mangled,
        )
    assert "<body></body>" in tools.read_file("slides/slide_01.html")


def test_edit_file_recovers_from_escaped_old_argument(tmp_path: Path) -> None:
    """Guards against a real gpt-4o-mini failure mode: after write_file
    auto-corrects double-escaped HTML on disk, the model's next edit_file
    call sometimes still tracks the file content in escaped form and passes
    a literal ``\\"``-laden ``old`` that can never match the already-clean
    file, causing "found 0" to repeat identically forever (observed during
    13.7 E2E smoke testing). edit_file must retry against an unescaped
    ``old``/``new`` before giving up.
    """
    tools = WorkspaceTools(tmp_path / "workspace")
    tools.write_file(
        "slides/slide_01.html",
        '<p data-slidex-id="overview-1">Pico is great.</p>',
    )
    tools.edit_file(
        "slides/slide_01.html",
        old='<p data-slidex-id=\\"overview-1\\">Pico is great.</p>',
        new='<p data-slidex-id=\\"overview-1\\">Pico is great.</p>\\n    <div></div>',
    )
    assert tools.read_file("slides/slide_01.html") == (
        '<p data-slidex-id="overview-1">Pico is great.</p>\n    <div></div>'
    )

    # An old snippet that genuinely has no match (escaped or not) must still
    # fail loudly rather than silently no-op or match the wrong thing.
    with pytest.raises(ValueError, match="found 0"):
        tools.edit_file(
            "slides/slide_01.html",
            old='<p data-slidex-id=\\"nonexistent\\">',
            new="anything",
        )


@pytest.mark.unit
def test_generate_cli_rejects_invalid_output_before_onboarding(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from deeppresenter.cli import app

    result = CliRunner().invoke(
        app,
        ["generate", "Quarterly review", "--output", str(tmp_path / "deck.pdf")],
    )

    assert result.exit_code == 2
    assert "Output path must end with .pptx" in result.output


@pytest.mark.unit
def test_generate_cli_runs_request_to_output_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from deeppresenter.cli import app
    from deeppresenter.cli import commands

    attachment = tmp_path / "brief.txt"
    attachment.write_text("source material", encoding="utf-8")
    generated = tmp_path / "workspace" / "generated.pptx"
    generated.parent.mkdir()
    generated.write_bytes(b"mock-pptx")
    output = tmp_path / "deliverables" / "review.pptx"
    captured: dict[str, object] = {}

    class FakeAgentLoop:
        def __init__(self, config, session_id, workspace, language) -> None:
            captured["language"] = language
            self.workspace = generated.parent
            self.intermediate_output: dict[str, str] = {}

        async def run(self, request):
            captured["request"] = request
            yield generated

    async def fake_shutdown() -> None:
        captured["shutdown"] = True

    config = SimpleNamespace(
        mcp_config_file=None,
        offline_mode=False,
        multiagent_mode=False,
        context_folding=True,
        heavy_reflect=False,
    )
    monkeypatch.setattr(commands, "is_onboarded", lambda: True)
    monkeypatch.setattr(
        commands.DeepPresenterConfig, "load_from_file", lambda _: config
    )
    monkeypatch.setattr(commands, "uses_local_model", lambda _: False)
    monkeypatch.setattr(commands, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(commands.PlaywrightConverter, "shutdown", fake_shutdown)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "Quarterly review",
            "--output",
            str(output),
            "--file",
            str(attachment),
            "--pages",
            "5-7",
            "--aspect",
            "4:3",
            "--lang",
            "zh",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b"mock-pptx"
    request = captured["request"]
    assert isinstance(request, InputRequest)
    assert request.instruction == "Quarterly review"
    assert request.attachments == [str(attachment.resolve())]
    assert request.num_pages == "5-7"
    assert request.powerpoint_type.value == "4:3"
    assert captured == {
        "language": "zh",
        "request": request,
        "shutdown": True,
    }


@pytest.mark.unit
def test_generate_cli_fails_when_agent_returns_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from deeppresenter.cli import app
    from deeppresenter.cli import commands

    class EmptyAgentLoop:
        def __init__(self, config, session_id, workspace, language) -> None:
            self.workspace = tmp_path / "workspace"
            self.intermediate_output: dict[str, str] = {}

        async def run(self, request):
            if False:
                yield request

    async def fake_shutdown() -> None:
        return None

    config = SimpleNamespace(
        mcp_config_file=None,
        offline_mode=False,
        multiagent_mode=False,
        context_folding=True,
        heavy_reflect=False,
    )
    monkeypatch.setattr(commands, "is_onboarded", lambda: True)
    monkeypatch.setattr(
        commands.DeepPresenterConfig, "load_from_file", lambda _: config
    )
    monkeypatch.setattr(commands, "uses_local_model", lambda _: False)
    monkeypatch.setattr(commands, "AgentLoop", EmptyAgentLoop)
    monkeypatch.setattr(commands.PlaywrightConverter, "shutdown", fake_shutdown)

    result = CliRunner().invoke(
        app,
        ["generate", "Quarterly review", "--output", str(tmp_path / "deck.pptx")],
    )

    assert result.exit_code == 1
    assert "Generation completed without producing a PPTX file" in result.output
    assert "Success!" not in result.output


@pytest.mark.unit
def test_onboard_validation_failure_preserves_existing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from deeppresenter.cli import app
    from deeppresenter.cli import commands

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    mcp_file = config_dir / "mcp.json"
    config_file.write_text("existing: true\n", encoding="utf-8")
    mcp_file.write_text("[]", encoding="utf-8")

    class InvalidConfig:
        async def validate_llms(self) -> None:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(commands, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(commands, "CONFIG_FILE", config_file)
    monkeypatch.setattr(commands, "MCP_FILE", mcp_file)
    monkeypatch.setattr(commands, "is_onboarded", lambda: False)
    monkeypatch.setattr(commands, "check_playwright_browsers", lambda: True)
    monkeypatch.setattr(commands, "check_npm_dependencies", lambda: True)
    monkeypatch.setattr(commands, "check_poppler", lambda: True)
    monkeypatch.setattr(commands, "has_complete_model_config", lambda _: True)
    monkeypatch.setattr(
        commands,
        "prompt_llm_config",
        lambda *args, **kwargs: (
            {
                "base_url": "http://localhost:1/v1",
                "model": "test",
                "api_key": "test",
            }
            if not kwargs.get("optional")
            else None
        ),
    )
    monkeypatch.setattr(commands.Confirm, "ask", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        commands.DeepPresenterConfig, "load_from_file", lambda _: InvalidConfig()
    )
    monkeypatch.setattr(commands, "uses_local_model", lambda _: False)

    result = CliRunner().invoke(app, ["onboard"])

    assert result.exit_code == 1
    assert "Existing configuration was not changed" in result.output
    assert config_file.read_text(encoding="utf-8") == "existing: true\n"
    assert mcp_file.read_text(encoding="utf-8") == "[]"
    assert not (config_dir / ".config.yaml.pending").exists()
    assert not (config_dir / ".mcp.json.pending").exists()

@pytest.mark.unit
def test_stable_id_inspection_and_structured_slide_patch(tmp_path: Path) -> None:
    """Repair patches target stable IDs without brittle source-string matching."""
    tools = WorkspaceTools(tmp_path / "workspace")
    tools.write_file(
        "slides/slide_01.html",
        """<!doctype html><html><body><main class="slide-content" data-slidex-id="content">
        <h1 data-slidex-id="title" style="font-size: 42px">Before</h1>
        <p data-slidex-id="body">Body copy</p></main></body></html>""",
    )

    index = json.loads(tools.inspect_slide_element("slides/slide_01.html"))
    assert [item["element_id"] for item in index["elements"]] == ["body", "content", "title"]
    title = json.loads(tools.inspect_slide_element("slides/slide_01.html", "title"))
    assert title["inline_style"] == {"font-size": "42px"}
    assert title["text"] == "Before"

    patch = json.loads(
        tools.patch_slide_element(
            "slides/slide_01.html",
            "title",
            styles={"font-size": "32px", "line-height": "1.15"},
            text="After",
            attributes={"aria-label": "Slide title"},
        )
    )
    assert patch["element_id"] == "title"
    assert patch["before"]["text"] == "Before"
    assert patch["after"]["text"] == "After"
    assert patch["after"]["inline_style"] == {
        "font-size": "32px",
        "line-height": "1.15",
    }
    assert patch["after"]["attributes"] == {"aria-label": "Slide title"}

    with pytest.raises(ValueError, match="unsupported style"):
        tools.patch_slide_element(
            "slides/slide_01.html", "title", styles={"animation": "spin 1s"}
        )
    with pytest.raises(ValueError, match="No element"):
        tools.inspect_slide_element("slides/slide_01.html", "missing")
    with pytest.raises(ValueError, match="leaf child IDs.*title.*body"):
        tools.patch_slide_element("slides/slide_01.html", "content", text="Would delete children")

@pytest.mark.unit
def test_stable_id_inventory_exposes_actual_repair_targets(tmp_path: Path) -> None:
    """Repair context must expose source IDs instead of asking the model to infer them."""
    from deeppresenter.main import _repair_stable_id_context
    from deeppresenter.slidex.authoring import stable_id_inventory

    slides = tmp_path / "slides"
    slides.mkdir()
    (slides / "slide_01.html").write_text(
        """<html><body><main><h1 data-slidex-id="title">Overview</h1>
        <ul data-slidex-id="capability-list"><li data-slidex-id="capability-a">Search</li>
        <li data-slidex-id="capability-b">Repair</li></ul></main></body></html>""",
        encoding="utf-8",
    )

    inventory = stable_id_inventory(slides / "slide_01.html")
    assert inventory == {
        "slide": "slide_01.html",
        "elements": [
            {"element_id": "title", "tag": "h1", "text": "Overview", "text_patchable": True},
            {"element_id": "capability-list", "tag": "ul", "text": "Search Repair", "text_patchable": False},
            {"element_id": "capability-a", "tag": "li", "text": "Search", "text_patchable": True},
            {"element_id": "capability-b", "tag": "li", "text": "Repair", "text_patchable": True},
        ],
    }
    context = json.loads(_repair_stable_id_context(slides))
    assert context["stable_id_inventory"] == [inventory]


@pytest.mark.unit
def test_repair_stable_id_context_requires_existing_slides(tmp_path: Path) -> None:
    from deeppresenter.main import _repair_stable_id_context

    with pytest.raises(RuntimeError, match="existing HTML slides"):
        _repair_stable_id_context(tmp_path)
