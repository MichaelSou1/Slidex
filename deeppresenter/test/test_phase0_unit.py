import json
from pathlib import Path

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ToolCall,
)

from deeppresenter.agents.agent import Agent
from deeppresenter.agents.env import AgentEnv
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
    tools.write_file("/workspace/legacy.txt", "mapped")
    assert tools.read_file("legacy.txt") == "mapped"
    assert json.loads(tools.list_files(pattern="**/*.txt")) == [
        "legacy.txt",
        "nested/a.txt",
    ]
    result = json.loads(tools.run_command("printf command-ok"))
    assert result == {"exit_code": 0, "stdout": "command-ok", "stderr": ""}
    with pytest.raises(ValueError, match="escapes workspace"):
        tools.read_file("../outside.txt")
