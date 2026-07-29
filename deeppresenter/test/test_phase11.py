"""Phase 11 branding, migration, observability, and compatibility tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from deeppresenter.cli import common
from deeppresenter.utils import constants
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.log import sanitize_log_text
from deeppresenter.utils.typings import ConvertType, InputRequest

pytestmark = pytest.mark.unit


def _model(model: str = "fake") -> dict[str, object]:
    return {
        "base_url": "http://localhost:8000/v1",
        "model": model,
        "api_key": "top-secret",
        "capabilities": {
            "text": True,
            "vision": True,
            "tools": True,
            "structured_output": True,
        },
    }


def test_slidex_is_primary_command_and_brand() -> None:
    cli_source = Path("deeppresenter/cli/__init__.py").read_text(encoding="utf-8")
    assert "Slidex - Source-aware" in cli_source
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "slidex"' in pyproject
    assert 'slidex = "deeppresenter.cli:main"' in pyproject
    assert 'pptagent = "deeppresenter.cli:main"' in pyproject
    assert 'pptagent-mcp = "pptagent.mcp_server:main"' in pyproject


def test_workspace_environment_prefers_slidex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPPRESENTER_WORKSPACE_BASE", "/tmp/legacy")
    monkeypatch.setenv("SLIDEX_WORKSPACE_BASE", "/tmp/slidex")
    value = Path(
        constants.os.getenv(
            "SLIDEX_WORKSPACE_BASE",
            constants.os.getenv("DEEPPRESENTER_WORKSPACE_BASE", "missing"),
        )
    )
    assert value == Path("/tmp/slidex")


def test_legacy_config_is_copied_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "deeppresenter"
    target = tmp_path / "slidex"
    legacy.mkdir()
    payload = {
        "research_agent": _model(),
        "design_agent": _model(),
        "long_context_model": _model(),
    }
    (legacy / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    (legacy / "mcp.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(common, "LEGACY_CONFIG_DIR", legacy)
    monkeypatch.setattr(common, "CONFIG_DIR", target)
    monkeypatch.setattr(common, "CONFIG_FILE", target / "config.yaml")
    monkeypatch.setattr(common, "MCP_FILE", target / "mcp.json")

    assert common.migrate_legacy_config()
    text = common.sanitized_config_text(target / "config.yaml")
    assert "top-secret" not in text
    assert "REDACTED" in text
    assert json.loads((target / "mcp.json").read_text()) == []


def test_config_dump_excludes_api_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "research_agent": _model("research"),
                "design_agent": {
                    **_model("gemini-3.6-flash"),
                    "base_url": "https://aigc.sankuai.com/v1/openai/native",
                    "requests_per_minute": 20,
                },
                "long_context_model": _model("long"),
                "critic_model": _model("critic"),
                "judge_model": {
                    **_model("doubao-seed-2-1-turbo-260628"),
                    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                },
            }
        ),
        encoding="utf-8",
    )
    config = DeepPresenterConfig.load_from_file(str(path))
    assert "top-secret" not in config.model_dump_json()
    assert config.critic_model is not config.design_agent
    assert config.judge_model is not config.design_agent
    assert config.judge_model.model_name == "doubao-seed-2-1-turbo-260628"


def test_legacy_convert_type_remains_explicit() -> None:
    request = InputRequest(instruction="deck", convert_type="pptagent")
    assert request.convert_type is ConvertType.PPTAGENT
    assert request.convert_type.is_legacy_template
    assert InputRequest(instruction="deck").convert_type is ConvertType.SLIDEX


def test_logs_redact_secrets_and_base64() -> None:
    text = sanitize_log_text(
        "api_key=secret data:image/png;base64,QUJDREVGRw== attachment.pdf"
    )
    assert "secret" not in text
    assert "QUJD" not in text
    assert "attachment.pdf" in text


def test_frozen_example_assigns_friday_generation_and_ark_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_API_KEY", "friday-test")
    monkeypatch.setenv("ARK_JUDGE_API_KEY", "ark-test")
    config = DeepPresenterConfig.load_from_file("deeppresenter/config.yaml.example")
    assert config.design_agent.model_name == "gpt-4o-mini"
    assert config.design_agent._endpoints[0].base_url == (
        "https://aigc.sankuai.com/v1/openai/native"
    )
    assert config.design_agent.requests_per_minute == 200
    assert config.critic_model.model_name == "gpt-4o-mini"
    assert config.judge_model.model_name == "doubao-seed-2-1-turbo-260628"
    assert config.judge_model._endpoints[0].base_url == (
        "https://ark.cn-beijing.volces.com/api/v3"
    )


def test_model_role_policy_rejects_generator_or_judge_drift(tmp_path: Path) -> None:
    # The policy is a self-consistency check against the config's own frozen
    # snapshot fields, not a hardcoded model name: switching the generation
    # or judge model only requires updating the endpoint config and its
    # matching `frozen_*` fields together.
    base = {
        "slidex": {
            "enforce_model_role_policy": True,
            "frozen_generation_model": "gemini-3.6-flash",
            "frozen_generation_base_url": "https://aigc.sankuai.com/v1/openai/native",
            "frozen_generation_requests_per_minute": 20,
            "frozen_judge_model": "doubao-seed-2-1-turbo-260628",
            "frozen_judge_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        },
        "research_agent": _model("research"),
        "long_context_model": _model("long"),
        "design_agent": {
            **_model("gemini-3.6-flash"),
            "base_url": "https://aigc.sankuai.com/v1/openai/native",
            "requests_per_minute": 20,
        },
        "judge_model": {
            **_model("doubao-seed-2-1-turbo-260628"),
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        },
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    DeepPresenterConfig.load_from_file(str(path))

    base["design_agent"]["model"] = "other-generator"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="drifted from the frozen generation snapshot"):
        DeepPresenterConfig.load_from_file(str(path))

    base["design_agent"]["model"] = "gemini-3.6-flash"
    base["judge_model"]["model"] = "other-judge"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="drifted from the frozen judge snapshot"):
        DeepPresenterConfig.load_from_file(str(path))


def test_model_role_policy_supports_switching_frozen_snapshot(
    tmp_path: Path,
) -> None:
    """Switching to a new model is a config-only change: update the endpoint
    and its matching `frozen_*` fields together, with no code edit."""
    base = {
        "slidex": {
            "enforce_model_role_policy": True,
            "frozen_generation_model": "gpt-4o-mini",
            "frozen_generation_base_url": "https://example.com/v1",
            "frozen_generation_requests_per_minute": 200,
            "frozen_judge_model": "doubao-seed-2-1-turbo-260628",
            "frozen_judge_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        },
        "research_agent": _model("research"),
        "long_context_model": _model("long"),
        "design_agent": {
            **_model("gpt-4o-mini"),
            "base_url": "https://example.com/v1",
            "requests_per_minute": 200,
        },
        "judge_model": {
            **_model("doubao-seed-2-1-turbo-260628"),
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        },
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    DeepPresenterConfig.load_from_file(str(path))


def test_model_role_policy_requires_frozen_snapshot_fields(tmp_path: Path) -> None:
    base = {
        "slidex": {"enforce_model_role_policy": True},
        "research_agent": _model("research"),
        "long_context_model": _model("long"),
        "design_agent": {
            **_model("gemini-3.6-flash"),
            "base_url": "https://aigc.sankuai.com/v1/openai/native",
            "requests_per_minute": 20,
        },
        "judge_model": {
            **_model("doubao-seed-2-1-turbo-260628"),
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        },
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen generation snapshot"):
        DeepPresenterConfig.load_from_file(str(path))
