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
                "design_agent": _model("design"),
                "long_context_model": _model("long"),
                "critic_model": _model("critic"),
            }
        ),
        encoding="utf-8",
    )
    config = DeepPresenterConfig.load_from_file(str(path))
    assert "top-secret" not in config.model_dump_json()
    assert config.critic_model is not config.design_agent


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
