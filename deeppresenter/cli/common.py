import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from deeppresenter import __version__ as version
from deeppresenter.utils.constants import PACKAGE_DIR

console = Console()
CONFIG_DIR = Path.home() / ".config" / "slidex"
LEGACY_CONFIG_DIR = Path.home() / ".config" / "deeppresenter"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
MCP_FILE = CONFIG_DIR / "mcp.json"
CACHE_DIR = Path.home() / ".cache" / "slidex"
LEGACY_CACHE_DIR = Path.home() / ".cache" / "deeppresenter"

LOCAL_MODEL = "Forceless/DeepPresenter-9B-GGUF:q4_K_M"
LOCAL_LID_MODEL = "Forceless/fasttext-language-id"
LOCAL_BASE_URL = "http://127.0.0.1:7811/v1"
REQUIRED_LLM_KEYS = ["research_agent", "design_agent", "long_context_model"]
_SECRET_KEYS = {"api_key", "authorization", "token", "access_token", "secret"}


def migrate_legacy_config() -> bool:
    """Copy legacy user configuration into the Slidex directory once."""
    if CONFIG_FILE.exists() and MCP_FILE.exists():
        return False
    legacy_config = LEGACY_CONFIG_DIR / "config.yaml"
    legacy_mcp = LEGACY_CONFIG_DIR / "mcp.json"
    if not legacy_config.exists() or not legacy_mcp.exists():
        return False
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    for source, destination in ((legacy_config, CONFIG_FILE), (legacy_mcp, MCP_FILE)):
        if not destination.exists():
            shutil.copy2(source, destination)
    console.print(
        f"[yellow]Migrated legacy configuration from {LEGACY_CONFIG_DIR} to {CONFIG_DIR}.[/yellow]"
    )
    return True


def redact_secrets(value: Any) -> Any:
    """Recursively redact credentials while retaining useful config structure."""
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if key.lower() in _SECRET_KEYS else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def sanitized_config_text(path: Path) -> str:
    """Serialize YAML or JSON configuration without exposing credentials."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)
    redacted = redact_secrets(payload)
    if path.suffix == ".json":
        return json.dumps(redacted, indent=2, ensure_ascii=False)
    return yaml.safe_dump(redacted, sort_keys=False, allow_unicode=True).rstrip()


def format_command(cmd: list[str]) -> str:
    """Format command for display."""
    return shlex.join(cmd)


def run_streaming_command(
    cmd: list[str],
    *,
    success_message: str | None = None,
    failure_message: str | None = None,
) -> bool:
    """Run command and stream output to console."""
    console.print(f"[dim]$ {format_command(cmd)}[/dim]")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to start command: {e}")
        return False

    if process.stdout is not None:
        for line in process.stdout:
            console.print(line.rstrip())

    if process.wait() == 0:
        if success_message:
            console.print(success_message)
        return True

    if failure_message:
        console.print(failure_message)
    return False


__all__ = [
    "CACHE_DIR",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "LEGACY_CACHE_DIR",
    "LEGACY_CONFIG_DIR",
    "LOCAL_BASE_URL",
    "LOCAL_LID_MODEL",
    "LOCAL_MODEL",
    "MCP_FILE",
    "PACKAGE_DIR",
    "REQUIRED_LLM_KEYS",
    "console",
    "format_command",
    "migrate_legacy_config",
    "redact_secrets",
    "run_streaming_command",
    "sanitized_config_text",
    "version",
]
