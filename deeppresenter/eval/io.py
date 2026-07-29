"""Canonical hashing, immutable records, and environment capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SECRET_KEYS = {"api_key", "authorization", "token", "secret", "password"}


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "REDACTED" if key.lower() in _SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def write_immutable(path: Path, value: BaseModel | dict[str, Any]) -> str:
    payload = redact(
        value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    )
    digest = content_hash(payload)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"immutable record already exists: {path}")
    path.write_text(text, encoding="utf-8")
    return digest


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _version(command: list[str], aliases: tuple[str, ...] = ()) -> str:
    """Return the first line of ``command --version`` output.

    ``aliases`` lets callers probe alternate executable names (for example
    LibreOffice ships as ``soffice`` on most platforms, not ``libreoffice``).
    """
    executable = shutil.which(command[0])
    for alias in aliases:
        if executable:
            break
        executable = shutil.which(alias)
    if not executable:
        return "unavailable"
    result = subprocess.run(
        [executable, *command[1:]], capture_output=True, text=True, check=False
    )
    return (result.stdout or result.stderr).splitlines()[0].strip() or "unknown"


def capture_environment() -> dict[str, str]:
    """Capture replay-relevant versions without paths or credentials."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "node": _version(["node", "--version"]),
        "libreoffice": _version(["libreoffice", "--version"], aliases=("soffice",)),
        "poppler": _version(["pdftoppm", "-v"]),
        "chromium": os.getenv("PLAYWRIGHT_CHROMIUM_VERSION", "managed-by-playwright"),
        "captured_at": datetime.now(UTC).isoformat(),
    }
