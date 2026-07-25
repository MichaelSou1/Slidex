"""Content-addressed cache for immutable IR, render, inspection, and neural outputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ContentCache:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(namespace: str, *values: Any) -> str:
        digest = hashlib.sha256(namespace.encode())
        for value in values:
            if isinstance(value, BaseModel):
                value = value.model_dump(mode="json", exclude_none=False)
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
            digest.update(b"\0")
            digest.update(encoded)
        return digest.hexdigest()

    def get_json(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self._path(namespace, key, ".json")
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def get_bytes(self, namespace: str, key: str, suffix: str) -> bytes | None:
        path = self._path(namespace, key, suffix)
        return path.read_bytes() if path.is_file() else None

    def put_bytes(self, namespace: str, key: str, suffix: str, value: bytes) -> Path:
        path = self._path(namespace, key, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(value)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    def put_json(
        self, namespace: str, key: str, value: BaseModel | dict[str, Any]
    ) -> Path:
        payload = (
            value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        )
        path = self._path(namespace, key, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    def _path(self, namespace: str, key: str, suffix: str) -> Path:
        if (
            not namespace.replace("_", "").replace("-", "").isalnum()
            or len(key) != 64
            or not suffix.startswith(".")
            or "/" in suffix
        ):
            raise ValueError("invalid cache namespace, key, or suffix")
        return self.root / namespace / key[:2] / f"{key}{suffix}"
