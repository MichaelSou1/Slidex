"""Atomic, content-addressed persistence for Slidex episode artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deeppresenter.slidex.models import (
    ArtifactManifest,
    ArtifactReference,
    EpisodeManifest,
    Provenance,
    SlideArtifact,
)


class ArtifactStore:
    """Persist immutable artifacts under isolated episode workspaces."""

    def __init__(
        self,
        root: Path,
        max_workspace_bytes: int = 2 * 1024**3,
        max_artifacts: int = 1000,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.max_workspace_bytes = max_workspace_bytes
        self.max_artifacts = max_artifacts
        self.root.mkdir(parents=True, exist_ok=True)

    def create_episode(
        self,
        episode_id: str | None = None,
        versions: dict[str, str] | None = None,
    ) -> EpisodeManifest:
        episode_id = episode_id or str(uuid.uuid4())
        workspace = self.root / episode_id
        workspace.mkdir(mode=0o700, parents=False, exist_ok=False)
        (workspace / "artifacts").mkdir()
        manifest = EpisodeManifest(
            episode_id=episode_id,
            workspace_uri=workspace.as_uri(),
            versions=versions or {},
        )
        self._atomic_json(workspace / "episode.json", manifest.model_dump(mode="json"))
        return manifest

    def write_artifact(
        self,
        episode_id: str,
        files: dict[str, Path | bytes | str],
        provenance: Provenance,
        slide_artifact: SlideArtifact | None = None,
    ) -> ArtifactManifest:
        workspace = self._episode_path(episode_id)
        artifacts_dir = workspace / "artifacts"
        if len(list(artifacts_dir.iterdir())) >= self.max_artifacts:
            raise ValueError("artifact quota exceeded")

        materialized = {name: self._read_content(value) for name, value in files.items()}
        content_hash = self._content_hash(materialized)
        artifact_id = f"{uuid.uuid4().hex[:12]}-{content_hash[:16]}"
        final_dir = artifacts_dir / artifact_id
        temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-", dir=artifacts_dir))
        try:
            references: dict[str, ArtifactReference] = {}
            for name, content in materialized.items():
                destination = self._safe_child(temp_dir, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                references[name] = ArtifactReference(
                    uri=f"artifact://{episode_id}/{artifact_id}/{name}",
                    sha256=sha256_bytes(content),
                    size_bytes=len(content),
                )
            if slide_artifact is not None:
                payload = slide_artifact.model_copy(update={"artifact_id": artifact_id})
                content = payload.model_dump_json(indent=2).encode()
                (temp_dir / "slide_artifact.json").write_bytes(content)
                references["slide_artifact.json"] = ArtifactReference(
                    uri=f"artifact://{episode_id}/{artifact_id}/slide_artifact.json",
                    sha256=sha256_bytes(content),
                    media_type="application/json",
                    size_bytes=len(content),
                )
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                files=references,
                provenance=provenance,
            )
            self._atomic_json(temp_dir / "manifest.json", manifest.model_dump(mode="json"))
            self._enforce_size_quota(workspace)
            os.rename(temp_dir, final_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        self._append_artifact(workspace, artifact_id)
        return manifest

    def verify_artifact(self, episode_id: str, artifact_id: str) -> bool:
        artifact_dir = self._episode_path(episode_id) / "artifacts" / artifact_id
        manifest = ArtifactManifest.model_validate_json(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return all(
            (path := artifact_dir / name).is_file()
            and sha256_file(path) == reference.sha256
            for name, reference in manifest.files.items()
        )

    def cleanup(self, max_age_seconds: int) -> list[str]:
        """Delete inactive episode workspaces older than the requested age."""
        now = datetime.now(UTC).timestamp()
        removed: list[str] = []
        for workspace in self.root.iterdir():
            manifest_path = workspace / "episode.json"
            if not manifest_path.is_file():
                continue
            manifest = EpisodeManifest.model_validate_json(manifest_path.read_text())
            age = now - manifest.updated_at.timestamp()
            if manifest.status != "active" and age > max_age_seconds:
                shutil.rmtree(workspace)
                removed.append(manifest.episode_id)
        return removed

    def _episode_path(self, episode_id: str) -> Path:
        workspace = self._safe_child(self.root, episode_id)
        if not workspace.is_dir():
            raise FileNotFoundError(f"episode does not exist: {episode_id}")
        return workspace

    def _append_artifact(self, workspace: Path, artifact_id: str) -> None:
        path = workspace / "episode.json"
        manifest = EpisodeManifest.model_validate_json(path.read_text(encoding="utf-8"))
        manifest.artifact_ids.append(artifact_id)
        manifest.updated_at = datetime.now(UTC)
        self._atomic_json(path, manifest.model_dump(mode="json"))

    def _enforce_size_quota(self, workspace: Path) -> None:
        if self._directory_size(workspace) > self.max_workspace_bytes:
            raise ValueError("workspace size quota exceeded")

    @staticmethod
    def _read_content(value: Path | bytes | str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, Path):
            return value.read_bytes()
        return value.encode()

    @staticmethod
    def _content_hash(files: dict[str, bytes]) -> str:
        digest = hashlib.sha256()
        for name in sorted(files):
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(files[name])
        return digest.hexdigest()

    @staticmethod
    def _safe_child(parent: Path, child: str) -> Path:
        path = (parent / child).resolve()
        if not path.is_relative_to(parent.resolve()):
            raise ValueError(f"path escapes artifact store: {child}")
        return path

    @staticmethod
    def _directory_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
