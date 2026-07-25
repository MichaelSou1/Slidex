"""External dataset acquisition with explicit revision, license, and digest checks."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from .models import SourceRecord


def download_source(
    source: SourceRecord, cache_dir: Path, allowed_licenses: set[str]
) -> Path:
    """Download one declared source and reject mutable or unlicensed bytes."""
    if source.license.lower() not in {
        license_name.lower() for license_name in allowed_licenses
    }:
        raise ValueError(f"unapproved or missing license: {source.license}")
    if not source.revision.strip():
        raise ValueError("dataset revision must be pinned")
    target = cache_dir / source.source_id / source.revision / Path(source.url).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(source.url, timeout=60) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != source.sha256:
        target.unlink(missing_ok=True)
        raise ValueError(f"dataset hash mismatch: {source.source_id}")
    return target


def validate_slideaudit_crosswalk(crosswalk: dict[str, list[str]]) -> None:
    """Keep one-to-many labels explicit instead of coercing a single class."""
    for source_label, target_labels in crosswalk.items():
        if (
            not source_label
            or not target_labels
            or len(target_labels) != len(set(target_labels))
        ):
            raise ValueError(f"invalid taxonomy crosswalk entry: {source_label}")
