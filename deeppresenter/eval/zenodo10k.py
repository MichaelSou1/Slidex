"""Reproducible acquisition and freezing for the Zenodo10K PPTX corpus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from .io import content_hash, file_hash, write_immutable
from .models import SourceRecord
from .prepare import template_fingerprint

ZENODO10K_REPO_ID = "Forceless/Zenodo10K"
ZENODO10K_REVISION = "e59bf3ec11f7518a6c84dc145d83c0675d412522"
ZENODO10K_METADATA_PATH = "data/pptx-00000-of-00001.parquet"
ZENODO10K_LICENSE_TEXT = (
    "Creative Commons Attribution 4.0 International (CC BY 4.0); "
    "https://creativecommons.org/licenses/by/4.0/legalcode"
)
ZENODO10K_LICENSE_TEXT_SHA256 = hashlib.sha256(
    ZENODO10K_LICENSE_TEXT.encode()
).hexdigest()
DEFAULT_ALLOWED_LICENSES = ("cc-by-4.0",)


def _endpoint(endpoint: str | None) -> str:
    return (endpoint or os.getenv("HF_ENDPOINT") or "https://hf-mirror.com").rstrip("/")


def _row_key(row: dict[str, Any], seed: int) -> str:
    identity = f"{seed}\0{row['checksum']}\0{row['doi']}\0{row['filename']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def _select_stratified(
    rows: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_year[str(row["created"])[:4]].append(row)
    for items in by_year.values():
        items.sort(key=lambda item: _row_key(item, seed))
    selected: list[dict[str, Any]] = []
    years = sorted(by_year)
    position = 0
    while len(selected) < sample_size:
        progressed = False
        for year in years:
            items = by_year[year]
            if position < len(items):
                selected.append(items[position])
                progressed = True
                if len(selected) == sample_size:
                    break
        if not progressed:
            break
        position += 1
    if len(selected) != sample_size:
        raise ValueError(
            f"only {len(selected)} eligible records for requested sample of {sample_size}"
        )
    return selected


def _repository_paths(endpoint: str, revision: str) -> dict[str, str]:
    info = HfApi(endpoint=endpoint).dataset_info(
        ZENODO10K_REPO_ID, revision=revision, files_metadata=True
    )
    if info.sha != revision:
        raise ValueError(
            f"dataset revision mismatch: expected {revision}, received {info.sha}"
        )
    paths: dict[str, str] = {}
    for sibling in info.siblings:
        path = sibling.rfilename
        if not path.lower().endswith(".pptx"):
            continue
        digest = Path(path).name.split("-", 1)[0]
        if len(digest) == 32:
            paths[digest] = path
    return paths


def _validate_pptx(path: Path, expected_md5: str) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    if md5.hexdigest() != expected_md5:
        raise ValueError(f"MD5 mismatch for {path.name}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt PPTX member {bad}: {path.name}")
        if not any(name.startswith("ppt/slides/slide") for name in archive.namelist()):
            raise ValueError(f"PPTX contains no slides: {path.name}")
    return sha256.hexdigest()


def freeze_zenodo10k_sample(
    cache_root: Path,
    *,
    revision: str = ZENODO10K_REVISION,
    sample_size: int = 60,
    seed: int = 13,
    allowed_licenses: tuple[str, ...] = DEFAULT_ALLOWED_LICENSES,
    min_bytes: int = 100_000,
    max_bytes: int = 8 * 1024 * 1024,
    endpoint: str | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Download and freeze a deterministic, license-filtered Zenodo10K sample."""
    endpoint = _endpoint(endpoint)
    output_dir = cache_root / "zenodo10k" / revision
    frozen_path = output_dir / "frozen-sources.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        policy = frozen.get("selection_policy", {})
        requested = {
            "revision": revision,
            "sample_size": sample_size,
            "seed": seed,
            "allowed_licenses": sorted(item.lower() for item in allowed_licenses),
            "min_bytes": min_bytes,
            "max_bytes": max_bytes,
        }
        mismatched = {
            key: (policy.get(key), value)
            for key, value in requested.items()
            if policy.get(key) != value
        }
        if mismatched:
            raise FileExistsError(
                "frozen Zenodo10K sample uses different selection inputs: "
                f"{mismatched}; use a different cache root"
            )
        for source in frozen["sources"]:
            path = output_dir / source["local_path"]
            if not path.exists() or file_hash(path) != source["sha256"]:
                raise ValueError(
                    f"frozen Zenodo10K artifact mismatch: {source['source_id']}"
                )
        return frozen

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_download = Path(
        hf_hub_download(
            ZENODO10K_REPO_ID,
            ZENODO10K_METADATA_PATH,
            repo_type="dataset",
            revision=revision,
            endpoint=endpoint,
        )
    )
    metadata_path = output_dir / "metadata.parquet"
    if not metadata_path.exists():
        try:
            os.link(metadata_download, metadata_path)
        except OSError:
            shutil.copyfile(metadata_download, metadata_path)
    metadata_sha256 = file_hash(metadata_path)
    frame = pd.read_parquet(metadata_path)
    required = {
        "filename",
        "size",
        "url",
        "license",
        "title",
        "created",
        "updated",
        "doi",
        "checksum",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Zenodo10K metadata columns missing: {sorted(required - set(frame.columns))}"
        )

    licenses = {item.lower() for item in allowed_licenses}
    eligible_frame = frame[
        frame["license"].str.lower().isin(licenses)
        & frame["size"].between(min_bytes, max_bytes, inclusive="both")
        & frame["checksum"].str.match(r"^md5:[0-9a-f]{32}$")
    ]
    eligible = eligible_frame[list(required)].to_dict("records")
    ranked = _select_stratified(eligible, len(eligible), seed)
    repo_paths = _repository_paths(endpoint, revision)

    downloads = output_dir / "pptx"
    downloads.mkdir(exist_ok=True)

    def download(row: dict[str, Any]) -> tuple[dict[str, Any], Path, str, str]:
        expected_md5 = str(row["checksum"]).removeprefix("md5:")
        repo_path = repo_paths.get(expected_md5)
        if not repo_path:
            raise ValueError(f"PPTX missing from pinned repository: {expected_md5}")
        cached = Path(
            hf_hub_download(
                ZENODO10K_REPO_ID,
                repo_path,
                repo_type="dataset",
                revision=revision,
                endpoint=endpoint,
            )
        )
        target = downloads / f"{expected_md5}.pptx"
        if not target.exists():
            try:
                os.link(cached, target)
            except OSError:
                shutil.copyfile(cached, target)
        try:
            sha256 = _validate_pptx(target, expected_md5)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return row, target, sha256, repo_path

    acquired: list[tuple[dict[str, Any], Path, str, str]] = []
    rejected_duplicates: list[dict[str, str]] = []
    rejected_invalid: list[dict[str, str]] = []
    seen_templates: set[str] = set()
    cursor = 0
    while len(acquired) < sample_size and cursor < len(ranked):
        batch = ranked[cursor : cursor + workers]
        cursor += len(batch)
        completed: list[tuple[dict[str, Any], Path, str, str]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download, row): row for row in batch}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    rejected_invalid.append(
                        {"checksum": str(row["checksum"]), "reason": str(error)}
                    )
        for result in sorted(completed, key=lambda item: _row_key(item[0], seed)):
            row, target, _, _ = result
            fingerprint = template_fingerprint(target)
            if fingerprint in seen_templates:
                rejected_duplicates.append(
                    {
                        "checksum": str(row["checksum"]),
                        "reason": "duplicate template fingerprint",
                        "template_fingerprint": fingerprint,
                    }
                )
                target.unlink(missing_ok=True)
                continue
            seen_templates.add(fingerprint)
            acquired.append(result)
            if len(acquired) == sample_size:
                for unused in completed:
                    if unused not in acquired:
                        unused[1].unlink(missing_ok=True)
                break
    if len(acquired) != sample_size:
        raise ValueError(
            f"only {len(acquired)} unique templates for requested sample of {sample_size}"
        )

    frozen_at = datetime.now(UTC)
    records: list[SourceRecord] = []
    selection: list[dict[str, Any]] = []
    for row, target, sha256, repo_path in sorted(acquired, key=lambda item: item[3]):
        md5 = str(row["checksum"]).removeprefix("md5:")
        source = SourceRecord(
            source_id=f"zenodo10k-{md5}",
            dataset_id=ZENODO10K_REPO_ID,
            url=str(row["url"]),
            license=str(row["license"]),
            license_text_sha256=ZENODO10K_LICENSE_TEXT_SHA256,
            revision=revision,
            upstream_commit=revision,
            sha256=sha256,
            acquired_at=frozen_at,
            local_path=target.relative_to(output_dir).as_posix(),
            redistributable=True,
            citation="PPTAgent, arXiv:2501.03936",
        )
        records.append(source)
        selection.append(
            {
                "source_id": source.source_id,
                "repo_path": repo_path,
                "original_filename": row["filename"],
                "title": row["title"],
                "doi": row["doi"],
                "created": row["created"],
                "updated": row["updated"],
                "declared_size": int(row["size"]),
                "md5": md5,
                "sha256": sha256,
                "selection_key": _row_key(row, seed),
                "template_fingerprint": template_fingerprint(target),
            }
        )

    policy = {
        "revision": revision,
        "sample_size": sample_size,
        "seed": seed,
        "allowed_licenses": sorted(licenses),
        "min_bytes": min_bytes,
        "max_bytes": max_bytes,
        "eligible_count": len(eligible),
        "examined_count": cursor,
        "duplicate_template_count": len(rejected_duplicates),
        "invalid_pptx_count": len(rejected_invalid),
        "strategy": "round-robin by creation year; SHA-256 rank within year; first valid unique templates",
    }
    frozen: dict[str, Any] = {
        "dataset_id": ZENODO10K_REPO_ID,
        "revision": revision,
        "metadata_path": metadata_path.relative_to(output_dir).as_posix(),
        "metadata_sha256": metadata_sha256,
        "license_text": ZENODO10K_LICENSE_TEXT,
        "license_text_sha256": ZENODO10K_LICENSE_TEXT_SHA256,
        "selection_policy": policy,
        "selection_policy_hash": content_hash(policy),
        "frozen_at": frozen_at.isoformat(),
        "sources": [record.model_dump(mode="json") for record in records],
        "selection": selection,
        "rejected_duplicates": rejected_duplicates,
        "rejected_invalid": rejected_invalid,
    }
    frozen["freeze_hash"] = content_hash(frozen)
    write_immutable(frozen_path, frozen)
    return frozen
