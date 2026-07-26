"""Reproducible acquisition and G1-G7/S1-S6 crosswalk for the SlideAudit corpus.

SlideAudit (Zhang et al. 2025, UIST'25, arXiv:2508.03630) ships 2,400 rendered
slide images with crowdsourced design-deficiency annotations. It has no native
IR, so every case built from it is evaluated under the ``image_only`` evidence
condition and is kept out of the native-IR intrinsic pool (Sec. 13.4).
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .datasets import apply_slideaudit_crosswalk, validate_slideaudit_crosswalk
from .io import content_hash, file_hash, write_immutable
from .models import (
    BenchmarkManifest,
    CrosswalkEntry,
    EvaluationCase,
    LineageRecord,
    PreparationRecord,
    SourceRecord,
    Split,
)

SLIDEAUDIT_REPO_URL = "https://github.com/zhuohaouw/SlideAudit.git"
SLIDEAUDIT_DATASET_ID = "zhuohaouw/SlideAudit"
SLIDEAUDIT_LICENSE_TEXT = (
    "Creative Commons Attribution 4.0 International (CC BY 4.0); "
    "https://creativecommons.org/licenses/by/4.0/"
)
SLIDEAUDIT_LICENSE_TEXT_SHA256 = hashlib.sha256(
    SLIDEAUDIT_LICENSE_TEXT.encode()
).hexdigest()
DEFAULT_ALLOWED_LICENSES = ("cc-by-4.0",)
SLIDEAUDIT_TAXONOMY_VERSION = "slideaudit-crosswalk-v1"
SLIDEAUDIT_SLIDE_COUNT = 2400

# Frozen, hand-reviewed crosswalk from the 19 SlideAudit design-deficiency
# labels to the Slidex G1-G7/S1-S6 taxonomy. Mappings are deliberately
# conservative: a source label is only mapped when its operational
# definition (Table 1, slide-examiner.pdf) matches: category descriptions of
# perceptual quality with no geometry/typography counterpart (e.g. "Poor
# Text Hierarchy") are left unmapped rather than forced onto a nearby class.
SLIDEAUDIT_CROSSWALK: tuple[CrosswalkEntry, ...] = (
    CrosswalkEntry(
        source_label="Content Overflow/Cut-off",
        target_labels=["G1", "G7"],
        rationale=(
            "Content exceeding slide/box bounds matches both declared text "
            "overflow (G1) and render-only containment overflow (G7); the "
            "image-only condition cannot distinguish declared-box legality."
        ),
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Occluded Content",
        target_labels=["G2"],
        rationale="Occlusion is the visible symptom of intersecting declared element boxes.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Content Alignment Issues",
        target_labels=["G3"],
        rationale="Direct operational match with alignment-grid departure.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Cluttered Layout",
        target_labels=["S4"],
        rationale="Clutter is the qualitative expression of over-packed slide density.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Unbalanced Space Distribution",
        target_labels=["S4"],
        rationale="Skewed whitespace distribution is treated as a density-violation variant.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Poor Visual Hierarchy",
        target_labels=[],
        rationale="No operational G/S counterpart; hierarchy quality is not defined geometrically.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Improper Font Sizing",
        target_labels=["G4"],
        rationale="Direct operational match with font-size-inconsistency.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Excessive Text Volume",
        target_labels=["S4"],
        rationale="Text volume overload is a density-violation instance.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Illegible Typeface Selection or Usage",
        target_labels=[],
        rationale="Typeface legibility judgment has no G/S counterpart in the frozen taxonomy.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Improper Text Styling",
        target_labels=[],
        rationale="Styling quality (weight/emphasis misuse) has no G/S counterpart in the frozen taxonomy.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Improper Line/Character Spacing",
        target_labels=[],
        rationale="Spacing quality is not equivalent to grid alignment (G3) and has no other match.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Poor Text Hierarchy",
        target_labels=[],
        rationale="Text hierarchy quality has no G/S counterpart in the frozen taxonomy.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Inappropriate or Mismatched Color Combinations",
        target_labels=["G5"],
        rationale="Mismatched color combinations are an off-palette symptom under brand-color violation.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Excessive or Inconsistent Color Usage",
        target_labels=["G5"],
        rationale="Inconsistent color usage is an off-palette symptom under brand-color violation.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Insufficient Color Contrast for Readability",
        target_labels=[],
        rationale=(
            "Readability contrast is a distinct construct from CIELAB "
            "palette-distance brand-color violation; not force-mapped to G5."
        ),
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Irrelevant Visual Content",
        target_labels=["S6"],
        rationale="Irrelevant imagery is a weak form of image/text contradiction.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Improper Image Sizing",
        target_labels=[],
        rationale="Image sizing quality has no G/S counterpart distinct from geometry classes already anchored on text/box declarations.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Inconsistent Visual Style Usage",
        target_labels=[],
        rationale="Cross-slide visual style consistency has no G/S counterpart in the frozen taxonomy.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
    CrosswalkEntry(
        source_label="Poor Image Quality/Editing",
        target_labels=[],
        rationale="Image editing/compression quality has no G/S counterpart in the frozen taxonomy.",
        evidence_condition="image_only",
        version=SLIDEAUDIT_TAXONOMY_VERSION,
        reviewed=True,
    ),
)

validate_slideaudit_crosswalk(list(SLIDEAUDIT_CROSSWALK))


def resolve_revision(repo_dir: Path) -> str:
    """Return the pinned commit SHA of a local SlideAudit clone."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _slide_ids(repo_dir: Path) -> list[str]:
    images_dir = repo_dir / "data" / "images"
    ids = sorted(path.stem for path in images_dir.glob("slide_*.png"))
    if not ids:
        raise ValueError(f"no SlideAudit slide images found under {images_dir}")
    return ids


def _read_metadata(repo_dir: Path) -> dict[str, dict[str, str]]:
    """Read metadata.csv, keyed by the ``slide_XXXX`` id used elsewhere.

    The upstream CSV uses a bare zero-padded ``id`` column (e.g. ``0001``)
    rather than the ``slide_0001`` stem used by image/annotation filenames.
    """
    metadata_path = repo_dir / "data" / "metadata.csv"
    rows: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_id = row.get("id", "").strip()
            if raw_id:
                rows[f"slide_{raw_id}"] = row
    return rows


def freeze_slideaudit_sample(
    output_dir: Path,
    repo_dir: Path,
    *,
    license: str = "cc-by-4.0",
    allowed_licenses: Iterable[str] = DEFAULT_ALLOWED_LICENSES,
) -> dict[str, Any]:
    """Copy, hash, and freeze the full SlideAudit corpus from a local clone.

    ``repo_dir`` must be a clone of ``SLIDEAUDIT_REPO_URL``; this function
    never fetches network data itself so the acquisition step stays explicit
    and auditable, matching the Zenodo10K acquisition contract.
    """
    if license.lower() not in {item.lower() for item in allowed_licenses}:
        raise ValueError(f"unapproved or missing license: {license}")
    revision = resolve_revision(repo_dir)
    freeze_path = output_dir / "manifests" / f"slideaudit-{revision}-freeze.json"
    if freeze_path.exists():
        cached = json.loads(freeze_path.read_text(encoding="utf-8"))
        if cached.get("revision") == revision and cached.get("license_text_sha256") == (
            SLIDEAUDIT_LICENSE_TEXT_SHA256
        ):
            return cached
        raise FileExistsError(
            f"SlideAudit freeze already exists with different inputs: {freeze_path}"
        )
    frozen_at = datetime.now(UTC)
    dataset_dir = output_dir / "sources" / "slideaudit" / revision
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = _read_metadata(repo_dir)
    slide_ids = _slide_ids(repo_dir)

    records: list[SourceRecord] = []
    rejected: list[dict[str, str]] = []
    for slide_id in slide_ids:
        try:
            image_src = repo_dir / "data" / "images" / f"{slide_id}.png"
            annotation_src = repo_dir / "data" / "annotations" / f"{slide_id}.json"
            description_src = repo_dir / "data" / "descriptions" / f"{slide_id}.json"
            for path in (image_src, annotation_src, description_src):
                if not path.exists():
                    raise ValueError(f"missing required file: {path.name}")
            image_dst = dataset_dir / "images" / image_src.name
            annotation_dst = dataset_dir / "annotations" / annotation_src.name
            description_dst = dataset_dir / "descriptions" / description_src.name
            for src, dst in (
                (image_src, image_dst),
                (annotation_src, annotation_dst),
                (description_src, description_dst),
            ):
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copyfile(src, dst)
            combined_sha256 = hashlib.sha256(
                (
                    file_hash(image_dst)
                    + file_hash(annotation_dst)
                    + file_hash(description_dst)
                ).encode()
            ).hexdigest()
            records.append(
                SourceRecord(
                    source_id=f"slideaudit-{slide_id}",
                    dataset_id=SLIDEAUDIT_DATASET_ID,
                    url=f"{SLIDEAUDIT_REPO_URL.removesuffix('.git')}/blob/{revision}/data/images/{image_src.name}",
                    license=license,
                    license_text_sha256=SLIDEAUDIT_LICENSE_TEXT_SHA256,
                    revision=revision,
                    upstream_commit=revision,
                    sha256=combined_sha256,
                    acquired_at=frozen_at,
                    local_path=image_dst.relative_to(output_dir).as_posix(),
                    redistributable=True,
                    citation="SlideAudit, arXiv:2508.03630, doi:10.1145/3746059.3747736",
                )
            )
        except Exception as error:
            rejected.append({"source_id": slide_id, "reason": str(error)})

    if not records:
        raise ValueError("no SlideAudit sources were acquired")

    metadata_index = {
        slide_id: metadata_rows.get(slide_id, {}) for slide_id in slide_ids
    }
    frozen: dict[str, Any] = {
        "dataset_id": SLIDEAUDIT_DATASET_ID,
        "revision": revision,
        "license_text": SLIDEAUDIT_LICENSE_TEXT,
        "license_text_sha256": SLIDEAUDIT_LICENSE_TEXT_SHA256,
        "taxonomy_version": SLIDEAUDIT_TAXONOMY_VERSION,
        "crosswalk_entries": [
            entry.model_dump(mode="json") for entry in SLIDEAUDIT_CROSSWALK
        ],
        "frozen_at": frozen_at.isoformat(),
        "slide_count": len(records),
        "rejected_count": len(rejected),
        "sources": [record.model_dump(mode="json") for record in records],
        "metadata": metadata_index,
        "rejected": rejected,
    }
    frozen["freeze_hash"] = content_hash(frozen)
    write_immutable(freeze_path, frozen)
    return frozen


def build_slideaudit_manifest(
    frozen: dict[str, Any], output_dir: Path, manifest_path: Path
) -> BenchmarkManifest:
    """Assemble and freeze the image-only SlideAudit BenchmarkManifest.

    This is the artifact consumed by ``eval run --suite slideaudit``; it
    carries the pinned sources, crosswalked cases, and taxonomy version in
    one immutable record.
    """
    sources = [SourceRecord.model_validate(item) for item in frozen["sources"]]
    cases = build_slideaudit_cases(frozen, output_dir)
    manifest = BenchmarkManifest(
        benchmark_id="slidex-slideaudit",
        revision=frozen["revision"],
        created_at=datetime.now(UTC),
        frozen_at=datetime.now(UTC),
        sources=sources,
        cases=sorted(cases, key=lambda case: case.case_id),
        taxonomy_version=SLIDEAUDIT_TAXONOMY_VERSION,
        crosswalk_entries=list(SLIDEAUDIT_CROSSWALK),
        preparation={
            "dataset": "slideaudit",
            "evidence_condition": "image_only",
            "note": (
                "No native IR is available; excluded from the native-IR "
                "intrinsic pool and reported separately per Sec. 13.4/13.9."
            ),
        },
    )
    manifest.manifest_hash = content_hash(
        manifest.model_dump(
            exclude={"manifest_hash", "created_at", "frozen_at"}, mode="json"
        )
    )
    if manifest_path.exists():
        existing = BenchmarkManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing.manifest_hash == manifest.manifest_hash:
            return existing
        raise FileExistsError(f"frozen manifest already exists: {manifest_path}")
    write_immutable(manifest_path, manifest)
    return manifest


def _case_id(slide_id: str, revision: str) -> str:
    payload = f"slidex-eval-v1\0slideaudit\0{revision}\0{slide_id}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def build_slideaudit_cases(
    frozen: dict[str, Any], output_dir: Path
) -> list[EvaluationCase]:
    """Build image-only EvaluationCase records from a frozen SlideAudit snapshot.

    Every SlideAudit slide is a distinct source with no clean twin, so cases
    are placed in ``sealed_test`` for open-world, image-only measurement and
    are excluded from any native-IR intrinsic pool via ``evidence_condition``.
    """
    crosswalk = list(SLIDEAUDIT_CROSSWALK)
    revision = frozen["revision"]
    cases: list[EvaluationCase] = []
    for source in frozen["sources"]:
        slide_id = source["source_id"].removeprefix("slideaudit-")
        image_path = output_dir / source["local_path"]
        annotation_path = (
            image_path.parent.parent / "annotations" / f"{slide_id}.json"
        )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        positive_labels = sorted(
            {
                entry["design_deficiency"]
                for entry in annotation["annotations"]
                if entry.get("response")
            }
        )
        image_sha256 = file_hash(image_path)
        artifact_id = hashlib.sha256(
            f"slideaudit:{revision}:{slide_id}:source_image".encode()
        ).hexdigest()[:24]
        preparation_record = PreparationRecord(
            original_sha256=image_sha256,
            source_url=source["url"],
            lineage=[
                LineageRecord(
                    artifact_id=artifact_id,
                    kind="source_image",
                    uri=source["local_path"],
                    sha256=image_sha256,
                )
            ],
        )
        case = EvaluationCase(
            case_id=_case_id(slide_id, revision),
            parent_deck_id=slide_id,
            source_id=source["source_id"],
            split=Split.SEALED_TEST,
            input_uri=source["local_path"],
            cluster_id=slide_id,
            content_sha256=image_sha256,
            preparation_record=preparation_record,
            metadata={
                "dataset": "slideaudit",
                "revision": revision,
                "image_dimensions": annotation.get("image_dimensions"),
                "source_type": frozen["metadata"]
                .get(slide_id, {})
                .get("source_type"),
            },
        )
        cases.append(
            apply_slideaudit_crosswalk(case, positive_labels, crosswalk)
        )
    return cases
