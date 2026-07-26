"""External dataset acquisition and taxonomy contracts."""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

from deeppresenter.slidex.models import DefectClass

from .models import CrosswalkEntry, DefectLabel, EvaluationCase, SourceRecord


def download_source(source: SourceRecord, cache_dir: Path, allowed_licenses: set[str]) -> Path:
    """Download one pinned source and reject mutable or unlicensed bytes."""
    if source.license.lower() not in {name.lower() for name in allowed_licenses}:
        raise ValueError(f"unapproved or missing license: {source.license}")
    if not source.revision.strip():
        raise ValueError("dataset revision must be pinned")
    filename = Path(urllib.parse.urlparse(source.url).path).name
    filename = filename or f"{source.source_id}.bin"
    target = cache_dir / source.source_id / source.revision / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(source.url, timeout=60) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != source.sha256:
        target.unlink(missing_ok=True)
        raise ValueError(f"dataset hash mismatch: {source.source_id}")
    return target


def validate_slideaudit_crosswalk(crosswalk: dict[str, list[str]] | list[CrosswalkEntry]) -> None:
    """Allow explicit unmapped labels while rejecting ambiguity and invalid targets."""
    if isinstance(crosswalk, dict):
        entries = [
            CrosswalkEntry(source_label=source, target_labels=targets, rationale="legacy crosswalk", evidence_condition="image_only", version="legacy", reviewed=True)
            for source, targets in crosswalk.items()
        ]
    else:
        entries = crosswalk
    labels: set[str] = set()
    for entry in entries:
        if entry.source_label in labels:
            raise ValueError(f"duplicate taxonomy crosswalk entry: {entry.source_label}")
        labels.add(entry.source_label)
        if not entry.rationale.strip() or not entry.version.strip():
            raise ValueError(f"incomplete taxonomy crosswalk entry: {entry.source_label}")
        if any(target not in DefectClass for target in entry.target_labels):
            raise ValueError(f"invalid taxonomy target: {entry.source_label}")


def apply_slideaudit_crosswalk(case: EvaluationCase, source_labels: list[str], entries: list[CrosswalkEntry]) -> EvaluationCase:
    """Preserve canonical source labels and add all mapped Slidex labels."""
    mapping = {entry.source_label: entry for entry in entries}
    mapped: list[DefectLabel] = []
    unmapped: list[str] = []
    for source_label in source_labels:
        entry = mapping.get(source_label)
        if entry is None or not entry.target_labels:
            unmapped.append(source_label)
            continue
        mapped.extend(DefectLabel(defect_class=target, defective=True, evidence_condition="image_only") for target in entry.target_labels)
    return case.model_copy(update={"labels": mapped, "metadata": {**case.metadata, "slideaudit_labels": source_labels, "unmapped_labels": unmapped, "evidence_condition": "image_only"}})


def image_arm_cases(cases: list[EvaluationCase]) -> list[EvaluationCase]:
    """Project the frozen nine-class image arm without resampling cases."""
    allowed = {DefectClass.G1, DefectClass.G2, DefectClass.G3, DefectClass.G5, DefectClass.G6, DefectClass.G7, DefectClass.S1, DefectClass.S4, DefectClass.S6}
    projected: list[EvaluationCase] = []
    for case in cases:
        labels = [label.model_copy(update={"evidence_condition": "image_only"}) for label in case.labels if label.defect_class in allowed]
        if not labels:
            continue
        render_uri = case.metadata.get("render_uri")
        if not render_uri and case.preparation_record:
            renders = [item.uri for item in case.preparation_record.lineage if item.kind == ("defective_render" if labels[0].defective else "clean_render")]
            render_uri = renders[0] if renders else None
        if not render_uri:
            raise ValueError(f"image arm case lacks render: {case.case_id}")
        projected.append(case.model_copy(update={"input_uri": str(render_uri), "labels": labels, "metadata": {**case.metadata, "evidence_condition": "image_only", "source_case_id": case.case_id}}))
    return projected
