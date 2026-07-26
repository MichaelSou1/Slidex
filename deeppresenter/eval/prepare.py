"""Deterministic benchmark preparation and integrity validation."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from lxml import etree
from PIL import Image, ImageChops, ImageStat

from .io import content_hash, file_hash, write_immutable
from .models import BenchmarkManifest, DefectLabel, EvaluationCase, SourceRecord, Split

SUPPORTED_DEFECTS = tuple(f"G{i}" for i in range(1, 8)) + tuple(
    f"S{i}" for i in range(1, 7)
)
IMAGE_ARM_DEFECTS = ("G1", "G2", "G3", "G5", "G6", "G7", "S1", "S4", "S6")


def deterministic_case_id(parent_deck_id: str, defect: str, variant: int) -> str:
    payload = f"slidex-eval-v1\0{parent_deck_id}\0{defect}\0{variant}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def deterministic_split(cluster_id: str, development_ratio: float = 0.2) -> Split:
    bucket = int(hashlib.sha256(cluster_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return Split.DEVELOPMENT if bucket < development_ratio else Split.SEALED_TEST


def validate_license(source: SourceRecord, allowed: set[str]) -> None:
    normalized = source.license.strip().lower()
    if normalized not in {item.lower() for item in allowed}:
        raise ValueError(f"unapproved or missing license: {source.license}")


def validate_source(source: SourceRecord, root: Path) -> None:
    if source.local_path:
        path = (root / source.local_path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("source path escapes preparation root")
        if file_hash(path) != source.sha256:
            raise ValueError(f"source hash mismatch: {source.source_id}")


def pixel_difference(clean: Path, defective: Path) -> float:
    with (
        Image.open(clean).convert("RGB") as left,
        Image.open(defective).convert("RGB") as right,
    ):
        if left.size != right.size:
            return 1.0
        stat = ImageStat.Stat(ImageChops.difference(left, right))
        return sum(stat.mean) / (len(stat.mean) * 255)


def require_nonzero_pixel_difference(clean: Path, defective: Path) -> float:
    difference = pixel_difference(clean, defective)
    if difference <= 0:
        raise ValueError(
            "dataset integrity failure: mutation has zero pixel difference"
        )
    return difference


def template_fingerprint(path: Path) -> str:
    """Hash normalized slide layout XML to group template near-duplicates."""
    if path.suffix.lower() != ".pptx":
        return file_hash(path)
    parts: list[bytes] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if "slideLayout" in name and name.endswith(".xml")
        )
        for name in names:
            root = etree.fromstring(archive.read(name))
            for node in root.xpath("//@id | //@name"):
                _ = node
            parts.append(etree.tostring(root, method="c14n"))
    return hashlib.sha256(b"".join(parts)).hexdigest()


def mutate_pptx_xml(source: Path, target: Path, defect_class: str) -> None:
    """Apply a deterministic native XML mutation for a smoke/paired fixture.

    Production preparation must render and validate this candidate before admitting it.
    """
    if defect_class not in SUPPORTED_DEFECTS:
        raise ValueError(f"unknown defect class: {defect_class}")
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
        mutated = False
        for info in src.infolist():
            data = src.read(info.filename)
            if (
                not mutated
                and info.filename.startswith("ppt/slides/slide")
                and info.filename.endswith(".xml")
            ):
                root = etree.fromstring(data)
                nodes = root.xpath("//*[local-name()='off' or local-name()='ext']")
                if nodes:
                    attribute = (
                        "x"
                        if defect_class.startswith("G") and "x" in nodes[0].attrib
                        else next(iter(nodes[0].attrib), None)
                    )
                    if attribute:
                        nodes[0].set(
                            attribute, str(int(nodes[0].get(attribute, "0")) + 12700)
                        )
                        data = etree.tostring(
                            root,
                            xml_declaration=True,
                            encoding="UTF-8",
                            standalone=True,
                        )
                        mutated = True
            dst.writestr(info, data)
    if not mutated:
        target.unlink(missing_ok=True)
        raise ValueError("no mutable native geometry found")


def prepare_manifest(spec_path: Path, output_path: Path) -> BenchmarkManifest:
    """Freeze a local, license-checked manifest; downloads are intentionally explicit."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root = spec_path.parent
    sources = [SourceRecord.model_validate(item) for item in spec["sources"]]
    allowed = set(spec.get("allowed_licenses", []))
    for source in sources:
        validate_license(source, allowed)
        validate_source(source, root)

    cases: list[EvaluationCase] = []
    for item in spec["cases"]:
        parent = item["parent_deck_id"]
        defect = item.get("defect_class", "clean")
        variant = int(item.get("variant", 0))
        cluster = item.get("cluster_id", parent)
        split = (
            Split(item["split"]) if item.get("split") else deterministic_split(cluster)
        )
        target_defect = item.get("target_defect_class")
        labels = []
        if defect != "clean":
            labels.append(
                DefectLabel(
                    defect_class=defect, defective=True, **item.get("label", {})
                )
            )
        elif target_defect:
            labels.append(
                DefectLabel(
                    defect_class=target_defect,
                    defective=False,
                    **item.get("label", {}),
                )
            )
        cases.append(
            EvaluationCase(
                case_id=deterministic_case_id(parent, defect, variant),
                parent_deck_id=parent,
                source_id=item["source_id"],
                split=split,
                input_uri=item["input_uri"],
                clean_reference_uri=item.get("clean_reference_uri"),
                labels=labels,
                task_brief=item.get("task_brief"),
                cluster_id=cluster,
                content_sha256=item["content_sha256"],
                template_fingerprint=item.get("template_fingerprint"),
                integrity_status=item.get("integrity_status", "valid"),
                preparation_record=item.get("preparation_record"),
                metadata=item.get("metadata", {}),
            )
        )
    manifest = BenchmarkManifest(
        benchmark_id=spec["benchmark_id"],
        revision=spec["revision"],
        created_at=datetime.now(UTC),
        frozen_at=datetime.now(UTC),
        sources=sources,
        cases=sorted(cases, key=lambda case: case.case_id),
        taxonomy_version=spec.get("taxonomy_version", "1.0"),
        crosswalk=spec.get("crosswalk", {}),
        crosswalk_entries=spec.get("crosswalk_entries", []),
        preparation=spec.get("preparation", {}),
    )
    manifest.manifest_hash = content_hash(
        manifest.model_dump(
            exclude={"manifest_hash", "created_at", "frozen_at"}, mode="json"
        )
    )
    if output_path.exists():
        existing = BenchmarkManifest.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if existing.manifest_hash == manifest.manifest_hash:
            return existing
        raise FileExistsError(f"frozen manifest already exists: {output_path}")
    write_immutable(output_path, manifest)
    return manifest
