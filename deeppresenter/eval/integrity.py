"""Dataset integrity, leakage, quota, and replay gates."""

from __future__ import annotations

import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree
from PIL import Image, ImageChops, ImageStat

from deeppresenter.slidex.models import DefectClass

from .io import content_hash, file_hash
from .models import (
    BenchmarkManifest,
    EvaluationCase,
    IntegrityRecord,
    IntegrityStatus,
    ReviewStatus,
    Split,
    TaskType,
)

_PRESENTATION_NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

SEMANTIC_DEFECTS = {
    DefectClass.S1,
    DefectClass.S2,
    DefectClass.S3,
    DefectClass.S4,
    DefectClass.S5,
    DefectClass.S6,
}
GEOMETRY_DEFECTS = set(DefectClass) - SEMANTIC_DEFECTS
IMAGE_ARM_DEFECTS = {
    DefectClass.G1,
    DefectClass.G2,
    DefectClass.G3,
    DefectClass.G5,
    DefectClass.G6,
    DefectClass.G7,
    DefectClass.S1,
    DefectClass.S4,
    DefectClass.S6,
}


def image_difference(clean: Path, defective: Path) -> IntegrityRecord:
    """Return deterministic pixel evidence including the changed region."""
    with (
        Image.open(clean).convert("RGB") as left,
        Image.open(defective).convert("RGB") as right,
    ):
        if left.size != right.size:
            raise ValueError("clean and defective renders must have equal dimensions")
        difference = ImageChops.difference(left, right)
        bbox = difference.getbbox()
        stat = ImageStat.Stat(difference)
        mean = sum(stat.mean) / (len(stat.mean) * 255)
        histogram = difference.convert("L").histogram()
        changed = sum(histogram[1:])
        ratio = changed / (left.width * left.height)
        perceptual = sum(value * count for value, count in enumerate(histogram)) / max(
            1, left.width * left.height * 255
        )
    return IntegrityRecord(
        pixel_difference=mean,
        changed_bbox=bbox,
        changed_pixel_ratio=ratio,
        perceptual_difference=perceptual,
        rejection_reason="mutation has zero pixel difference" if bbox is None else None,
    )


def _slide_id_sequence(pptx_path: Path) -> list[str]:
    with zipfile.ZipFile(pptx_path) as archive:
        root = etree.fromstring(archive.read("ppt/presentation.xml"))
    return root.xpath("//p:sldIdLst/p:sldId/@r:id", namespaces=_PRESENTATION_NS)


def deck_order_difference(source: Path, mutated: Path) -> IntegrityRecord:
    """Verify a deck-scope reorder (S2) without relying on per-frame pixels.

    S2 only permutes ``p:sldId`` references in ``presentation.xml``; the
    mutated slide's own rendered pixels are identical to the source, so
    pixel-diff cannot serve as ground truth here. The deck-level ordering
    itself is the trusted native-IR evidence.
    """
    source_order = _slide_id_sequence(source)
    mutated_order = _slide_id_sequence(mutated)
    if sorted(source_order) != sorted(mutated_order):
        return IntegrityRecord(
            target_rule_passed=False,
            rejection_reason="slide set changed during reorder mutation",
        )
    changed = source_order != mutated_order
    return IntegrityRecord(
        target_rule_passed=changed,
        clean_rule_passed=source_order == source_order,
        rejection_reason=None if changed else "mutation left slide order unchanged",
    )


def validate_integrity_record(record: IntegrityRecord) -> IntegrityStatus:
    if record.pixel_difference is None:
        # Deck-scope evidence (e.g. S2 reorder) has no per-frame pixels;
        # `target_rule_passed` carries the native-IR ground truth instead.
        if record.target_rule_passed is None:
            return IntegrityStatus.PENDING
        if record.target_rule_passed is False:
            return IntegrityStatus.TARGET_RULE_FAILED
        return IntegrityStatus.VALID
    if record.pixel_difference <= 0 or record.changed_bbox is None:
        return IntegrityStatus.PIXEL_DIFF_ZERO
    if record.target_rule_passed is False or record.clean_rule_passed is False:
        return IntegrityStatus.TARGET_RULE_FAILED
    if record.collateral_high_severity_defects:
        return IntegrityStatus.COLLATERAL_DEFECT
    return IntegrityStatus.VALID


def audit_split_leakage(cases: list[EvaluationCase]) -> list[str]:
    """Audit all declared cluster identities, including near-duplicate fingerprints."""
    dimensions = ("cluster_id", "parent_deck_id", "template_fingerprint")
    failures: list[str] = []
    for dimension in dimensions:
        seen: dict[str, Split] = {}
        for case in cases:
            value = getattr(case, dimension)
            if not value:
                continue
            previous = seen.setdefault(value, case.split)
            if previous is not case.split:
                failures.append(
                    f"{dimension}:{value}:{previous.value}->{case.split.value}"
                )
    for metadata_key in (
        "text_fingerprint",
        "image_fingerprint",
        "source_package_fingerprint",
    ):
        seen = {}
        for case in cases:
            value = str(case.metadata.get(metadata_key, ""))
            if not value:
                continue
            previous = seen.setdefault(value, case.split)
            if previous is not case.split:
                failures.append(
                    f"{metadata_key}:{value}:{previous.value}->{case.split.value}"
                )
    return sorted(set(failures))


def _case_defect(case: EvaluationCase) -> tuple[DefectClass, bool] | None:
    if len(case.labels) != 1:
        return None
    label = case.labels[0]
    return label.defect_class, label.defective


def validate_controlled_pair_quotas(
    manifest: BenchmarkManifest,
    minimum: int = 30,
    minimum_decks: int = 15,
    required_defects: set[DefectClass] | None = None,
) -> list[str]:
    failures: list[str] = []
    counts: Counter[tuple[DefectClass, bool]] = Counter()
    decks: defaultdict[tuple[DefectClass, bool], set[str]] = defaultdict(set)
    for case in manifest.cases:
        item = _case_defect(case)
        if item and case.integrity_status is IntegrityStatus.VALID:
            counts[item] += 1
            decks[item].add(case.parent_deck_id)
    for defect in required_defects or set(DefectClass):
        for defective in (True, False):
            key = defect, defective
            if counts[key] < minimum:
                failures.append(
                    f"{defect.value}:{'defect' if defective else 'clean'} count {counts[key]} < {minimum}"
                )
            if defective and len(decks[key]) < minimum_decks:
                failures.append(
                    f"{defect.value}:defect decks {len(decks[key])} < {minimum_decks}"
                )
    return failures


def validate_e2e_quotas(manifest: BenchmarkManifest) -> list[str]:
    counts: Counter[tuple[Split, str]] = Counter()
    for case in manifest.cases:
        task_type = str(case.metadata.get("task_type", ""))
        counts[case.split, task_type] += 1
    failures: list[str] = []
    for task_type in TaskType:
        if counts[Split.PILOT, task_type.value] != 5:
            failures.append(f"pilot {task_type.value} != 5")
        if counts[Split.SEALED_TEST, task_type.value] != 25:
            failures.append(f"sealed {task_type.value} != 25")
    return failures


def validate_reviews(manifest: BenchmarkManifest) -> list[str]:
    failures: list[str] = []
    geometry: defaultdict[tuple[DefectClass, Split], list[EvaluationCase]] = (
        defaultdict(list)
    )
    reviewed: Counter[tuple[DefectClass, Split]] = Counter()
    for case in manifest.cases:
        item = _case_defect(case)
        if not item or not item[1]:
            continue
        defect = item[0]
        review = case.preparation_record.review if case.preparation_record else None
        if defect in SEMANTIC_DEFECTS and (
            not review or review.status is not ReviewStatus.ACCEPTED
        ):
            failures.append(f"semantic review missing:{case.case_id}")
        if defect in GEOMETRY_DEFECTS:
            key = defect, case.split
            geometry[key].append(case)
            if review and review.status is ReviewStatus.ACCEPTED:
                reviewed[key] += 1
    for key, cases in geometry.items():
        required = max(1, (len(cases) + 4) // 5)
        if reviewed[key] < required:
            failures.append(
                f"geometry review {key[0].value}/{key[1].value} {reviewed[key]} < {required}"
            )
    return failures


def validate_lineage(
    case: EvaluationCase, cache_root: Path, require_files: bool = True
) -> list[str]:
    record = case.preparation_record
    if not record or not record.lineage:
        return [f"missing lineage:{case.case_id}"]
    failures: list[str] = []
    known: set[str] = set()
    for artifact in record.lineage:
        if any(parent not in known for parent in artifact.parent_artifact_ids):
            failures.append(
                f"lineage parent order:{case.case_id}:{artifact.artifact_id}"
            )
        known.add(artifact.artifact_id)
        path = cache_root / artifact.uri
        if require_files and (not path.exists() or file_hash(path) != artifact.sha256):
            failures.append(
                f"lineage artifact mismatch:{case.case_id}:{artifact.artifact_id}"
            )
    return failures


def freeze_gate(
    manifest: BenchmarkManifest,
    cache_root: Path,
    benchmark_kind: str,
    require_files: bool = True,
) -> dict[str, object]:
    failures = audit_split_leakage(manifest.cases)
    for source in manifest.sources:
        if not source.dataset_id:
            failures.append(f"missing dataset_id:{source.source_id}")
        if not source.license_text_sha256:
            failures.append(f"missing license text hash:{source.source_id}")
        if not source.url or not source.revision or not source.license:
            failures.append(f"incomplete source provenance:{source.source_id}")
    if benchmark_kind == "controlled_pairs":
        required_defects = (
            IMAGE_ARM_DEFECTS
            if manifest.preparation.get("scope") == "image_arm"
            else set(DefectClass)
        )
        failures.extend(
            validate_controlled_pair_quotas(
                manifest,
                minimum=int(manifest.preparation.get("minimum_pairs", 30)),
                minimum_decks=int(manifest.preparation.get("minimum_decks", 15)),
                required_defects=required_defects,
            )
        )
        failures.extend(validate_reviews(manifest))
        frozen_image_classes: set[DefectClass] = set()
        for case in manifest.cases:
            if case.integrity_status is not IntegrityStatus.VALID:
                failures.append(
                    f"invalid integrity:{case.case_id}:{case.integrity_status.value}"
                )
            item = _case_defect(case)
            if (
                item
                and item[0] in IMAGE_ARM_DEFECTS
                and case.metadata.get("defective_render_uris")
            ):
                frozen_image_classes.add(item[0])
        missing_image_classes = IMAGE_ARM_DEFECTS - frozen_image_classes
        if missing_image_classes:
            failures.append(
                "image arm classes missing:"
                + ",".join(sorted(defect.value for defect in missing_image_classes))
            )
    elif benchmark_kind == "e2e":
        failures.extend(validate_e2e_quotas(manifest))
        for case in manifest.cases:
            if case.task_brief is None:
                failures.append(f"missing task brief:{case.case_id}")
    for case in manifest.cases:
        failures.extend(validate_lineage(case, cache_root, require_files=require_files))
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "manifest_hash": manifest.manifest_hash,
        "audit_hash": content_hash(sorted(set(failures))),
    }
