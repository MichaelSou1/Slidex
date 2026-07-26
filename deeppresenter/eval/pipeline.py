"""Resumable Phase 13 dataset preparation stages."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .datasets import download_source
from .integrity import (
    deck_order_difference,
    freeze_gate,
    image_difference,
    validate_integrity_record,
)
from .io import capture_environment, content_hash, file_hash, write_immutable
from .models import (
    BenchmarkManifest,
    DefectLabel,
    EvaluationCase,
    HumanReview,
    IntegrityStatus,
    LineageRecord,
    PreparationRecord,
    ReviewStatus,
    SourceRecord,
    Split,
)
from .mutations import mutate_pptx
from .prepare import deterministic_case_id, deterministic_split, template_fingerprint
from .preregister import freeze_preregistration


class PreparationStage(StrEnum):
    ACQUIRE = "acquire"
    NORMALIZE = "normalize"
    CLUSTER = "cluster"
    SPLIT = "split"
    MUTATE = "mutate"
    RENDER = "render"
    VALIDATE = "validate"
    ANNOTATE = "annotate"
    FREEZE = "freeze"
    AUDIT = "audit"


_STAGE_ORDER = tuple(PreparationStage)


def defect_class_is_deck_scope(defect_class: str) -> bool:
    """Return True for defects whose evidence spans multiple slides.

    Deck-scope defects (currently only S2) have no single-frame render and
    are evaluated from the native IR/text rather than ``real_layout`` pixels.
    """
    return defect_class == "S2"


def _normalized_pptx_text(path: Path) -> str:
    if path.suffix.lower() != ".pptx":
        return " ".join(
            path.read_text(encoding="utf-8", errors="ignore").lower().split()
        )
    texts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("ppt/slides/slide") or not name.endswith(".xml"):
                continue
            root = __import__("lxml.etree", fromlist=["etree"]).fromstring(
                archive.read(name)
            )
            texts.extend(
                str(value) for value in root.xpath("//*[local-name()='t']/text()")
            )
    return " ".join(" ".join(texts).lower().split())


class DatasetPipeline:
    """Execute deterministic stages and persist every success or rejection."""

    def __init__(self, cache_root: Path) -> None:
        self.root = cache_root
        for name in (
            "sources",
            "normalized",
            "clusters",
            "mutations",
            "renders",
            "annotations",
            "manifests",
            "preregistrations",
            "runs",
            "audits",
            "rejected",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _stage_record(self, pipeline_id: str, stage: PreparationStage) -> Path:
        return self.root / "audits" / pipeline_id / f"{stage.value}.json"

    def _write_stage(
        self,
        pipeline_id: str,
        stage: PreparationStage,
        payload: dict[str, Any],
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self._stage_record(pipeline_id, stage)
        record = {
            "stage": stage.value,
            "pipeline_id": pipeline_id,
            "inputs": inputs or {},
            **payload,
        }
        record["record_hash"] = content_hash(record)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("record_hash") == record["record_hash"]:
                return existing
            raise FileExistsError(
                f"stage {stage.value} is frozen with different inputs; use a new pipeline ID"
            )
        write_immutable(path, record)
        return record

    def _require_previous(self, pipeline_id: str, stage: PreparationStage) -> None:
        index = _STAGE_ORDER.index(stage)
        if (
            index
            and not self._stage_record(pipeline_id, _STAGE_ORDER[index - 1]).exists()
        ):
            raise ValueError(
                f"stage {stage.value} requires {_STAGE_ORDER[index - 1].value}"
            )

    def acquire(
        self, pipeline_id: str, source_spec: Path, allowed_licenses: set[str]
    ) -> dict[str, Any]:
        records = json.loads(source_spec.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("source spec must be a JSON list")
        acquired: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for raw in records:
            try:
                source = SourceRecord.model_validate(raw)
                if source.license.lower() not in {
                    name.lower() for name in allowed_licenses
                }:
                    raise ValueError(f"unapproved or missing license: {source.license}")
                if source.local_path:
                    original = (source_spec.parent / source.local_path).resolve()
                    if not original.is_relative_to(source_spec.parent.resolve()):
                        raise ValueError("source path escapes source spec directory")
                    if file_hash(original) != source.sha256:
                        raise ValueError(f"dataset hash mismatch: {source.source_id}")
                    path = (
                        self.root
                        / "sources"
                        / source.source_id
                        / source.revision
                        / original.name
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        shutil.copyfile(original, path)
                else:
                    path = download_source(
                        source, self.root / "sources", allowed_licenses
                    )
                acquired.append(
                    {
                        "source": source.model_dump(mode="json"),
                        "path": path.relative_to(self.root).as_posix(),
                    }
                )
            except Exception as error:
                rejected.append(
                    {
                        "source_id": str(raw.get("source_id", "unknown")),
                        "reason": str(error),
                    }
                )
        return self._write_stage(
            pipeline_id,
            PreparationStage.ACQUIRE,
            {"acquired": acquired, "rejected": rejected},
            {
                "source_spec_sha256": file_hash(source_spec),
                "allowed_licenses": sorted(allowed_licenses),
            },
        )

    def normalize(self, pipeline_id: str) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.NORMALIZE)
        acquired = json.loads(
            self._stage_record(pipeline_id, PreparationStage.ACQUIRE).read_text()
        )["acquired"]
        normalized: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for item in acquired:
            source = self.root / item["path"]
            destination = self.root / "normalized" / pipeline_id / source.name
            try:
                if source.suffix.lower() == ".pptx":
                    with zipfile.ZipFile(source) as archive:
                        archive.testzip()
                        if not any(
                            name.startswith("ppt/slides/slide")
                            for name in archive.namelist()
                        ):
                            raise ValueError("PPTX contains no slides")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                normalized.append(
                    {
                        **item,
                        "normalized_path": destination.relative_to(
                            self.root
                        ).as_posix(),
                        "normalized_sha256": file_hash(destination),
                    }
                )
            except Exception as error:
                rejected.append(
                    {"source_id": item["source"]["source_id"], "reason": str(error)}
                )
        return self._write_stage(
            pipeline_id,
            PreparationStage.NORMALIZE,
            {
                "normalized": normalized,
                "rejected": rejected,
                "environment": capture_environment(),
            },
        )

    def cluster(self, pipeline_id: str) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.CLUSTER)
        items = json.loads(
            self._stage_record(pipeline_id, PreparationStage.NORMALIZE).read_text()
        )["normalized"]
        clusters = []
        for item in items:
            path = self.root / item["normalized_path"]
            template = template_fingerprint(path)
            text = _normalized_pptx_text(path)
            text_fingerprint = hashlib.sha256(text.encode()).hexdigest()
            cluster_id = hashlib.sha256(template.encode()).hexdigest()[:24]
            clusters.append(
                {
                    **item,
                    "template_fingerprint": template,
                    "text_fingerprint": text_fingerprint,
                    "cluster_id": cluster_id,
                }
            )
        return self._write_stage(
            pipeline_id, PreparationStage.CLUSTER, {"clusters": clusters}
        )

    def split(self, pipeline_id: str) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.SPLIT)
        clusters = json.loads(
            self._stage_record(pipeline_id, PreparationStage.CLUSTER).read_text()
        )["clusters"]
        assigned = [
            {**item, "split": deterministic_split(item["cluster_id"]).value}
            for item in clusters
        ]
        return self._write_stage(
            pipeline_id, PreparationStage.SPLIT, {"items": assigned}
        )

    def mutate(self, pipeline_id: str, variants: int = 1) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.MUTATE)
        from deeppresenter.slidex.models import DefectClass

        items = json.loads(
            self._stage_record(pipeline_id, PreparationStage.SPLIT).read_text()
        )["items"]
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for item in items:
            source = self.root / item["normalized_path"]
            if source.suffix.lower() != ".pptx":
                continue
            for defect in DefectClass:
                for variant in range(variants):
                    target = (
                        self.root
                        / "mutations"
                        / pipeline_id
                        / defect.value
                        / f"{item['source']['source_id']}-{variant}.pptx"
                    )
                    try:
                        mutation = mutate_pptx(
                            source, target, defect, item["source"]["source_id"], variant
                        )
                        candidates.append(
                            {
                                **item,
                                "defect_class": defect.value,
                                "variant": variant,
                                "mutation": mutation.model_dump(mode="json"),
                                "mutated_path": target.relative_to(
                                    self.root
                                ).as_posix(),
                                "mutated_sha256": file_hash(target),
                            }
                        )
                    except Exception as error:
                        rejected.append(
                            {
                                "source_id": item["source"]["source_id"],
                                "defect_class": defect.value,
                                "variant": str(variant),
                                "reason": str(error),
                            }
                        )
        return self._write_stage(
            pipeline_id,
            PreparationStage.MUTATE,
            {"candidates": candidates, "rejected": rejected},
            {"variants": variants},
        )

    async def render(
        self, pipeline_id: str, dpi: int = 144, concurrency: int = 2
    ) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.RENDER)
        from deeppresenter.slidex.export import LibreOfficeRenderer

        candidates = json.loads(
            self._stage_record(pipeline_id, PreparationStage.MUTATE).read_text()
        )["candidates"]
        # S2 (narrative-order break) only permutes slide references in
        # presentation.xml; the mutated slide's own pixels are unchanged, so
        # single-frame pixel-diff cannot validate it. It is verified via
        # deck-order integrity in `validate_deck_order` instead of rendering.
        candidates = [
            item
            for item in candidates
            if item["defect_class"] != "S2" and item["split"] == "sealed_test"
        ]
        renderer = LibreOfficeRenderer(dpi=dpi)
        semaphore = asyncio.Semaphore(concurrency)
        clean_tasks: dict[str, asyncio.Task[tuple[Path, list[Path], Any]]] = {}

        async def render_clean(item: dict[str, Any]) -> tuple[Path, list[Path], Any]:
            clean_key = item["normalized_sha256"]
            async with semaphore:
                return await renderer.render(
                    self.root / item["normalized_path"],
                    self.root / "renders" / pipeline_id / "clean" / clean_key,
                )

        for item in candidates:
            clean_key = item["normalized_sha256"]
            if clean_key not in clean_tasks:
                clean_tasks[clean_key] = asyncio.create_task(render_clean(item))

        async def render_candidate(
            item: dict[str, Any],
        ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
            try:
                _, clean_pages, _ = await clean_tasks[item["normalized_sha256"]]
                async with semaphore:
                    _, defective_pages, command = await renderer.render(
                        self.root / item["mutated_path"],
                        self.root
                        / "renders"
                        / pipeline_id
                        / "defective"
                        / item["mutation"]["mutation_id"],
                    )
                if len(clean_pages) != len(defective_pages):
                    raise ValueError("clean/defective page count mismatch")
                return (
                    {
                        **item,
                        "clean_renders": [
                            path.relative_to(self.root).as_posix()
                            for path in clean_pages
                        ],
                        "defective_renders": [
                            path.relative_to(self.root).as_posix()
                            for path in defective_pages
                        ],
                        "render_command": command.model_dump(mode="json"),
                    },
                    None,
                )
            except Exception as error:
                return None, {
                    "mutation_id": item["mutation"]["mutation_id"],
                    "reason": str(error),
                }

        outcomes = await asyncio.gather(
            *(render_candidate(item) for item in candidates)
        )
        rendered = [item for item, _ in outcomes if item is not None]
        rejected = [error for _, error in outcomes if error is not None]
        return self._write_stage(
            pipeline_id,
            PreparationStage.RENDER,
            {"rendered": rendered, "rejected": rejected},
            {"dpi": dpi, "concurrency": concurrency},
        )

    def validate(self, pipeline_id: str) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.VALIDATE)
        rendered = json.loads(
            self._stage_record(pipeline_id, PreparationStage.RENDER).read_text()
        )["rendered"]
        valid: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        environment_hash = content_hash(capture_environment())
        for item in rendered:
            changed_records = []
            for clean, defective in zip(
                item["clean_renders"], item["defective_renders"], strict=True
            ):
                record = image_difference(self.root / clean, self.root / defective)
                changed_records.append(record)
            changed = [
                record
                for record in changed_records
                if record.pixel_difference and record.pixel_difference > 0
            ]
            if not changed:
                rejected.append(
                    {
                        "mutation_id": item["mutation"]["mutation_id"],
                        "reason": "pixel_diff_zero",
                    }
                )
                continue
            aggregate = max(changed, key=lambda record: record.pixel_difference or 0)
            aggregate.render_environment_hash = environment_hash
            status = validate_integrity_record(aggregate)
            valid.append(
                {
                    **item,
                    "integrity": aggregate.model_dump(mode="json"),
                    "integrity_status": status.value,
                }
            )
        # S2 (narrative-order break) never enters the render stage; its
        # ground truth is the deck-level slide ordering in presentation.xml,
        # not per-frame pixels.
        mutated = json.loads(
            self._stage_record(pipeline_id, PreparationStage.MUTATE).read_text()
        )["candidates"]
        for item in mutated:
            if item["defect_class"] != "S2" or item["split"] != "sealed_test":
                continue
            record = deck_order_difference(
                self.root / item["normalized_path"], self.root / item["mutated_path"]
            )
            record.render_environment_hash = environment_hash
            status = validate_integrity_record(record)
            if status is not IntegrityStatus.VALID:
                rejected.append(
                    {
                        "mutation_id": item["mutation"]["mutation_id"],
                        "reason": record.rejection_reason or status.value,
                    }
                )
                continue
            valid.append(
                {
                    **item,
                    "integrity": record.model_dump(mode="json"),
                    "integrity_status": status.value,
                }
            )
        return self._write_stage(
            pipeline_id,
            PreparationStage.VALIDATE,
            {"valid": valid, "rejected": rejected},
        )

    def annotate(self, pipeline_id: str, annotations: Path) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.ANNOTATE)
        valid = json.loads(
            self._stage_record(pipeline_id, PreparationStage.VALIDATE).read_text()
        )["valid"]
        raw = json.loads(annotations.read_text(encoding="utf-8"))
        reviews = {
            item["mutation_id"]: HumanReview.model_validate(item["review"])
            for item in raw
        }
        annotated = []
        rejected = []
        for item in valid:
            mutation_id = item["mutation"]["mutation_id"]
            review = reviews.get(mutation_id, HumanReview(status=ReviewStatus.PENDING))
            target = (
                annotated
                if review.status in {ReviewStatus.ACCEPTED, ReviewStatus.NOT_REQUIRED}
                else rejected
            )
            target.append({**item, "review": review.model_dump(mode="json")})
        return self._write_stage(
            pipeline_id,
            PreparationStage.ANNOTATE,
            {"annotated": annotated, "rejected": rejected},
            {"annotations_sha256": file_hash(annotations)},
        )

    def freeze(
        self,
        pipeline_id: str,
        benchmark_id: str,
        revision: str,
        config_hashes: dict[str, str],
        scope: str = "full_taxonomy",
    ) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.FREEZE)
        annotated = json.loads(
            self._stage_record(pipeline_id, PreparationStage.ANNOTATE).read_text()
        )["annotated"]
        sources_by_id: dict[str, SourceRecord] = {}
        cases: list[EvaluationCase] = []
        for item in annotated:
            source = SourceRecord.model_validate(item["source"])
            sources_by_id[source.source_id] = source
            mutation_id = item["mutation"]["mutation_id"]
            lineage: list[LineageRecord] = []
            parent = None
            # Deck-scope mutations (S2) carry no render artifacts; their
            # trusted evidence is the source/normalized/defective PPTX only.
            clean_render_uris = item.get("clean_renders", [])
            defective_render_uris = item.get("defective_renders", [])
            artifact_uris = [
                ("source", item["path"]),
                ("normalized", item["normalized_path"]),
                ("defective", item["mutated_path"]),
                *(("clean_render", uri) for uri in clean_render_uris),
                *(("defective_render", uri) for uri in defective_render_uris),
            ]
            for index, (kind, uri) in enumerate(artifact_uris):
                path = self.root / uri
                artifact_id = hashlib.sha256(
                    f"{mutation_id}:{kind}:{index}".encode()
                ).hexdigest()[:24]
                lineage.append(
                    LineageRecord(
                        artifact_id=artifact_id,
                        kind=kind,
                        uri=uri,
                        sha256=file_hash(path),
                        parent_artifact_ids=[parent] if parent else [],
                    )
                )
                parent = artifact_id
            record = PreparationRecord(
                original_sha256=source.sha256,
                normalized_sha256=item["normalized_sha256"],
                source_url=source.url,
                renderer_versions={
                    "environment": item["integrity"]["render_environment_hash"]
                },
                mutation=item["mutation"],
                integrity=item["integrity"],
                review=item["review"],
                lineage=lineage,
            )
            defect = item["defect_class"]
            evidence_condition = (
                "native_ir" if defect_class_is_deck_scope(defect) else "real_layout"
            )
            common = dict(
                parent_deck_id=source.source_id,
                source_id=source.source_id,
                split=Split(item["split"]),
                clean_reference_uri=item["normalized_path"],
                cluster_id=item["cluster_id"],
                template_fingerprint=item["template_fingerprint"],
                preparation_record=record,
                metadata={
                    "text_fingerprint": item["text_fingerprint"],
                    "clean_render_uris": clean_render_uris,
                    "defective_render_uris": defective_render_uris,
                    "target_slide_part": item["mutation"]["parameters"].get(
                        "slide_part"
                    ),
                    "evidence_condition": "native_ir"
                    if defect_class_is_deck_scope(item["defect_class"])
                    else "real_layout",
                },
            )
            cases.append(
                EvaluationCase(
                    case_id=deterministic_case_id(
                        source.source_id, defect, item["variant"]
                    ),
                    input_uri=item["mutated_path"],
                    labels=[
                        DefectLabel(
                            defect_class=defect,
                            defective=True,
                            element_id=item["mutation"].get("target_element_id"),
                            evidence_condition=evidence_condition,
                        )
                    ],
                    content_sha256=item["mutated_sha256"],
                    **common,
                )
            )
            cases.append(
                EvaluationCase(
                    case_id=deterministic_case_id(
                        source.source_id, "clean-" + defect, item["variant"]
                    ),
                    input_uri=item["normalized_path"],
                    labels=[
                        DefectLabel(
                            defect_class=defect,
                            defective=False,
                            evidence_condition=evidence_condition,
                        )
                    ],
                    content_sha256=item["normalized_sha256"],
                    **common,
                )
            )
        now = datetime.now(UTC)
        manifest = BenchmarkManifest(
            benchmark_id=benchmark_id,
            revision=revision,
            created_at=now,
            frozen_at=now,
            sources=sorted(sources_by_id.values(), key=lambda x: x.source_id),
            cases=sorted(cases, key=lambda x: x.case_id),
            preparation={
                "pipeline_id": pipeline_id,
                "environment": capture_environment(),
                "scope": scope,
                "minimum_pairs": 10,
                "minimum_decks": 10,
                "human_scoring": "skipped_by_user",
            },
        )
        manifest.manifest_hash = content_hash(
            manifest.model_dump(
                exclude={"manifest_hash", "created_at", "frozen_at"}, mode="json"
            )
        )
        manifest_path = self.root / "manifests" / f"{benchmark_id}-{revision}.json"
        write_immutable(manifest_path, manifest)
        preregistration = freeze_preregistration(
            self.root / "preregistrations" / f"{benchmark_id}-{revision}.json",
            config_hashes,
        )
        return self._write_stage(
            pipeline_id,
            PreparationStage.FREEZE,
            {
                "manifest": manifest_path.relative_to(self.root).as_posix(),
                "manifest_hash": manifest.manifest_hash,
                "preregistration_hash": preregistration.preregistration_hash,
            },
            {
                "benchmark_id": benchmark_id,
                "revision": revision,
                "config_hashes": config_hashes,
            },
        )

    def audit(
        self, pipeline_id: str, benchmark_kind: str, require_files: bool = True
    ) -> dict[str, Any]:
        self._require_previous(pipeline_id, PreparationStage.AUDIT)
        frozen = json.loads(
            self._stage_record(pipeline_id, PreparationStage.FREEZE).read_text()
        )
        manifest = BenchmarkManifest.model_validate_json(
            (self.root / frozen["manifest"]).read_text()
        )
        report = freeze_gate(
            manifest, self.root, benchmark_kind, require_files=require_files
        )
        return self._write_stage(
            pipeline_id,
            PreparationStage.AUDIT,
            report,
            {"benchmark_kind": benchmark_kind, "require_files": require_files},
        )


def run_render_stage(
    pipeline: DatasetPipeline, pipeline_id: str, dpi: int, concurrency: int = 2
) -> dict[str, Any]:
    return asyncio.run(pipeline.render(pipeline_id, dpi=dpi, concurrency=concurrency))
