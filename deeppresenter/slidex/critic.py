"""Frozen symbolic-neural hybrid critic guided by failure attribution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.cache import ContentCache
from deeppresenter.slidex.inspectors import (
    AlignmentInspector,
    AtomicNeuralClient,
    BrandColorInspector,
    DeckSemanticInspector,
    DeclaredOverflowInspector,
    DensityInspector,
    ImageTextContradictionInspector,
    MarginInspector,
    OverlapInspector,
    ReferenceInspector,
    RenderAnomalyInspector,
    RenderOnlyGeometryInspector,
    RenderOverflowInspector,
    TerminologyInspector,
    TitleBodyMismatchInspector,
    TypographyInspector,
    inspect_safely,
)
from deeppresenter.slidex.inspectors.neural import _RENDER_ONLY_DEFINITIONS
from deeppresenter.slidex.inspectors.base import Inspector
from deeppresenter.slidex.models import (
    ArtifactTrust,
    DefectClass,
    Evidence,
    EvidenceSource,
    InspectionContext,
    InspectionReport,
    InspectionResult,
    InspectionStatus,
    Provenance,
    RepairHint,
    RouteRecord,
    SlideArtifact,
    WholeRubricVerdict,
)
from deeppresenter.slidex.router import (
    EvidenceAvailability,
    FrozenCriticRouter,
    FrozenRouterConfig,
    RouteDecision,
)
from deeppresenter.utils.config import LLM, SlidexConfig


class SymbolicCritic:
    """Run deterministic inspectors and preserve every component result."""

    def __init__(
        self,
        inspectors: Iterable[Inspector],
        *,
        router_version: str = "1.0",
        taxonomy_version: str = "1.0",
    ) -> None:
        self.inspectors = list(inspectors)
        self.router_version = router_version
        self.taxonomy_version = taxonomy_version

    def inspect(self, artifact: SlideArtifact) -> InspectionReport:
        results = [
            item
            for inspector in self.inspectors
            for item in inspect_safely(inspector, artifact)
        ]
        return _report(
            artifact,
            results,
            router_version=self.router_version,
            taxonomy_version=self.taxonomy_version,
        )


class GenericWholeRubricCritic:
    """Phase 13 E2E ``generic critic`` control: one whole-rubric VLM verdict.

    Unlike :class:`HybridCritic` (frozen symbolic-neural-reference router),
    this issues exactly one VLM call per slide asking it to find any
    presentation-quality defect using its own judgment over the full 13-class
    taxonomy, with no per-class routing or reference-assisted comparison.
    This isolates the causal contribution of the frozen router/reference
    machinery from merely having *some* automated critic in the loop (13.7).
    """

    _WHOLE_RUBRIC_TAXONOMY = (
        "G1 text overflowing its container; G2 elements overlapping; "
        "G3 misaligned elements; G4 illegible or inconsistent font sizing; "
        "G5 off-palette/clashing colors; G6 content crossing the safe margin "
        "or bleeding off the slide; G7 elements crammed together without "
        "breathing room; S1 body text unrelated to the slide title; "
        "S3 inconsistent terminology for the same concept; S4 text far too "
        "dense to read at a glance; S5 a logically necessary section "
        "missing; S6 an image contradicting its caption or nearby text; "
        "S2 (narrative ordering) is out of scope for a single-slide view."
    )

    def __init__(self, *, critic_model: LLM, router_version: str = "1.0", taxonomy_version: str = "1.0") -> None:
        self.client = AtomicNeuralClient(critic_model, require_multimodal=True)
        self.router_version = router_version
        self.taxonomy_version = taxonomy_version

    async def inspect(self, context: InspectionContext) -> InspectionReport:
        render_path = context.render_path
        if not render_path:
            # No render evidence: whole-rubric VLM has nothing to look at.
            result = InspectionResult(
                defect_class=DefectClass.G1,
                status=InspectionStatus.DEFER,
                severity=0.0,
                confidence=0.0,
                evidence=[
                    Evidence(
                        source=EvidenceSource.MODEL,
                        detail="generic critic requires a rendered slide image",
                    )
                ],
                inspector_version="generic-whole-rubric-v1",
            )
            return _report(
                context.artifact,
                [result],
                router_version=self.router_version,
                taxonomy_version=self.taxonomy_version,
            )
        prompt = (
            "Inspect this single presentation slide for ANY presentation-quality "
            "defect using your own judgment. Consider the full defect taxonomy: "
            f"{self._WHOLE_RUBRIC_TAXONOMY} Report every defect you find with its "
            "class, severity, confidence, and localizing evidence; if you find "
            "nothing wrong, return overall_verdict=pass with an empty findings list."
        )
        response = await self.client.model.run(
            [
                {
                    "role": "user",
                    "content": self.client.build_content(prompt, [render_path]),
                }
            ],
            response_format=WholeRubricVerdict,
            retry_times=1,
        )
        raw = response.choices[0].message.content or ""
        verdict = WholeRubricVerdict.model_validate_json(raw)
        results = [
            InspectionResult(
                defect_class=finding.defect_class,
                status=InspectionStatus.FAIL,
                severity=finding.severity,
                confidence=finding.confidence,
                evidence=[
                    Evidence(source=EvidenceSource.MODEL, detail=detail)
                    for detail in finding.evidence
                ],
                element_ids=finding.element_ids,
                repair_hint=finding.repair_suggestion,
                inspector_version="generic-whole-rubric-v1",
            )
            for finding in verdict.findings
        ]
        found_classes = {item.defect_class for item in results}
        for defect_class in DefectClass:
            if defect_class in found_classes or defect_class is DefectClass.S2:
                continue
            status = (
                InspectionStatus.DEFER
                if verdict.overall_verdict == "defer"
                else InspectionStatus.PASS
            )
            results.append(
                InspectionResult(
                    defect_class=defect_class,
                    status=status,
                    severity=0.0,
                    confidence=0.5,
                    evidence=[
                        Evidence(
                            source=EvidenceSource.MODEL,
                            detail=verdict.defer_reason
                            or "whole-rubric pass reported no defect of this class",
                        )
                    ],
                    inspector_version="generic-whole-rubric-v1",
                )
            )
        return _report(
            context.artifact,
            results,
            router_version=self.router_version,
            taxonomy_version=self.taxonomy_version,
        )


class HybridCritic:
    """Execute a frozen route without consulting earlier model verdicts to reroute."""

    def __init__(
        self,
        config: SlidexConfig,
        *,
        critic_model: LLM | None = None,
        semantic_model: LLM | None = None,
        router: FrozenCriticRouter | None = None,
        cache: ContentCache | None = None,
    ) -> None:
        self.config = config
        router_config = FrozenRouterConfig(
            router_version=config.router_version,
            taxonomy_version=config.taxonomy_version,
            reference_policy=config.reference_policy,
        )
        self.router = router or FrozenCriticRouter(router_config)
        self.cache = cache
        self.critic_client = (
            AtomicNeuralClient(
                critic_model, require_multimodal=True, cache_results=True
            )
            if critic_model and critic_model.is_multimodal is True
            else None
        )
        self.semantic_client = (
            AtomicNeuralClient(semantic_model or critic_model, cache_results=True)
            if (semantic_model or critic_model)
            else None
        )

    async def inspect(self, context: InspectionContext) -> InspectionReport:
        """Inspect every taxonomy class and retain conflicts and capability limits."""
        cache_key = ContentCache.key(
            "hybrid-inspection",
            context,
            self.router.config.config_hash,
            bool(self.critic_client),
            bool(self.semantic_client),
        )
        if self.cache is not None:
            cached = self.cache.get_json("inspection", cache_key)
            if cached is not None:
                return InspectionReport.model_validate(cached)
        decisions = [
            self.router.route(
                defect_class,
                EvidenceAvailability.from_context(context),
                context.artifact.trust,
            )
            for defect_class in DefectClass
        ]
        grouped_results = await asyncio.gather(
            *(self._execute(decision, context) for decision in decisions)
        )
        results = [result for group in grouped_results for result in group]
        conflicts = _conflicts(results, context.artifact.trust)
        resolved_status = _resolve_status(results, context.artifact.trust)
        limits = sorted(
            {item.capability_limit for item in decisions if item.capability_limit}
        )
        report = _report(
            context.artifact,
            results,
            router_version=self.router.config.router_version,
            taxonomy_version=self.router.config.taxonomy_version,
            router_hash=self.router.config.config_hash,
            routes=[
                RouteRecord(
                    defect_class=item.defect_class,
                    stages=[stage.inspector for stage in item.stages],
                    reason=item.reason,
                    missing_evidence=item.missing_evidence,
                    capability_limit=item.capability_limit,
                )
                for item in decisions
            ],
            conflicts=conflicts,
            resolved_status=resolved_status,
            capability_limits=limits,
        )
        if self.cache is not None:
            self.cache.put_json("inspection", cache_key, report)
        return report

    async def _execute(
        self, decision: RouteDecision, context: InspectionContext
    ) -> list[InspectionResult]:
        if not decision.stages:
            return [
                self._defer(
                    decision.defect_class, decision.reason, decision.missing_evidence
                )
            ]
        collected: list[InspectionResult] = []
        for index, stage in enumerate(decision.stages):
            inspector = self._inspector(stage.inspector, decision.defect_class, context)
            if inspector is None:
                current = [
                    self._defer(
                        decision.defect_class,
                        f"inspector unavailable: {stage.inspector}",
                        stage.evidence,
                    )
                ]
            elif hasattr(inspector, "inspect") and stage.inspector in {
                "title-body-mismatch",
                "density",
                "deck-semantic",
                "image-text-contradiction",
                "unresolved-render-anomaly",
                "reference-assisted",
                "render-only-geometry",
            }:
                current = [
                    await self._inspect_async(inspector, context, decision.defect_class)
                ]
            else:
                current = await asyncio.to_thread(
                    inspect_safely, inspector, context.artifact
                )
            collected.extend(current)
            may_continue = stage.on_defer == "next" and all(
                item.status in {InspectionStatus.DEFER, InspectionStatus.PASS}
                for item in current
            )
            if not may_continue:
                break
            if index + 1 == len(decision.stages):
                break
        return collected

    async def _inspect_async(
        self, inspector: Any, context: InspectionContext, defect_class: DefectClass
    ) -> InspectionResult:
        if isinstance(inspector, ReferenceInspector):
            return await inspector.inspect(
                context, defect_class, _definition(defect_class)
            )
        return await inspector.inspect(context)

    def _inspector(
        self, name: str, defect_class: DefectClass, context: InspectionContext
    ) -> Any | None:
        symbolic: dict[str, Any] = {
            "geometry.declared_overflow": DeclaredOverflowInspector(),
            "geometry.overlap": OverlapInspector(
                self.config.overlap_tolerance_px, self.config.overlap_min_area_px
            ),
            "geometry.alignment": AlignmentInspector(
                tolerance_px=self.config.alignment_tolerance_px
            ),
            "style.typography": TypographyInspector(
                self.config.typography_tolerance_px
            ),
            "style.brand_color": BrandColorInspector(
                delta_e_threshold=self.config.color_delta_e_threshold
            ),
            "geometry.margin": MarginInspector(self.config.safety_margin_px),
            "geometry.render_overflow": RenderOverflowInspector(),
            "terminology": TerminologyInspector(),
        }
        if name in symbolic:
            return symbolic[name]
        client = (
            self.semantic_client
            if name in {"title-body-mismatch", "density", "deck-semantic"}
            else self.critic_client
        )
        if client is None:
            return None
        if name == "title-body-mismatch":
            return TitleBodyMismatchInspector(client)
        if name == "density":
            return DensityInspector(client)
        if name == "deck-semantic":
            return DeckSemanticInspector(client, defect_class)
        if name == "image-text-contradiction":
            pair = _image_text_ids(context.artifact)
            return ImageTextContradictionInspector(client, *pair) if pair else None
        if name == "unresolved-render-anomaly":
            return RenderAnomalyInspector(client, _first_content_id(context.artifact))
        if name == "reference-assisted":
            return ReferenceInspector(client)
        if name == "render-only-geometry":
            return RenderOnlyGeometryInspector(
                client, defect_class, _RENDER_ONLY_DEFINITIONS.get(defect_class, defect_class.value)
            )
        return None

    @staticmethod
    def _defer(
        defect_class: DefectClass, reason: str, missing: list[str]
    ) -> InspectionResult:
        detail = reason + (f"; missing={sorted(set(missing))}" if missing else "")
        return InspectionResult(
            defect_class=defect_class,
            status=InspectionStatus.DEFER,
            severity=0,
            confidence=1,
            evidence=[Evidence(source=EvidenceSource.DECLARED_IR, detail=detail)],
            repair_hint=RepairHint(
                action="provide_evidence", parameters={"missing": sorted(set(missing))}
            ),
            inspector_name="frozen-router",
            inspector_version="1.0",
        )


def persist_report(
    store: ArtifactStore,
    episode_id: str,
    report: InspectionReport,
    *,
    parent_artifact_id: str,
) -> str:
    """Persist a report as an immutable artifact and return its artifact URI."""
    provenance = Provenance(
        parent_artifact_id=parent_artifact_id,
        creation_action="hybrid_critic",
        versions={
            "router": report.router_version,
            "taxonomy": report.taxonomy_version,
            "router_hash": report.router_hash or "",
        },
    )
    manifest = store.write_artifact(
        episode_id,
        {"critic/report.json": report.model_dump_json(indent=2)},
        provenance,
    )
    return f"artifact://{episode_id}/{manifest.artifact_id}/critic/report.json"


def _report(
    artifact: SlideArtifact,
    results: list[InspectionResult],
    *,
    router_version: str,
    taxonomy_version: str,
    router_hash: str | None = None,
    routes: list[RouteRecord] | None = None,
    conflicts: list[DefectClass] | None = None,
    resolved_status: dict[DefectClass, InspectionStatus] | None = None,
    capability_limits: list[str] | None = None,
) -> InspectionReport:
    counts = {
        status.value: sum(item.status == status for item in results)
        for status in InspectionStatus
    }
    counts["hard_failures"] = sum(
        item.status == InspectionStatus.FAIL and item.severity > 0 for item in results
    )
    return InspectionReport(
        artifact_id=artifact.artifact_id,
        slide_id=artifact.declared_ir.slide_id,
        results=results,
        summary=counts,
        router_version=router_version,
        taxonomy_version=taxonomy_version,
        router_hash=router_hash,
        routes=routes or [],
        conflicts=conflicts or [],
        resolved_status=resolved_status or {},
        capability_limits=capability_limits or [],
    )


_TRUSTED_NATIVE_INSPECTORS = {
    "geometry.declared_overflow",
    "geometry.overlap",
    "geometry.alignment",
    "style.typography",
    "style.brand_color",
    "geometry.margin",
    "geometry.render_overflow",
    "terminology",
}


def _resolve_status(
    results: list[InspectionResult], trust: ArtifactTrust
) -> dict[DefectClass, InspectionStatus]:
    """Resolve action gating while preserving all underlying results."""
    grouped: dict[DefectClass, list[InspectionResult]] = defaultdict(list)
    for item in results:
        grouped[item.defect_class].append(item)
    resolved: dict[DefectClass, InspectionStatus] = {}
    for defect_class, items in grouped.items():
        native = [
            item
            for item in items
            if item.inspector_name in _TRUSTED_NATIVE_INSPECTORS
            and item.status in {InspectionStatus.PASS, InspectionStatus.FAIL}
        ]
        candidates = (
            native
            if trust in {ArtifactTrust.TRUSTED_SOURCE, ArtifactTrust.PARTIAL_NATIVE}
            and native
            else items
        )
        statuses = {item.status for item in candidates}
        resolved[defect_class] = next(
            status
            for status in (
                InspectionStatus.FAIL,
                InspectionStatus.ERROR,
                InspectionStatus.DEFER,
                InspectionStatus.PASS,
                InspectionStatus.NOT_APPLICABLE,
            )
            if status in statuses
        )
    return resolved


def _conflicts(
    results: list[InspectionResult], trust: ArtifactTrust
) -> list[DefectClass]:
    grouped: dict[DefectClass, set[InspectionStatus]] = defaultdict(set)
    for item in results:
        grouped[item.defect_class].add(item.status)
    # Trusted deterministic predicates have priority for action gating, but all
    # results remain in the report; recovered/image inputs never receive priority.
    return sorted(
        (
            key
            for key, values in grouped.items()
            if InspectionStatus.PASS in values and InspectionStatus.FAIL in values
        ),
        key=lambda item: item.value,
    )


def _first_content_id(artifact: SlideArtifact) -> str:
    stack = list(artifact.declared_ir.elements)
    while stack:
        item = stack.pop(0)
        if item.semantic_role not in {"background", "decoration"}:
            return item.element_id
        stack[0:0] = item.children
    return artifact.declared_ir.slide_id


def _image_text_ids(artifact: SlideArtifact) -> tuple[str, str] | None:
    stack = list(artifact.declared_ir.elements)
    image_id = text_id = None
    while stack:
        item = stack.pop(0)
        if image_id is None and (
            item.tag == "img" or item.semantic_role in {"image", "figure"}
        ):
            image_id = item.element_id
        if (
            text_id is None
            and item.text.strip()
            and item.semantic_role not in {"title"}
        ):
            text_id = item.element_id
        stack[0:0] = item.children
    return (image_id, text_id) if image_id and text_id else None


def _definition(defect_class: DefectClass) -> str:
    return {
        DefectClass.G1: "text/container fit that cannot be established from source constraints alone",
        DefectClass.S6: "contradiction between a rendered figure and its associated claim or caption",
    }.get(defect_class, defect_class.value)
