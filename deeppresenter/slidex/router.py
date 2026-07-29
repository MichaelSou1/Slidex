"""Frozen, evidence-only routing for the Slidex hybrid critic."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field

from deeppresenter.slidex.models import (
    ArtifactTrust,
    DefectClass,
    InspectionContext,
    SlidexModel,
)


class EvidenceAvailability(SlidexModel):
    available: set[str] = Field(default_factory=set)
    missing_bookkeeping: list[str] = Field(default_factory=list)

    @classmethod
    def from_context(cls, context: InspectionContext) -> "EvidenceAvailability":
        available = {"declared_ir"}
        if context.artifact.computed_ir and context.artifact.computed_ir.render_ready:
            available.add("computed_ir")
        if context.render_path:
            available.add("render")
        if context.reference_artifact and context.reference_render_path:
            available.add("clean_reference")
        if (
            context.deck_outline
            or context.slide_summaries
            or context.artifact.declared_ir.elements
        ):
            available.add("deck_text")
        return cls(
            available=available,
            missing_bookkeeping=context.artifact.missing_bookkeeping,
        )


class RouteStage(SlidexModel):
    inspector: str
    evidence: list[str]
    on_defer: Literal["stop", "next"] = "stop"


class RouteDecision(SlidexModel):
    defect_class: DefectClass
    stages: list[RouteStage] = Field(default_factory=list)
    reason: str
    missing_evidence: list[str] = Field(default_factory=list)
    capability_limit: str | None = None


class FrozenRouterConfig(SlidexModel):
    """Machine-readable v1 mapping frozen before inspection."""

    router_version: Literal["1.0"] = "1.0"
    taxonomy_version: Literal["1.0"] = "1.0"
    reference_policy: Literal["never", "on_defer", "always"] = "on_defer"
    mapping: dict[DefectClass, list[str]] = Field(
        default_factory=lambda: {
            DefectClass.G1: ["geometry.declared_overflow", "reference-assisted"],
            DefectClass.G2: ["geometry.overlap"],
            DefectClass.G3: ["geometry.alignment"],
            DefectClass.G4: ["style.typography"],
            DefectClass.G5: ["style.brand_color"],
            DefectClass.G6: ["geometry.margin"],
            DefectClass.G7: ["geometry.render_overflow", "unresolved-render-anomaly"],
            DefectClass.S1: ["title-body-mismatch"],
            DefectClass.S2: ["deck-semantic"],
            DefectClass.S3: ["terminology"],
            DefectClass.S4: ["density"],
            DefectClass.S5: ["deck-semantic"],
            DefectClass.S6: ["image-text-contradiction", "reference-assisted"],
        }
    )

    @property
    def config_hash(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode()).hexdigest()


class FrozenCriticRouter:
    """Select inspectors using only class, evidence, trust, and frozen config."""

    def __init__(self, config: FrozenRouterConfig | None = None) -> None:
        self.config = config or FrozenRouterConfig()

    def route(
        self,
        defect_class: DefectClass,
        evidence: EvidenceAvailability,
        trust: ArtifactTrust,
    ) -> RouteDecision:
        available = evidence.available
        missing: list[str] = []
        limit: str | None = None
        if trust == ArtifactTrust.PARTIAL_NATIVE:
            missing.extend(evidence.missing_bookkeeping)
            limit = "Third-party native XML is partially trusted; missing bookkeeping is recorded."
        elif trust == ArtifactTrust.RECOVERED:
            limit = "Recovered boxes are untrusted; symbolic results do not carry native-IR guarantees."

        if trust == ArtifactTrust.IMAGE_ONLY:
            limit = "Image-only input has no native-IR guarantees; only neural predicates are available."
            return self._image_only(defect_class, available, limit)

        if defect_class in {
            DefectClass.G2,
            DefectClass.G3,
            DefectClass.G4,
            DefectClass.G5,
            DefectClass.G6,
        }:
            inspector = self.config.mapping[defect_class][0]
            source = "computed_ir"
            if source not in available:
                missing.append(source)
            return RouteDecision(
                defect_class=defect_class,
                stages=[RouteStage(inspector=inspector, evidence=[source])],
                reason="Frozen native-structure predicate mapping.",
                missing_evidence=missing,
                capability_limit=limit,
            )

        if defect_class == DefectClass.S3:
            return RouteDecision(
                defect_class=defect_class,
                stages=[RouteStage(inspector="terminology", evidence=["deck_text"])],
                reason="Cross-slide terminology matching is deterministic and was not rescued by neural elicitation.",
                missing_evidence=[] if "deck_text" in available else ["deck_text"],
            )

        if defect_class == DefectClass.G7:
            stages = [
                RouteStage(
                    inspector="geometry.render_overflow",
                    evidence=["computed_ir"],
                    on_defer="next",
                )
            ]
            if "render" in available:
                stages.append(
                    RouteStage(
                        inspector="unresolved-render-anomaly", evidence=["render"]
                    )
                )
            else:
                missing.append("render")
            return RouteDecision(
                defect_class=defect_class,
                stages=stages,
                reason="DOM overflow is authoritative; atomic vision is permitted only after deterministic defer.",
                missing_evidence=missing,
            )

        if defect_class == DefectClass.G1:
            stages = [
                RouteStage(
                    inspector="geometry.declared_overflow",
                    evidence=["declared_ir"],
                    on_defer="next",
                )
            ]
            self._append_reference(stages, available, missing)
            return RouteDecision(
                defect_class=defect_class,
                stages=stages,
                reason="Use source/computed constraints first; ambiguous single-page cases follow the frozen reference policy.",
                missing_evidence=missing,
            )

        if defect_class == DefectClass.S4:
            return RouteDecision(
                defect_class=defect_class,
                stages=[
                    RouteStage(
                        inspector="density", evidence=["computed_ir", "declared_ir"]
                    )
                ],
                reason="Density uses deterministic statistics before one semantic boundary judgment.",
            )

        if defect_class in {DefectClass.S2, DefectClass.S5}:
            return RouteDecision(
                defect_class=defect_class,
                stages=[RouteStage(inspector="deck-semantic", evidence=["deck_text"])],
                reason="Narrative order and required-step completeness are deck-level semantic predicates.",
                missing_evidence=[] if "deck_text" in available else ["deck_text"],
            )

        if defect_class == DefectClass.S6:
            stages = [
                RouteStage(
                    inspector="image-text-contradiction",
                    evidence=["render", "declared_ir"],
                    on_defer="next",
                )
            ]
            self._append_reference(stages, available, missing)
            return RouteDecision(
                defect_class=defect_class,
                stages=stages,
                reason="Atomic image-text contradiction precedes an explicitly policy-gated clean reference.",
                missing_evidence=missing,
            )

        return RouteDecision(
            defect_class=defect_class,
            stages=[
                RouteStage(
                    inspector="title-body-mismatch", evidence=["declared_ir", "render"]
                )
            ],
            reason="S1 is a direct atomic semantic predicate.",
            missing_evidence=[] if "declared_ir" in available else ["declared_ir"],
        )

    def _append_reference(
        self,
        stages: list[RouteStage],
        available: set[str],
        missing: list[str],
    ) -> None:
        if self.config.reference_policy == "never":
            return
        if "clean_reference" in available:
            stages.append(
                RouteStage(inspector="reference-assisted", evidence=["clean_reference"])
            )
        else:
            missing.append("clean_reference")

    def _image_only(
        self,
        defect_class: DefectClass,
        available: set[str],
        limit: str,
    ) -> RouteDecision:
        inspector = {
            DefectClass.G7: "unresolved-render-anomaly",
            DefectClass.S1: "title-body-mismatch",
            DefectClass.S4: "density",
            DefectClass.S6: "image-text-contradiction",
        }.get(defect_class)
        if inspector and "render" in available:
            return RouteDecision(
                defect_class=defect_class,
                stages=[RouteStage(inspector=inspector, evidence=["render"])],
                reason="Open-world image-only downgrade to an atomic VLM predicate.",
                capability_limit=limit,
            )
        # G1-G6, S2, S3, S5 normally need computed_ir/native XML; on image-only
        # trust, fall back to one atomic VLM query over the render instead of
        # deferring unconditionally, matching the disclosed capability_limit.
        if "render" in available:
            return RouteDecision(
                defect_class=defect_class,
                stages=[RouteStage(inspector="render-only-geometry", evidence=["render"])],
                reason="Open-world image-only downgrade to an atomic VLM predicate.",
                capability_limit=limit,
            )
        return RouteDecision(
            defect_class=defect_class,
            reason="No reliable inspector is available without native structure.",
            missing_evidence=["trusted_native_ir"],
            capability_limit=limit,
        )
