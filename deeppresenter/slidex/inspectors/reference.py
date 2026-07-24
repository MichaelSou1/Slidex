"""Order-controlled, clean-reference-assisted neural inspection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from deeppresenter.slidex.inspectors.neural import AtomicNeuralClient
from deeppresenter.slidex.models import (
    DefectClass,
    Evidence,
    EvidenceSource,
    InspectionContext,
    InspectionResult,
    InspectionStatus,
    RepairHint,
)


class ReferenceInspector:
    """Compare target/reference in both orders and reject positional disagreement."""

    name = "reference-assisted"
    version = "reference/1.0"

    def __init__(self, client: AtomicNeuralClient) -> None:
        self.client = client

    async def inspect(
        self,
        context: InspectionContext,
        defect_class: DefectClass,
        definition: str,
        *,
        run_clean_control: bool = True,
    ) -> InspectionResult:
        reason = self._compatibility_error(context)
        if reason:
            return self._defer(defect_class, reason)
        target = context.render_path or ""
        reference = context.reference_render_path or ""
        ab, _ = await self.client.compare(defect_class, definition, target, reference, condition="AB")
        ba, _ = await self.client.compare(defect_class, definition, reference, target, condition="BA")
        control = None
        if run_clean_control:
            control, _ = await self.client.compare(defect_class, definition, reference, reference, condition="clean-clean")
        agrees_target_worse = ab.verdict == "right" and ba.verdict == "left"
        agrees_equal = ab.verdict == ba.verdict == "tie"
        control_ok = control is None or control.verdict == "tie"
        details = [f"AB={ab.verdict}", f"BA={ba.verdict}"]
        if control:
            details.append(f"clean-clean={control.verdict}")
        if not control_ok:
            return self._defer(defect_class, "forced_choice_bias_detected", details)
        if agrees_target_worse:
            return InspectionResult(
                defect_class=defect_class, status=InspectionStatus.FAIL, severity=1,
                confidence=min(ab.confidence, ba.confidence),
                evidence=[Evidence(source=EvidenceSource.CLEAN_REFERENCE, detail="; ".join(details))],
                repair_hint=RepairHint(action="match_clean_reference", explanation="Order-controlled comparison identifies the target as worse."),
                inspector_name=self.name, inspector_version=self.version,
                input_hash=hashlib.sha256((target + reference).encode()).hexdigest(),
            )
        if agrees_equal:
            return InspectionResult(
                defect_class=defect_class, status=InspectionStatus.PASS, severity=0,
                confidence=min(ab.confidence, ba.confidence),
                evidence=[Evidence(source=EvidenceSource.CLEAN_REFERENCE, detail="; ".join(details))],
                inspector_name=self.name, inspector_version=self.version,
            )
        return self._defer(defect_class, "positional_disagreement", details)

    @staticmethod
    def _compatibility_error(context: InspectionContext) -> str | None:
        if not context.reference_artifact or not context.reference_render_path:
            return "clean_reference_required"
        if not context.render_path:
            return "target_render_required"
        target = context.artifact.declared_ir
        reference = context.reference_artifact.declared_ir
        if abs(target.page_width / target.page_height - reference.page_width / reference.page_height) > 0.01:
            return "reference_aspect_ratio_mismatch"
        target_render = context.artifact.renders[0] if context.artifact.renders else None
        reference_render = context.reference_artifact.renders[0] if context.reference_artifact.renders else None
        if target_render and reference_render and target_render.renderer.name != reference_render.renderer.name:
            return "reference_renderer_mismatch"
        target_role = target.expected_roles.get(target.slide_id)
        reference_role = reference.expected_roles.get(reference.slide_id)
        if target_role and reference_role and target_role != reference_role:
            return "reference_page_role_mismatch"
        if not Path(context.render_path).is_file() or not Path(context.reference_render_path).is_file():
            return "reference_render_missing"
        return None

    def _defer(self, defect_class: DefectClass, reason: str, details: list[str] | None = None) -> InspectionResult:
        detail = reason + (f": {'; '.join(details)}" if details else "")
        return InspectionResult(
            defect_class=defect_class, status=InspectionStatus.DEFER, severity=0,
            confidence=1, evidence=[Evidence(source=EvidenceSource.CLEAN_REFERENCE, detail=detail)],
            repair_hint=RepairHint(action="provide_compatible_reference", explanation=reason),
            inspector_name=self.name, inspector_version=self.version,
        )
