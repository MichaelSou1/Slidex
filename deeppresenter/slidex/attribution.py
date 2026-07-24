"""Experimental A/B/C observation protocol for failure attribution."""

from __future__ import annotations

from typing import Any

from deeppresenter.slidex.inspectors.neural import AtomicNeuralClient
from deeppresenter.slidex.models import (
    AtomicVerdict,
    AttributionLabel,
    DefectClass,
    FailureAttribution,
    InspectionContext,
)


class FailureAttributor:
    """Measure evidence sufficiency without claiming model-internal causality."""

    def __init__(self, client: AtomicNeuralClient) -> None:
        self.client = client

    async def run(
        self,
        context: InspectionContext,
        defect_class: DefectClass,
        definition: str,
        structured_evidence: dict[str, Any],
        *,
        whole_rubric_definition: str | None = None,
        repeated_whole_rubric_budget: int = 0,
    ) -> FailureAttribution:
        if not context.render_path:
            raise ValueError("failure attribution condition A requires a render")
        conditions: dict[str, AtomicVerdict] = {}
        records = []
        conditions["A_image"], record = await self.client.inspect(
            defect_class, definition, {}, image_paths=[context.render_path], condition="A_image"
        )
        records.append(record)
        conditions["B_structured_ir"], record = await self.client.inspect(
            defect_class, definition, structured_evidence, condition="B_structured_ir"
        )
        records.append(record)
        conditions["C_image_ir"], record = await self.client.inspect(
            defect_class, definition, structured_evidence, image_paths=[context.render_path], condition="C_image_ir"
        )
        records.append(record)
        if context.reference_render_path:
            conditions["reference"], record = await self.client.inspect(
                defect_class, definition,
                {**structured_evidence, "reference_role": "clean"},
                image_paths=[context.render_path, context.reference_render_path],
                condition="reference",
            )
            records.append(record)
        whole_rubric: list[AtomicVerdict] = []
        if whole_rubric_definition:
            for index in range(repeated_whole_rubric_budget):
                verdict, record = await self.client.inspect(
                    defect_class, whole_rubric_definition, structured_evidence,
                    image_paths=[context.render_path], condition=f"whole_rubric_{index}",
                )
                whole_rubric.append(verdict)
                records.append(record)
        label, explanation = self._classify(conditions)
        return FailureAttribution(
            defect_class=defect_class, label=label, conditions=conditions,
            whole_rubric=whole_rubric, records=records, explanation=explanation,
        )

    @staticmethod
    def _classify(conditions: dict[str, AtomicVerdict]) -> tuple[AttributionLabel, str]:
        a = conditions["A_image"].verdict
        b = conditions["B_structured_ir"].verdict
        c = conditions["C_image_ir"].verdict
        reference = conditions.get("reference")
        if a != "defer" and a == c:
            return AttributionLabel.IMAGE_SUFFICIENT, "The image-only condition supplied enough observable evidence."
        if a == "defer" and b != "defer" and c == b:
            return AttributionLabel.STRUCTURE_RESCUED, "Trusted structured IR resolved evidence absent from image-only input."
        if b != "defer" and c == "defer":
            return AttributionLabel.FORMAT_SUPPRESSED, "Adding image formatting reduced an otherwise resolvable structured judgment."
        if reference and c == "defer" and reference.verdict != "defer":
            return AttributionLabel.REFERENCE_ASSISTED, "A clean reference supplied the missing comparative evidence."
        return AttributionLabel.UNRESOLVED, "The tested observation formats did not produce a stable operational verdict."
