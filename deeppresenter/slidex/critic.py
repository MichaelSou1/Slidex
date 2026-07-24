"""Model-free symbolic critic orchestration for Phase 4."""

from __future__ import annotations

from collections.abc import Iterable

from deeppresenter.slidex.inspectors import Inspector, inspect_safely
from deeppresenter.slidex.models import (
    InspectionReport,
    InspectionStatus,
    SlideArtifact,
)


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
        counts = {
            status.value: sum(result.status == status for result in results)
            for status in InspectionStatus
        }
        return InspectionReport(
            artifact_id=artifact.artifact_id,
            slide_id=artifact.declared_ir.slide_id,
            results=results,
            summary=counts,
            router_version=self.router_version,
            taxonomy_version=self.taxonomy_version,
        )
