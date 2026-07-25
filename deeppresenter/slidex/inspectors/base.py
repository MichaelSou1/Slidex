"""Common protocol and result helpers for deterministic Slidex inspectors."""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

from deeppresenter.utils.log import debug

from deeppresenter.slidex.models import (
    DefectClass,
    Evidence,
    InspectionResult,
    InspectionStatus,
    RepairHint,
    SlideArtifact,
)


class Inspector(Protocol):
    """Uniform interface implemented by every Slidex inspector."""

    name: str
    version: str
    defect_class: DefectClass

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]: ...


def input_hash(artifact: SlideArtifact) -> str:
    """Hash the exact strict model input supplied to an inspector."""
    return hashlib.sha256(artifact.model_dump_json().encode()).hexdigest()


def result(
    inspector: Inspector,
    artifact: SlideArtifact,
    status: InspectionStatus,
    *,
    severity: float = 0,
    confidence: float = 1,
    evidence: list[Evidence] | None = None,
    element_ids: list[str] | None = None,
    repair_hint: RepairHint | None = None,
    started_at: float | None = None,
) -> InspectionResult:
    """Construct a complete, replayable inspection result."""
    return InspectionResult(
        defect_class=inspector.defect_class,
        status=status,
        severity=severity,
        confidence=confidence,
        evidence=evidence or [],
        element_ids=element_ids or [],
        repair_hint=repair_hint,
        inspector_name=inspector.name,
        inspector_version=inspector.version,
        input_hash=input_hash(artifact),
        latency_ms=(time.perf_counter() - started_at) * 1000 if started_at else 0,
    )


def inspect_safely(
    inspector: Inspector, artifact: SlideArtifact
) -> list[InspectionResult]:
    """Convert inspector bugs into explicit errors rather than false passes."""
    started = time.perf_counter()
    try:
        results = inspector.inspect(artifact)
        debug(
            "inspector_timing inspector=%s artifact_id=%s latency_ms=%.3f",
            inspector.name,
            artifact.artifact_id,
            (time.perf_counter() - started) * 1000,
        )
        return results
    except Exception as exc:
        debug(
            "inspector_timing inspector=%s artifact_id=%s latency_ms=%.3f status=error",
            inspector.name,
            artifact.artifact_id,
            (time.perf_counter() - started) * 1000,
        )
        return [
            result(
                inspector,
                artifact,
                InspectionStatus.ERROR,
                confidence=0,
                evidence=[],
                started_at=started,
                repair_hint=RepairHint(action="inspect_manually", explanation=str(exc)),
            )
        ]
