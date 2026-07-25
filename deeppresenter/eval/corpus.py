"""Natural failure corpus admission and privacy policy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .models import Arm, EvaluationCase, EvaluationResult


class NaturalFailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    cluster_id: str
    task_type: str
    model: str
    artifact_lineage: list[str]
    defects: list[str]
    repair_success: bool | None
    collateral_defects: int
    public: bool


def collect_natural_failures(
    cases: list[EvaluationCase], result: EvaluationResult
) -> list[NaturalFailureRecord]:
    if result.run.arm is not Arm.NO_CRITIC:
        raise ValueError(
            "natural failures must be sampled from the no-critic first artifact"
        )
    case_map = {case.case_id: case for case in cases}
    records = []
    for item in result.results:
        if not item.predicted_defects:
            continue
        case = case_map[item.case_id]
        public = bool(case.metadata.get("redistributable", False)) and not bool(
            case.metadata.get("sensitive", True)
        )
        records.append(
            NaturalFailureRecord(
                case_id=item.case_id,
                cluster_id=case.cluster_id,
                task_type=str(case.metadata.get("task_type", "unknown")),
                model=next(iter(result.run.models.values())).model
                if result.run.models
                else "unknown",
                artifact_lineage=item.artifact_lineage,
                defects=item.predicted_defects,
                repair_success=item.target_removed,
                collateral_defects=item.collateral_defects,
                public=public,
            )
        )
    return records
