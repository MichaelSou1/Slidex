"""Natural failure corpus admission, persistence, and privacy policy."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .io import content_hash, write_immutable
from .models import Arm, EvaluationCase, EvaluationResult


class NaturalFailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    cluster_id: str
    task_type: str
    model: str
    seed: int
    artifact_lineage: list[str]
    source_package_uri: str | None = None
    task_brief: dict[str, object] | None = None
    render_uris: list[str] = Field(default_factory=list)
    inspection_uri: str | None = None
    human_annotation_uri: str | None = None
    repair_action_uris: list[str] = Field(default_factory=list)
    repaired_artifact_uri: str | None = None
    defects: list[str]
    severities: dict[str, str] = Field(default_factory=dict)
    repair_success: bool | None
    collateral_defects: int
    public: bool


class NaturalFailureCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    corpus_id: str
    source_run_hash: str
    records: list[NaturalFailureRecord]
    summary: dict[str, object]
    corpus_hash: str | None = None


def collect_natural_failures(cases: list[EvaluationCase], result: EvaluationResult) -> list[NaturalFailureRecord]:
    if result.run.arm is not Arm.NO_CRITIC:
        raise ValueError("natural failures must be sampled from the no-critic first artifact")
    case_map = {case.case_id: case for case in cases}
    records = []
    for item in result.results:
        if not item.predicted_defects:
            continue
        case = case_map[item.case_id]
        public = bool(case.metadata.get("redistributable", False)) and not bool(case.metadata.get("sensitive", True))
        records.append(NaturalFailureRecord(
            case_id=item.case_id,
            cluster_id=case.cluster_id,
            task_type=str(case.metadata.get("task_type", "unknown")),
            model=next(iter(result.run.models.values())).model if result.run.models else "unknown",
            seed=result.run.seed,
            artifact_lineage=item.artifact_lineage,
            source_package_uri=case.metadata.get("source_package_uri"),
            task_brief=case.task_brief.model_dump(mode="json") if case.task_brief else None,
            render_uris=list(case.metadata.get("render_uris", [])),
            inspection_uri=case.metadata.get("inspection_uri"),
            human_annotation_uri=case.metadata.get("human_annotation_uri"),
            repair_action_uris=list(case.metadata.get("repair_action_uris", [])),
            repaired_artifact_uri=case.metadata.get("repaired_artifact_uri"),
            defects=item.predicted_defects,
            severities=dict(case.metadata.get("defect_severities", {})),
            repair_success=item.target_removed,
            collateral_defects=item.collateral_defects,
            public=public,
        ))
    return records


def freeze_natural_failure_corpus(path: Path, cases: list[EvaluationCase], result: EvaluationResult) -> NaturalFailureCorpus:
    records = collect_natural_failures(cases, result)
    deck_keys = {(record.case_id, record.seed) for record in records}
    defect_counts = Counter(defect for record in records for defect in set(record.defects))
    cooccurrence = Counter(tuple(sorted(set(record.defects))) for record in records if len(set(record.defects)) > 1)
    repaired = [record for record in records if record.repair_success is not None]
    summary: dict[str, object] = {
        "record_count": len(records),
        "clustered_deck_seed_count": len(deck_keys),
        "defect_counts": dict(sorted(defect_counts.items())),
        "cooccurrence": {"|".join(key): value for key, value in sorted(cooccurrence.items())},
        "repair_success_rate": sum(record.repair_success is True for record in repaired) / len(repaired) if repaired else None,
        "collateral_defects": sum(record.collateral_defects for record in records),
        "public_records": sum(record.public for record in records),
    }
    corpus = NaturalFailureCorpus(corpus_id=result.run.run_id, source_run_hash=result.immutable_hash or "unfrozen", records=records, summary=summary)
    corpus.corpus_hash = content_hash(corpus.model_dump(exclude={"corpus_hash"}, mode="json"))
    write_immutable(path, corpus)
    return corpus
