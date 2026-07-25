"""Strict, replayable schemas for the Slidex evaluation harness."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EVAL_SCHEMA_VERSION = "1.0"


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Split(StrEnum):
    DEVELOPMENT = "development"
    PILOT = "pilot"
    SEALED_TEST = "sealed_test"


class Suite(StrEnum):
    INTRINSIC = "intrinsic"
    E2E = "e2e"
    SLIDEAUDIT = "slideaudit"
    NATURAL_FAILURE = "natural_failure"


class Arm(StrEnum):
    NO_CRITIC = "no_critic"
    GENERIC = "generic"
    HYBRID = "hybrid"
    C0 = "c0"
    C0_X10 = "c0_x10"
    C0_PLUS = "c0_plus"
    ATOMIC = "atomic"
    SYMBOLIC_ONLY = "symbolic_only"
    VLM_ONLY = "vlm_only"
    REFERENCE_DISABLED = "reference_disabled"
    MISMATCHED_ROUTER = "mismatched_router"


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    DEFER = "defer"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"
    TIMEOUT = "timeout"
    MISSING = "missing"


class SourceRecord(EvalModel):
    source_id: str
    url: str
    license: str
    revision: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acquired_at: datetime
    local_path: str | None = None
    redistributable: bool = False


class DefectLabel(EvalModel):
    defect_class: str
    defective: bool
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    element_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    evidence_condition: Literal["native_ir", "real_layout", "image_only"] = "native_ir"


class EvaluationCase(EvalModel):
    case_id: str
    parent_deck_id: str
    source_id: str
    split: Split
    input_uri: str
    clean_reference_uri: str | None = None
    labels: list[DefectLabel] = Field(default_factory=list)
    task_brief: dict[str, Any] | None = None
    cluster_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_fingerprint: str | None = None
    integrity_status: Literal["valid", "pixel_diff_zero", "render_error", "invalid"] = (
        "valid"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkManifest(EvalModel):
    schema_version: Literal["1.0"] = EVAL_SCHEMA_VERSION
    benchmark_id: str
    revision: str
    created_at: datetime
    frozen_at: datetime | None = None
    sources: list[SourceRecord]
    cases: list[EvaluationCase]
    taxonomy_version: str = "1.0"
    crosswalk: dict[str, list[str]] = Field(default_factory=dict)
    preparation: dict[str, Any] = Field(default_factory=dict)
    manifest_hash: str | None = None

    @model_validator(mode="after")
    def validate_isolation(self) -> BenchmarkManifest:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        cluster_splits: dict[str, Split] = {}
        parent_splits: dict[str, Split] = {}
        template_splits: dict[str, Split] = {}
        for case in self.cases:
            identities = [
                (case.cluster_id, cluster_splits),
                (case.parent_deck_id, parent_splits),
            ]
            if case.template_fingerprint:
                identities.append((case.template_fingerprint, template_splits))
            for key, mapping in identities:
                previous = mapping.setdefault(key, case.split)
                if previous != case.split:
                    raise ValueError(f"split leakage for {key}")
        return self


class ModelRecord(EvalModel):
    provider: str
    model: str
    sampling: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, bool] = Field(default_factory=dict)


class EvaluationRun(EvalModel):
    schema_version: Literal["1.0"] = EVAL_SCHEMA_VERSION
    run_id: str
    suite: Suite
    arm: Arm
    seed: int
    manifest_hash: str
    config_hash: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    models: dict[str, ModelRecord] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    router_hash: str | None = None
    reward_hash: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    git_commit: str
    initial_artifact_id: str | None = None
    max_repairs: int = 3
    model_budget: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseResult(EvalModel):
    case_id: str
    outcome: Outcome
    predicted_defects: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    element_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    raw_output: Any = None
    artifact_lineage: list[str] = Field(default_factory=list)
    repair_rounds: int = Field(default=0, ge=0)
    target_removed: bool | None = None
    collateral_defects: int = Field(default=0, ge=0)
    export_passed: bool | None = None
    render_fidelity_passed: bool | None = None
    severe_defects: int | None = Field(default=None, ge=0)
    required_facts_retained: bool | None = None
    grounding_score: float | None = Field(default=None, ge=0, le=1)
    tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0, ge=0)
    error: str | None = None
    capability_downgrade: str | None = None


class EvaluationResult(EvalModel):
    schema_version: Literal["1.0"] = EVAL_SCHEMA_VERSION
    run: EvaluationRun
    results: list[CaseResult]
    immutable_hash: str | None = None


class Preregistration(EvalModel):
    schema_version: Literal["1.0"] = EVAL_SCHEMA_VERSION
    research_questions: list[str]
    primary_endpoints: list[str]
    confirmatory_metrics: list[str]
    secondary_metrics: list[str]
    exploratory_metrics: list[str]
    minimum_effects: dict[str, float]
    noninferiority_margins: dict[str, float]
    frozen_config_hashes: dict[str, str]
    git_commit: str
    environment: dict[str, str]
    frozen_at: datetime
    preregistration_hash: str | None = None


def default_cache_dir() -> Path:
    """Return the evaluation cache without creating it."""
    return Path.home() / ".cache" / "deeppresenter" / "eval"
