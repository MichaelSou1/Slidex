"""Strict, replayable schemas for the Slidex evaluation harness."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from deeppresenter.slidex.models import DefectClass

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
    FROZEN_HYBRID = "frozen_hybrid"
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


class ReviewStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    NOT_REQUIRED = "not_required"


class IntegrityStatus(StrEnum):
    VALID = "valid"
    PENDING = "pending"
    PIXEL_DIFF_ZERO = "pixel_diff_zero"
    RENDER_ERROR = "render_error"
    TARGET_RULE_FAILED = "target_rule_failed"
    COLLATERAL_DEFECT = "collateral_defect"
    INVALID = "invalid"


class TaskType(StrEnum):
    ACADEMIC = "academic"
    BUSINESS = "business"
    PRODUCT = "product"
    TEACHING = "teaching"


class LineageRecord(EvalModel):
    artifact_id: str
    kind: str
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_artifact_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_relative_uri(self) -> LineageRecord:
        if Path(self.uri).is_absolute() or self.uri.startswith("file:"):
            raise ValueError("artifact URI must be relative to the evaluation cache")
        return self


class MutationRecord(EvalModel):
    mutation_id: str
    defect_class: DefectClass
    operator: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int
    target_element_id: str | None = None
    reversible: bool = True


class IntegrityRecord(EvalModel):
    pixel_difference: float | None = Field(default=None, ge=0, le=1)
    changed_bbox: tuple[int, int, int, int] | None = None
    changed_pixel_ratio: float | None = Field(default=None, ge=0, le=1)
    perceptual_difference: float | None = Field(default=None, ge=0)
    target_rule_passed: bool | None = None
    clean_rule_passed: bool | None = None
    collateral_high_severity_defects: list[str] = Field(default_factory=list)
    render_environment_hash: str | None = None
    rejection_reason: str | None = None


class HumanReview(EvalModel):
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_version: str | None = None
    reviewed_at: datetime | None = None
    reason: str | None = None
    blinded: bool = True


class PreparationRecord(EvalModel):
    original_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    normalized_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_url: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    converter_versions: dict[str, str] = Field(default_factory=dict)
    renderer_versions: dict[str, str] = Field(default_factory=dict)
    font_environment_hash: str | None = None
    mutation: MutationRecord | None = None
    integrity: IntegrityRecord = Field(default_factory=IntegrityRecord)
    review: HumanReview = Field(default_factory=HumanReview)
    lineage: list[LineageRecord] = Field(default_factory=list)


class RequiredFact(EvalModel):
    fact_id: str
    text: str
    source_locator: str


class TaskBrief(EvalModel):
    audience: str
    purpose: str
    language: str
    page_count: tuple[int, int]
    required_sections: list[str]
    required_facts: list[RequiredFact]
    required_visuals: list[str] = Field(default_factory=list)
    style_constraints: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    acceptable_summarization: str = ""
    automatic_checks: list[str] = Field(default_factory=list)
    human_rating_prompt: str = ""

    @model_validator(mode="after")
    def validate_page_count(self) -> TaskBrief:
        if self.page_count[0] < 1 or self.page_count[0] > self.page_count[1]:
            raise ValueError("invalid task brief page_count")
        if len({fact.fact_id for fact in self.required_facts}) != len(
            self.required_facts
        ):
            raise ValueError("required fact IDs must be unique")
        return self


class CrosswalkEntry(EvalModel):
    source_label: str
    target_labels: list[DefectClass] = Field(default_factory=list)
    rationale: str
    evidence_condition: Literal["native_ir", "real_layout", "image_only"]
    version: str
    reviewed: bool = False

    @model_validator(mode="after")
    def validate_targets(self) -> CrosswalkEntry:
        if not self.source_label.strip():
            raise ValueError("crosswalk source label cannot be empty")
        if len(self.target_labels) != len(set(self.target_labels)):
            raise ValueError("crosswalk target labels must be unique")
        return self


class SourceRecord(EvalModel):
    source_id: str
    url: str
    license: str
    revision: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acquired_at: datetime
    local_path: str | None = None
    redistributable: bool = False
    dataset_id: str | None = None
    upstream_commit: str | None = None
    license_text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    citation: str | None = None

    @model_validator(mode="after")
    def validate_source_contract(self) -> SourceRecord:
        if not self.revision.strip():
            raise ValueError("source revision must be pinned")
        if self.local_path and (
            Path(self.local_path).is_absolute() or self.local_path.startswith("file:")
        ):
            raise ValueError("source local_path must be relative")
        return self


class DefectLabel(EvalModel):
    defect_class: DefectClass
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
    task_brief: TaskBrief | None = None
    cluster_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_fingerprint: str | None = None
    integrity_status: IntegrityStatus = IntegrityStatus.VALID
    preparation_record: PreparationRecord | None = None
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
    crosswalk_entries: list[CrosswalkEntry] = Field(default_factory=list)
    preparation: dict[str, Any] = Field(default_factory=dict)
    manifest_hash: str | None = None

    @model_validator(mode="after")
    def validate_isolation(self) -> BenchmarkManifest:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique")
        known_sources = set(source_ids)
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        cluster_splits: dict[str, Split] = {}
        parent_splits: dict[str, Split] = {}
        template_splits: dict[str, Split] = {}
        text_splits: dict[str, Split] = {}
        image_splits: dict[str, Split] = {}
        for case in self.cases:
            if known_sources and case.source_id not in known_sources:
                raise ValueError(f"unknown source ID: {case.source_id}")
            identities = [
                (case.cluster_id, cluster_splits),
                (case.parent_deck_id, parent_splits),
            ]
            if case.template_fingerprint:
                identities.append((case.template_fingerprint, template_splits))
            text_fingerprint = case.metadata.get("text_fingerprint")
            image_fingerprint = case.metadata.get("image_fingerprint")
            if text_fingerprint:
                identities.append((str(text_fingerprint), text_splits))
            if image_fingerprint:
                identities.append((str(image_fingerprint), image_splits))
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
    judge_model: ModelRecord | None = None
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
    task_constraints_passed: bool | None = None
    section_coverage: float | None = Field(default=None, ge=0, le=1)
    page_count_passed: bool | None = None
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
    taxonomy_version: str = "1.0"
    split_policy: dict[str, str] = Field(
        default_factory=lambda: {
            "development": "threshold and router development only",
            "pilot": "harness validation only",
            "sealed_test": "confirmatory evaluation only",
        }
    )
    exclusion_policy: list[str] = Field(
        default_factory=lambda: [
            "license or provenance unavailable",
            "render or export failed",
            "mutation has zero pixel difference",
            "target rule not established",
            "new non-target high-severity defect introduced",
        ]
    )
    error_policy: str = "defer, error, timeout, and missing are never converted to pass"
    preregistration_hash: str | None = None


def default_cache_dir() -> Path:
    """Return the evaluation cache without creating it."""
    return Path.home() / ".cache" / "deeppresenter" / "eval"
