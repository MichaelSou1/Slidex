"""Deterministic, hard-gated rewards derived from persisted Slidex evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, computed_field, model_validator

from deeppresenter.slidex.models import (
    DefectClass,
    DefectTransition,
    ExportManifest,
    FinalArtifactStatus,
    InspectionReport,
    InspectionResult,
    InspectionStatus,
    PolicyViolation,
    RepairAction,
    RepairExecutionStatus,
    RepairOperation,
    SlidexModel,
)
from deeppresenter.slidex.repair import inspection_result_id

_GEOMETRY = tuple(DefectClass(f"G{index}") for index in range(1, 8))
_SEMANTIC = tuple(DefectClass(f"S{index}") for index in range(1, 7))
_HARD_GEOMETRY = {DefectClass.G1, DefectClass.G2, DefectClass.G6, DefectClass.G7}


class EvaluationSplit(StrEnum):
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class ClassReward(SlidexModel):
    defect_class: DefectClass
    status: InspectionStatus
    severity: float = Field(ge=0, le=1)
    score: float = Field(ge=-1, le=1)
    inspection_result_ids: list[str] = Field(default_factory=list)


class DefectReward(SlidexModel):
    classes: dict[DefectClass, ClassReward] = Field(default_factory=dict)
    score: float = Field(ge=-1, le=1)
    coverage: float = Field(ge=0, le=1)
    pass_count: int = Field(ge=0)
    fail_count: int = Field(ge=0)
    defer_count: int = Field(ge=0)
    error_count: int = Field(ge=0)


class ValiditySignals(SlidexModel):
    source_valid: bool = True
    browser_valid: bool = True


class ValidityReward(SlidexModel):
    source_valid: bool
    browser_valid: bool
    export_valid: bool
    rerender_valid: bool
    score: float = Field(ge=-1, le=1)
    terminal: bool
    evidence_ids: list[str] = Field(default_factory=list)


class FidelityReward(SlidexModel):
    available: bool
    score: float = Field(ge=-1, le=1)
    page_count_matches: bool | None = None
    failed_pages: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class TaskOutcome(SlidexModel):
    actual_page_count: int = Field(ge=0)
    expected_page_count: int | None = Field(default=None, ge=0)
    outline_checks: dict[str, bool] = Field(default_factory=dict)
    required_content: dict[str, bool] = Field(default_factory=dict)
    user_constraints: dict[str, bool] = Field(default_factory=dict)


class TaskReward(SlidexModel):
    score: float = Field(ge=-1, le=1)
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    failed_requirements: list[str] = Field(default_factory=list)


class EfficiencyUsage(SlidexModel):
    tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    repair_steps: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)


class EfficiencyReward(SlidexModel):
    score: float = Field(ge=-1, le=0)
    usage: EfficiencyUsage
    normalized_costs: dict[str, float] = Field(default_factory=dict)


class RepairDeltaReward(SlidexModel):
    score: float = Field(ge=-1, le=1)
    potential_before: float = Field(ge=-1, le=0)
    potential_after: float = Field(ge=-1, le=0)
    fail_delta: int
    severity_delta: float
    targeted_resolved: list[DefectClass] = Field(default_factory=list)
    introduced_defects: list[DefectClass] = Field(default_factory=list)
    transitions: list[DefectTransition] = Field(default_factory=list)
    deterministic_steps: int = Field(ge=0)
    policy_steps: int = Field(ge=0)


class RewardVector(SlidexModel):
    validity_reward: ValidityReward
    geometry_reward: DefectReward
    semantic_reward: DefectReward
    fidelity_reward: FidelityReward
    task_reward: TaskReward
    efficiency_reward: EfficiencyReward
    policy_violation_penalty: float = Field(ge=-1, le=0)
    repair_delta_reward: RepairDeltaReward | None = None


class RewardExplanation(SlidexModel):
    reward_id: str
    artifact_ids: list[str]
    critic_versions: list[str]
    reward_version: str
    reward_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregate: float
    vector: RewardVector
    weights: dict[str, float]
    formula: str
    gate_reasons: list[str] = Field(default_factory=list)
    component_inspection_ids: dict[str, list[str]] = Field(default_factory=dict)
    coverage: dict[str, float] = Field(default_factory=dict)
    episode_invalid: bool = False


class RewardConfig(SlidexModel):
    """Frozen aggregation policy; changing any threshold changes its hash."""

    reward_version: str = "1.0"
    critic_version: str = "1.0"
    terminal_hard_negative: float = Field(default=-1, ge=-1, le=0)
    inspector_error_penalty: float = Field(default=0.2, ge=0, le=1)
    inspector_error_invalidation_count: int = Field(default=2, ge=1)
    severe_defect_threshold: float = Field(default=0.7, ge=0, le=1)
    policy_penalty_per_severity: float = Field(default=0.25, ge=0, le=1)
    new_defect_penalty: float = Field(default=0.25, ge=0, le=1)
    targeted_repair_bonus: float = Field(default=0.1, ge=0, le=1)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "validity": 0.20,
            "geometry": 0.25,
            "semantic": 0.20,
            "fidelity": 0.15,
            "task": 0.20,
        }
    )
    efficiency_budgets: EfficiencyUsage = Field(
        default_factory=lambda: EfficiencyUsage(
            tokens=100_000,
            model_calls=20,
            tool_calls=100,
            repair_steps=10,
            latency_ms=600_000,
        )
    )

    @model_validator(mode="after")
    def validate_weights(self) -> RewardConfig:
        expected = {"validity", "geometry", "semantic", "fidelity", "task"}
        if set(self.weights) != expected or any(value < 0 for value in self.weights.values()):
            raise ValueError(f"weights must contain non-negative values for {sorted(expected)}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("at least one reward weight must be positive")
        return self

    @classmethod
    def from_slidex_config(cls, config: Any) -> RewardConfig:
        """Build the frozen reward policy from the application configuration."""
        return cls(
            reward_version=config.reward_version,
            critic_version=f"{config.taxonomy_version}/{config.router_version}",
            terminal_hard_negative=config.reward_terminal_hard_negative,
            inspector_error_penalty=config.reward_inspector_error_penalty,
            inspector_error_invalidation_count=config.reward_inspector_error_invalidation_count,
            severe_defect_threshold=config.reward_severe_defect_threshold,
            policy_penalty_per_severity=config.reward_policy_penalty_per_severity,
        )

    @computed_field
    @property
    def config_hash(self) -> str:
        payload = self.model_dump_json(exclude={"config_hash"})
        return hashlib.sha256(payload.encode()).hexdigest()

class RewardEngine:
    """Compute replayable rewards without invoking inspectors or models."""

    FORMULA = (
        "gate(valid export) ? weighted(validity, geometry, semantic, fidelity, task) "
        "+ efficiency + policy + repair_delta - reliability : terminal_hard_negative"
    )

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        reports: Sequence[InspectionReport],
        *,
        artifact_ids: Sequence[str] = (),
        export: ExportManifest | None = None,
        validity: ValiditySignals | None = None,
        task: TaskOutcome | None = None,
        usage: EfficiencyUsage | None = None,
        policy_violations: Sequence[PolicyViolation] = (),
        before_reports: Sequence[InspectionReport] = (),
        repair_actions: Sequence[RepairAction] = (),
    ) -> RewardExplanation:
        """Aggregate only supplied, persisted evidence; this method performs no I/O."""
        all_violations = [
            *policy_violations,
            *(violation for report in reports for violation in report.policy_violations),
        ]
        result_ids = {
            inspection_result_id(result)
            for report in reports
            for result in report.results
        }
        geometry = _defect_reward(reports, _GEOMETRY)
        semantic = _defect_reward(reports, _SEMANTIC)
        validity_reward = _validity_reward(reports, export, validity)
        fidelity = _fidelity_reward(export)
        task_reward = _task_reward(task, export)
        efficiency = _efficiency_reward(usage or EfficiencyUsage(), self.config)
        policy_penalty = -min(
            1,
            sum(item.severity for item in all_violations)
            * self.config.policy_penalty_per_severity,
        )
        repair_delta = (
            _repair_delta(before_reports, reports, repair_actions, self.config)
            if before_reports
            else None
        )
        vector = RewardVector(
            validity_reward=validity_reward,
            geometry_reward=geometry,
            semantic_reward=semantic,
            fidelity_reward=fidelity,
            task_reward=task_reward,
            efficiency_reward=efficiency,
            policy_violation_penalty=policy_penalty,
            repair_delta_reward=repair_delta,
        )
        gate_reasons = _gate_reasons(vector, reports, export, all_violations, self.config)
        errors = geometry.error_count + semantic.error_count
        episode_invalid = errors >= self.config.inspector_error_invalidation_count
        aggregate = self._aggregate(vector, gate_reasons, errors)
        ids = sorted(set(artifact_ids) | {report.artifact_id for report in reports})
        critic_versions = sorted(
            {f"{report.taxonomy_version}/{report.router_version}" for report in reports}
        )
        component_ids = {
            "geometry_reward": sorted(
                item for value in geometry.classes.values() for item in value.inspection_result_ids
            ),
            "semantic_reward": sorted(
                item for value in semantic.classes.values() for item in value.inspection_result_ids
            ),
            "validity_reward": sorted(result_ids),
            "fidelity_reward": fidelity.evidence_ids,
        }
        payload = {
            "artifacts": ids,
            "critic_versions": critic_versions,
            "reward_config_hash": self.config.config_hash,
            "vector": vector.model_dump(mode="json"),
            "aggregate": aggregate,
            "gates": gate_reasons,
        }
        reward_id = "reward-" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        return RewardExplanation(
            reward_id=reward_id,
            artifact_ids=ids,
            critic_versions=critic_versions,
            reward_version=self.config.reward_version,
            reward_config_hash=self.config.config_hash,
            aggregate=aggregate,
            vector=vector,
            weights=self.config.weights,
            formula=self.FORMULA,
            gate_reasons=gate_reasons,
            component_inspection_ids=component_ids,
            coverage={"geometry": geometry.coverage, "semantic": semantic.coverage},
            episode_invalid=episode_invalid,
        )

    def _aggregate(
        self, vector: RewardVector, gate_reasons: Sequence[str], error_count: int
    ) -> float:
        if "invalid_export" in gate_reasons:
            return self.config.terminal_hard_negative
        values = {
            "validity": vector.validity_reward.score,
            "geometry": vector.geometry_reward.score,
            "semantic": vector.semantic_reward.score,
            "fidelity": vector.fidelity_reward.score,
            "task": vector.task_reward.score,
        }
        weighted = sum(self.config.weights[key] * values[key] for key in values)
        weighted /= sum(self.config.weights.values())
        if "aesthetic_bonus_withheld" in gate_reasons:
            weighted = min(weighted, 0)
        reliability = min(1, error_count * self.config.inspector_error_penalty)
        delta = vector.repair_delta_reward.score if vector.repair_delta_reward else 0
        aggregate = (
            weighted
            + vector.efficiency_reward.score
            + vector.policy_violation_penalty
            + delta
            - reliability
        )
        return round(max(-1, min(1, aggregate)), 6)


def explain_reward(reward: RewardExplanation) -> dict[str, Any]:
    """Return the complete machine-readable audit explanation."""
    return reward.model_dump(mode="json")


def persist_reward(
    store: Any, episode_id: str, reward: RewardExplanation
) -> str:
    """Persist an immutable reward explanation with versioned provenance."""
    from deeppresenter.slidex.models import Provenance

    manifest = store.write_artifact(
        episode_id,
        {"reward/reward.json": reward.model_dump_json(indent=2)},
        Provenance(
            parent_artifact_id=reward.artifact_ids[-1] if reward.artifact_ids else None,
            creation_action="reward_aggregation",
            versions={
                "reward": reward.reward_version,
                "reward_config_hash": reward.reward_config_hash,
                "critic": ",".join(reward.critic_versions),
            },
        ),
    )
    return f"artifact://{episode_id}/{manifest.artifact_id}/reward/reward.json"

def _defect_reward(
    reports: Sequence[InspectionReport], classes: Sequence[DefectClass]
) -> DefectReward:
    grouped: dict[DefectClass, list[tuple[str, InspectionResult]]] = {
        defect: [] for defect in classes
    }
    for report in reports:
        for result in report.results:
            if result.defect_class in grouped:
                grouped[result.defect_class].append(
                    (inspection_result_id(result), result)
                )
    resolved: dict[DefectClass, ClassReward] = {}
    for defect_class, findings in grouped.items():
        status, severity = _resolve_findings(findings)
        resolved[defect_class] = ClassReward(
            defect_class=defect_class,
            status=status,
            severity=severity,
            score=_status_score(status, severity),
            inspection_result_ids=sorted({item[0] for item in findings}),
        )
    counts = Counter(item.status for item in resolved.values())
    covered = counts[InspectionStatus.PASS] + counts[InspectionStatus.FAIL]
    decisive = [item.score for item in resolved.values() if item.status in {InspectionStatus.PASS, InspectionStatus.FAIL}]
    return DefectReward(
        classes=resolved,
        score=round(sum(decisive) / len(classes), 6),
        coverage=covered / len(classes),
        pass_count=counts[InspectionStatus.PASS],
        fail_count=counts[InspectionStatus.FAIL],
        defer_count=counts[InspectionStatus.DEFER] + counts[InspectionStatus.NOT_APPLICABLE],
        error_count=counts[InspectionStatus.ERROR],
    )


def _resolve_findings(
    findings: Sequence[tuple[str, InspectionResult]],
) -> tuple[InspectionStatus, float]:
    if not findings:
        return InspectionStatus.DEFER, 0
    statuses = [item.status for _, item in findings]
    if InspectionStatus.ERROR in statuses:
        return InspectionStatus.ERROR, 0
    failures = [item for _, item in findings if item.status == InspectionStatus.FAIL]
    if failures:
        return InspectionStatus.FAIL, max(item.severity for item in failures)
    if InspectionStatus.PASS in statuses:
        return InspectionStatus.PASS, 0
    if InspectionStatus.DEFER in statuses:
        return InspectionStatus.DEFER, 0
    return InspectionStatus.NOT_APPLICABLE, 0


def _status_score(status: InspectionStatus, severity: float) -> float:
    if status == InspectionStatus.PASS:
        return 1
    if status == InspectionStatus.FAIL:
        return -severity
    return 0


def _validity_reward(
    reports: Sequence[InspectionReport],
    export: ExportManifest | None,
    signals: ValiditySignals | None,
) -> ValidityReward:
    source = signals.source_valid if signals else bool(reports) and all(report.artifact_id for report in reports)
    inferred_browser = bool(reports) and all(
        not any("render readiness" in limit.lower() for limit in report.capability_limits)
        for report in reports
    )
    browser = signals.browser_valid if signals else inferred_browser
    export_valid = export is not None and export.status in {
        FinalArtifactStatus.PPTX_EXPORTED,
        FinalArtifactStatus.PPTX_RENDER_VALIDATED,
    }
    rerender = export is not None and export.status == FinalArtifactStatus.PPTX_RENDER_VALIDATED
    flags = [source, browser, export_valid, rerender]
    terminal = export is not None and (
        export.hard_penalty
        or export.status in {FinalArtifactStatus.INVALID_ARTIFACT, FinalArtifactStatus.CAPABILITY_ERROR}
    )
    score = -1 if terminal else sum(flags) / len(flags)
    evidence = [export.export_id] if export else []
    return ValidityReward(
        source_valid=source,
        browser_valid=browser,
        export_valid=export_valid,
        rerender_valid=rerender,
        score=score,
        terminal=terminal,
        evidence_ids=evidence,
    )


def _fidelity_reward(export: ExportManifest | None) -> FidelityReward:
    report = export.fidelity_report if export else None
    if report is None:
        return FidelityReward(available=False, score=0)
    failed = [item.slide_id for item in report.page_results if not item.passed]
    scores = [
        (item.perceptual_similarity + item.text_presence + (1 - item.pixel_difference)) / 3
        for item in report.page_results
    ]
    score = sum(scores) / len(scores) if scores else 0
    if report.export_fidelity_failure:
        score = min(0, score - 1)
    return FidelityReward(
        available=True,
        score=round(score, 6),
        page_count_matches=report.page_count_matches,
        failed_pages=failed,
        evidence_ids=[export.export_id] if export else [],
    )


def _task_reward(task: TaskOutcome | None, export: ExportManifest | None) -> TaskReward:
    if task is None:
        return TaskReward(score=0, passed=0, total=0)
    checks: dict[str, bool] = {}
    if task.expected_page_count is not None:
        checks["page_count"] = task.actual_page_count == task.expected_page_count
    checks.update({f"outline:{key}": value for key, value in task.outline_checks.items()})
    checks.update({f"content:{key}": value for key, value in task.required_content.items()})
    checks.update({f"constraint:{key}": value for key, value in task.user_constraints.items()})
    passed = sum(checks.values())
    total = len(checks)
    return TaskReward(
        score=(2 * passed / total - 1) if total else 0,
        passed=passed,
        total=total,
        failed_requirements=sorted(key for key, value in checks.items() if not value),
    )


def _efficiency_reward(usage: EfficiencyUsage, config: RewardConfig) -> EfficiencyReward:
    normalized: dict[str, float] = {}
    for name in type(usage).model_fields:
        value = getattr(usage, name)
        budget = getattr(config.efficiency_budgets, name)
        normalized[name] = min(1, value / budget) if budget else (1 if value else 0)
    return EfficiencyReward(
        score=-round(sum(normalized.values()) / len(normalized), 6),
        usage=usage,
        normalized_costs=normalized,
    )

def _repair_delta(
    before: Sequence[InspectionReport],
    after: Sequence[InspectionReport],
    actions: Sequence[RepairAction],
    config: RewardConfig,
) -> RepairDeltaReward:
    before_state = _report_state(before)
    after_state = _report_state(after)
    transitions: list[DefectTransition] = []
    for defect_class in (*_GEOMETRY, *_SEMANTIC):
        old_status, old_severity = before_state[defect_class]
        new_status, new_severity = after_state[defect_class]
        old_value = _defect_potential(old_status, old_severity)
        new_value = _defect_potential(new_status, new_severity)
        transition: Literal["improved", "unchanged", "worsened"] = (
            "improved" if new_value > old_value else "worsened" if new_value < old_value else "unchanged"
        )
        transitions.append(
            DefectTransition(
                defect_class=defect_class,
                before=old_status,
                after=new_status,
                before_severity=old_severity,
                after_severity=new_severity,
                transition=transition,
            )
        )
    before_fail = sum(status == InspectionStatus.FAIL for status, _ in before_state.values())
    after_fail = sum(status == InspectionStatus.FAIL for status, _ in after_state.values())
    before_severity = sum(severity for status, severity in before_state.values() if status == InspectionStatus.FAIL)
    after_severity = sum(severity for status, severity in after_state.values() if status == InspectionStatus.FAIL)
    targeted = {
        result.defect_class
        for report in before
        for result in report.results
        if result.status == InspectionStatus.FAIL
        and inspection_result_id(result)
        in {item for action in actions for item in action.source_inspection_ids}
    }
    targeted_resolved = sorted(
        defect for defect in targeted if before_state[defect][0] == InspectionStatus.FAIL and after_state[defect][0] == InspectionStatus.PASS
    )
    introduced = sorted(
        defect for defect in (*_GEOMETRY, *_SEMANTIC)
        if before_state[defect][0] != InspectionStatus.FAIL and after_state[defect][0] == InspectionStatus.FAIL
    )
    potential_before = _potential(before_state)
    potential_after = _potential(after_state)
    score = (
        potential_after
        - potential_before
        + config.targeted_repair_bonus * len(targeted_resolved)
        - config.new_defect_penalty * len(introduced)
    )
    applied = [item for item in actions if item.status == RepairExecutionStatus.APPLIED]
    return RepairDeltaReward(
        score=round(max(-1, min(1, score)), 6),
        potential_before=potential_before,
        potential_after=potential_after,
        fail_delta=before_fail - after_fail,
        severity_delta=round(before_severity - after_severity, 6),
        targeted_resolved=targeted_resolved,
        introduced_defects=introduced,
        transitions=transitions,
        deterministic_steps=sum(item.operation != RepairOperation.POLICY_EDIT for item in applied),
        policy_steps=sum(item.operation == RepairOperation.POLICY_EDIT for item in actions),
    )


def _report_state(
    reports: Sequence[InspectionReport],
) -> dict[DefectClass, tuple[InspectionStatus, float]]:
    grouped: dict[DefectClass, list[tuple[str, InspectionResult]]] = {
        defect: [] for defect in DefectClass
    }
    for report in reports:
        for result in report.results:
            grouped[result.defect_class].append((inspection_result_id(result), result))
    return {defect: _resolve_findings(findings) for defect, findings in grouped.items()}


def _defect_potential(status: InspectionStatus, severity: float) -> float:
    if status == InspectionStatus.FAIL:
        return -severity
    if status == InspectionStatus.ERROR:
        return -1
    return 0


def _potential(state: dict[DefectClass, tuple[InspectionStatus, float]]) -> float:
    values = [_defect_potential(status, severity) for status, severity in state.values()]
    return round(sum(values) / len(values), 6)


def _gate_reasons(
    vector: RewardVector,
    reports: Sequence[InspectionReport],
    export: ExportManifest | None,
    violations: Sequence[PolicyViolation],
    config: RewardConfig,
) -> list[str]:
    reasons: list[str] = []
    if vector.validity_reward.terminal:
        reasons.append("invalid_export")
    severe = any(
        result.defect_class in _HARD_GEOMETRY
        and result.status == InspectionStatus.FAIL
        and result.severity >= config.severe_defect_threshold
        for report in reports
        for result in report.results
    )
    missing_asset = bool(
        export
        and export.fidelity_report
        and any(item.missing_images for item in export.fidelity_report.page_results)
    )
    if severe or missing_asset:
        reasons.append("aesthetic_bonus_withheld")
    if vector.geometry_reward.error_count + vector.semantic_reward.error_count:
        reasons.append("inspector_reliability_penalty")
    if violations:
        reasons.append("policy_violation")
    return reasons


class CalibrationExample(SlidexModel):
    pair_id: str
    defect_class: DefectClass
    expected_defective: bool
    predicted_status: InspectionStatus
    localized: bool = False
    split: EvaluationSplit
    provider: str | None = None
    model: str | None = None
    order: Literal["AB", "BA", "clean_clean"] | None = None
    repeat: int = Field(default=0, ge=0)


class CalibrationMetrics(SlidexModel):
    count: int = Field(ge=0)
    recall: float = Field(ge=0, le=1)
    specificity: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    localization_recall: float = Field(ge=0, le=1)
    defer_rate: float = Field(ge=0, le=1)
    error_rate: float = Field(ge=0, le=1)


class CalibrationReport(SlidexModel):
    reward_version: str
    split: EvaluationSplit
    overall: CalibrationMetrics
    by_defect_class: dict[DefectClass, CalibrationMetrics]
    by_neural_executor: dict[str, CalibrationMetrics]
    clean_clean_false_positive_rate: float = Field(ge=0, le=1)
    order_disagreement_rate: float = Field(ge=0, le=1)
    repeat_disagreement_rate: float = Field(ge=0, le=1)


class RewardCalibrator:
    """Evaluate frozen checkers on one declared split without changing config."""

    def __init__(self, config: RewardConfig, *, frozen: bool = True) -> None:
        if not frozen:
            raise ValueError("reward configuration must be frozen before calibration")
        self.config = config

    def evaluate(
        self, examples: Sequence[CalibrationExample], *, split: EvaluationSplit
    ) -> CalibrationReport:
        if not examples or any(item.split != split for item in examples):
            raise ValueError("all calibration examples must belong to the requested split")
        return CalibrationReport(
            reward_version=self.config.reward_version,
            split=split,
            overall=_metrics(examples),
            by_defect_class={
                defect: _metrics([item for item in examples if item.defect_class == defect])
                for defect in sorted({item.defect_class for item in examples})
            },
            by_neural_executor={
                key: _metrics([item for item in examples if f"{item.provider}/{item.model}" == key])
                for key in sorted(
                    {f"{item.provider}/{item.model}" for item in examples if item.provider and item.model}
                )
            },
            clean_clean_false_positive_rate=_clean_clean_bias(examples),
            order_disagreement_rate=_group_disagreement(examples, "order"),
            repeat_disagreement_rate=_group_disagreement(examples, "repeat"),
        )

def _metrics(examples: Sequence[CalibrationExample]) -> CalibrationMetrics:
    positives = [item for item in examples if item.expected_defective]
    negatives = [item for item in examples if not item.expected_defective]
    true_positive = sum(item.predicted_status == InspectionStatus.FAIL for item in positives)
    true_negative = sum(item.predicted_status == InspectionStatus.PASS for item in negatives)
    recall = true_positive / len(positives) if positives else 0
    specificity = true_negative / len(negatives) if negatives else 0
    localized = sum(
        item.predicted_status == InspectionStatus.FAIL and item.localized for item in positives
    )
    return CalibrationMetrics(
        count=len(examples),
        recall=recall,
        specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2,
        localization_recall=localized / len(positives) if positives else 0,
        defer_rate=sum(item.predicted_status == InspectionStatus.DEFER for item in examples) / len(examples),
        error_rate=sum(item.predicted_status == InspectionStatus.ERROR for item in examples) / len(examples),
    )


def _clean_clean_bias(examples: Sequence[CalibrationExample]) -> float:
    controls = [item for item in examples if item.order == "clean_clean"]
    return (
        sum(item.predicted_status == InspectionStatus.FAIL for item in controls) / len(controls)
        if controls
        else 0
    )


def _group_disagreement(
    examples: Sequence[CalibrationExample], dimension: Literal["order", "repeat"]
) -> float:
    groups: dict[tuple[Any, ...], list[CalibrationExample]] = {}
    for item in examples:
        if dimension == "order":
            if item.order not in {"AB", "BA"}:
                continue
            key = (item.pair_id, item.defect_class, item.repeat)
        else:
            key = (item.pair_id, item.defect_class, item.order)
        groups.setdefault(key, []).append(item)
    eligible = [
        items
        for items in groups.values()
        if len(items) > 1
        and (
            len({item.order for item in items}) > 1
            if dimension == "order"
            else len({item.repeat for item in items}) > 1
        )
    ]
    return (
        sum(len({item.predicted_status for item in items}) > 1 for items in eligible) / len(eligible)
        if eligible
        else 0
    )
