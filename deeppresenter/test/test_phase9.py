import hashlib
import json

import pytest

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.models import (
    DefectClass,
    ExportManifest,
    FinalArtifactStatus,
    InspectionReport,
    InspectionResult,
    InspectionStatus,
    PolicyViolation,
    PolicyViolationCode,
    RepairAction,
    RepairExecutionStatus,
    RepairOperation,
)
from deeppresenter.slidex.repair import inspection_result_id
from deeppresenter.slidex.reward import (
    CalibrationExample,
    EfficiencyUsage,
    EvaluationSplit,
    RewardCalibrator,
    RewardConfig,
    RewardEngine,
    TaskOutcome,
    explain_reward,
    persist_reward,
)


def _result(
    defect: DefectClass,
    status: InspectionStatus,
    severity: float = 0,
) -> InspectionResult:
    return InspectionResult(
        defect_class=defect,
        status=status,
        severity=severity,
        confidence=1,
        inspector_name="test",
        inspector_version="1",
        input_hash=hashlib.sha256(defect.value.encode()).hexdigest(),
    )


def _report(
    artifact: str,
    results: list[InspectionResult],
    violations: list[PolicyViolation] | None = None,
) -> InspectionReport:
    return InspectionReport(
        artifact_id=artifact,
        slide_id=artifact,
        results=results,
        router_version="1.0",
        taxonomy_version="1.0",
        policy_violations=violations or [],
    )


def _export(status: FinalArtifactStatus, *, hard: bool = False) -> ExportManifest:
    return ExportManifest(export_id="export-1", status=status, hard_penalty=hard)


@pytest.mark.unit
def test_reward_is_replayable_and_preserves_defer_coverage() -> None:
    report = _report(
        "artifact-1",
        [
            _result(DefectClass.G1, InspectionStatus.PASS),
            _result(DefectClass.G2, InspectionStatus.FAIL, 0.5),
            _result(DefectClass.S1, InspectionStatus.DEFER),
            _result(DefectClass.S2, InspectionStatus.ERROR),
        ],
    )
    engine = RewardEngine()
    kwargs = {
        "export": _export(FinalArtifactStatus.PPTX_RENDER_VALIDATED),
        "task": TaskOutcome(
            actual_page_count=1,
            expected_page_count=1,
            required_content={"title": True},
        ),
        "usage": EfficiencyUsage(tokens=1000, model_calls=1),
    }
    first = engine.compute([report], **kwargs)
    second = engine.compute([report], **kwargs)
    assert first == second
    assert first.reward_id == second.reward_id
    assert first.vector.semantic_reward.defer_count == 5
    assert first.vector.semantic_reward.error_count == 1
    assert first.coverage["semantic"] == 0
    result_id = inspection_result_id(report.results[0])
    assert result_id in first.component_inspection_ids["geometry_reward"]
    assert explain_reward(first)["aggregate"] == first.aggregate


@pytest.mark.unit
def test_invalid_export_is_terminal_and_soft_scores_cannot_offset_hard_defect() -> None:
    clean = [_result(item, InspectionStatus.PASS) for item in DefectClass]
    invalid = RewardEngine().compute(
        [_report("artifact", clean)],
        export=_export(FinalArtifactStatus.INVALID_ARTIFACT, hard=True),
        task=TaskOutcome(
            actual_page_count=1,
            expected_page_count=1,
            user_constraints={"all": True},
        ),
    )
    assert invalid.aggregate == -1
    assert "invalid_export" in invalid.gate_reasons

    severe = RewardEngine().compute(
        [_report("artifact", [*clean, _result(DefectClass.G7, InspectionStatus.FAIL, 1)])],
        export=_export(FinalArtifactStatus.PPTX_RENDER_VALIDATED),
        task=TaskOutcome(actual_page_count=1, expected_page_count=1),
    )
    assert "aesthetic_bonus_withheld" in severe.gate_reasons
    assert severe.aggregate <= 0


@pytest.mark.unit
def test_policy_penalty_and_potential_repair_delta() -> None:
    before_g1 = _result(DefectClass.G1, InspectionStatus.FAIL, 1)
    before = _report("before", [before_g1])
    after = _report(
        "after",
        [
            _result(DefectClass.G1, InspectionStatus.PASS),
            _result(DefectClass.S1, InspectionStatus.FAIL, 0.5),
        ],
        [
            PolicyViolation(
                code=PolicyViolationCode.HIDDEN_CONTENT,
                slide_id="after",
                detail="hidden",
            )
        ],
    )
    action = RepairAction(
        action_id="repair-1",
        operation=RepairOperation.MOVE_ELEMENT,
        target_ids=["title"],
        source_inspection_ids=[inspection_result_id(before_g1)],
        before_artifact_id="before",
        after_artifact_id="after",
        status=RepairExecutionStatus.APPLIED,
    )
    reward = RewardEngine().compute(
        [after],
        before_reports=[before],
        repair_actions=[action],
        export=_export(FinalArtifactStatus.PPTX_RENDER_VALIDATED),
    )
    delta = reward.vector.repair_delta_reward
    assert delta is not None
    assert delta.fail_delta == 0
    assert delta.severity_delta == 0.5
    assert delta.targeted_resolved == [DefectClass.G1]
    assert delta.introduced_defects == [DefectClass.S1]
    assert delta.deterministic_steps == 1 and delta.policy_steps == 0
    assert reward.vector.policy_violation_penalty < 0
    assert {item.transition for item in delta.transitions} >= {"improved", "worsened", "unchanged"}


@pytest.mark.unit
def test_reward_persistence_and_config_hash(tmp_path) -> None:
    config = RewardConfig(severe_defect_threshold=0.8)
    assert config.config_hash != RewardConfig().config_hash
    reward = RewardEngine(config).compute(
        [_report("artifact", [_result(DefectClass.G1, InspectionStatus.PASS)])]
    )
    store = ArtifactStore(tmp_path / "store")
    store.create_episode("episode")
    uri = persist_reward(store, "episode", reward)
    _, location = uri.split("://", 1)
    _, artifact_id, name = location.split("/", 2)
    restored = json.loads(store.read_artifact_file("episode", artifact_id, name))
    assert restored["reward_id"] == reward.reward_id
    assert restored["reward_config_hash"] == config.config_hash


@pytest.mark.unit
def test_matched_pair_calibration_reports_bias_and_separates_provider() -> None:
    examples = [
        CalibrationExample(
            pair_id="p1",
            defect_class=DefectClass.G7,
            expected_defective=True,
            predicted_status=InspectionStatus.FAIL,
            localized=True,
            split=EvaluationSplit.HELD_OUT,
            provider="vendor",
            model="critic",
            order="AB",
        ),
        CalibrationExample(
            pair_id="p1",
            defect_class=DefectClass.G7,
            expected_defective=False,
            predicted_status=InspectionStatus.PASS,
            split=EvaluationSplit.HELD_OUT,
            provider="vendor",
            model="critic",
            order="BA",
        ),
        CalibrationExample(
            pair_id="control",
            defect_class=DefectClass.G7,
            expected_defective=False,
            predicted_status=InspectionStatus.FAIL,
            split=EvaluationSplit.HELD_OUT,
            provider="vendor",
            model="critic",
            order="clean_clean",
        ),
    ]
    report = RewardCalibrator(RewardConfig()).evaluate(
        examples, split=EvaluationSplit.HELD_OUT
    )
    assert report.overall.recall == 1
    assert report.overall.specificity == 0.5
    assert report.overall.balanced_accuracy == 0.75
    assert report.overall.localization_recall == 1
    assert report.clean_clean_false_positive_rate == 1
    assert "vendor/critic" in report.by_neural_executor
    with pytest.raises(ValueError):
        RewardCalibrator(RewardConfig()).evaluate(
            examples, split=EvaluationSplit.DEVELOPMENT
        )
