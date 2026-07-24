import hashlib
from pathlib import Path

import pytest

from deeppresenter.slidex.deck import DeckInspector, enforce_export_gate
from deeppresenter.slidex.models import (
    ComputedSlideElement,
    ComputedSlideIR,
    DeclaredSlideIR,
    DefectClass,
    InspectionReport,
    InspectionResult,
    InspectionStatus,
    PolicyViolationCode,
    Provenance,
    RepairExecutionStatus,
    RepairHint,
    RepairOperation,
    SlideArtifact,
    SlideElement,
)
from deeppresenter.slidex.repair import (
    DeterministicRepairer,
    actions_from_report,
    bind_after_artifact,
    detect_policy_violations,
)
from deeppresenter.utils.config import SlidexConfig


def _artifact(slide_id: str = "slide_01", *, hidden: bool = False) -> SlideArtifact:
    declared = SlideElement(
        element_id="body",
        tag="p",
        semantic_role="body",
        text="Acme platform",
        style={"allow_overlap": "true"} if hidden else {},
    )
    computed = ComputedSlideElement(
        element_id="body",
        tag="p",
        semantic_role="decoration" if hidden else "body",
        text="Acme platform",
        bbox={
            "x": 20,
            "y": 20,
            "width": 200,
            "height": 40,
            "page_width": 1280,
            "page_height": 720,
        },
        visible_bbox=None
        if hidden
        else {
            "x": 20,
            "y": 20,
            "width": 200,
            "height": 40,
            "page_width": 1280,
            "page_height": 720,
        },
        client_width=200,
        client_height=40,
        scroll_width=200,
        scroll_height=40,
        visible=not hidden,
        partially_outside_page=hidden,
        computed_style={
            "display": "none" if hidden else "block",
            "visibility": "visible",
            "opacity": "0" if hidden else "1",
            "fontSize": "8px" if hidden else "24px",
        },
        style={"allow_overlap": "true"} if hidden else {},
    )
    return SlideArtifact(
        artifact_id=f"artifact-{slide_id}",
        source_uri=f"{slide_id}.html",
        source_sha256=hashlib.sha256(slide_id.encode()).hexdigest(),
        declared_ir=DeclaredSlideIR(
            slide_id=slide_id,
            page_width=1280,
            page_height=720,
            elements=[declared],
        ),
        computed_ir=ComputedSlideIR(
            slide_id=slide_id,
            page_width=1280,
            page_height=720,
            elements=[computed],
            browser="Chromium",
            browser_version="1",
        ),
        provenance=Provenance(creation_action="test"),
    )


def _report(artifact_id: str = "before") -> InspectionReport:
    finding = InspectionResult(
        defect_class=DefectClass.S3,
        status=InspectionStatus.FAIL,
        severity=0.5,
        confidence=1,
        element_ids=["body"],
        repair_hint=RepairHint(
            action="normalize_terminology",
            targets=["body"],
            parameters={"canonical": "Slidex", "variants": ["Slide-X"]},
        ),
        inspector_version="1",
    )
    return InspectionReport(
        artifact_id=artifact_id,
        slide_id="slide_01",
        results=[finding],
        router_version="1",
        taxonomy_version="1",
    )


@pytest.mark.unit
def test_machine_readable_action_preserves_lineage_and_requires_reinspection(
    tmp_path: Path,
) -> None:
    html = tmp_path / "slide.html"
    html.write_text('<p data-slidex-id="body">Slide-X works</p>')
    action = actions_from_report(_report())[0]
    assert action.operation == RepairOperation.RENAME_TERM
    assert action.source_inspection_ids[0].startswith("inspection-")
    executed = DeterministicRepairer().apply(html, action)
    assert executed.status == RepairExecutionStatus.APPLIED
    assert "Slidex works" in html.read_text()
    assert executed.after_artifact_id is None
    after_report = _report("after").model_copy(
        update={
            "results": [
                _report("after")
                .results[0]
                .model_copy(update={"status": InspectionStatus.PASS, "severity": 0})
            ]
        }
    )
    closed = bind_after_artifact(
        executed, "after", before_report=_report(), after_report=after_report
    )
    assert (closed.before_artifact_id, closed.after_artifact_id) == ("before", "after")
    assert closed.defect_delta[0].transition == "improved"


@pytest.mark.unit
def test_risky_or_free_text_repair_remains_policy_edit() -> None:
    report = _report()
    finding = report.results[0].model_copy(
        update={"repair_hint": "shrink the text until it fits"}
    )
    action = actions_from_report(report.model_copy(update={"results": [finding]}))[0]
    assert action.operation == RepairOperation.POLICY_EDIT
    assert action.status == RepairExecutionStatus.SUGGESTED


@pytest.mark.unit
def test_anti_reward_hacking_findings_are_hard_and_localized() -> None:
    violations = detect_policy_violations(
        _artifact(hidden=True), required_text=["required manuscript sentence"]
    )
    codes = {item.code for item in violations}
    assert {
        PolicyViolationCode.HIDDEN_CONTENT,
        PolicyViolationCode.OFFSCREEN_CONTENT,
        PolicyViolationCode.TINY_TEXT,
        PolicyViolationCode.EXEMPTION_ABUSE,
        PolicyViolationCode.REQUIRED_CONTENT_REMOVED,
    } <= codes
    assert all(item.severity == 1 for item in violations)


class FakeCritic:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def inspect(self, context):
        slide_id = context.artifact.declared_ir.slide_id
        self.calls.append(slide_id)
        return InspectionReport(
            artifact_id=context.artifact.artifact_id,
            slide_id=slide_id,
            results=[],
            router_version="1.0",
            taxonomy_version="1.0",
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deck_reinspection_is_incremental_and_export_gate_blocks_hard_violation() -> (
    None
):
    critic = FakeCritic()
    inspector = DeckInspector(critic, SlidexConfig())
    first = await inspector.inspect([_artifact("slide_01"), _artifact("slide_02")])
    assert critic.calls == ["slide_01", "slide_02"]
    critic.calls.clear()
    second = await inspector.inspect(
        [_artifact("slide_01", hidden=True), _artifact("slide_02")],
        previous=first,
        changed_slide_ids=["slide_01"],
    )
    assert critic.calls == ["slide_01"]
    assert second.affected_slide_ids == ["slide_01"]
    assert not second.export_allowed
    with pytest.raises(RuntimeError, match="export blocked"):
        enforce_export_gate(second)
    overridden = second.model_copy(
        update={"export_allowed": True, "override_reason": "human approval"}
    )
    enforce_export_gate(overridden)
