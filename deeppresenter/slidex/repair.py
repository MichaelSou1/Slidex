"""Explicit, auditable source repairs and anti-reward-hacking gates."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from deeppresenter.slidex.models import (
    InspectionReport,
    InspectionResult,
    DefectClass,
    DefectTransition,
    InspectionStatus,
    PolicyViolation,
    PolicyViolationCode,
    RepairAction,
    RepairExecutionStatus,
    RepairHint,
    RepairOperation,
    SlideArtifact,
    SlideElement,
)

_HINT_OPERATIONS = {
    "move_element": RepairOperation.MOVE_ELEMENT,
    "clamp_to_safe_margin": RepairOperation.MOVE_ELEMENT,
    "move_inside_safe_area": RepairOperation.MOVE_ELEMENT,
    "resize_container": RepairOperation.RESIZE_CONTAINER,
    "resize_declared_container": RepairOperation.RESIZE_CONTAINER,
    "reduce_text": RepairOperation.REDUCE_TEXT,
    "change_font_size": RepairOperation.CHANGE_FONT_SIZE,
    "apply_typography_scale": RepairOperation.CHANGE_FONT_SIZE,
    "replace_color": RepairOperation.REPLACE_COLOR,
    "replace_with_palette_color": RepairOperation.REPLACE_COLOR,
    "rename_term": RepairOperation.RENAME_TERM,
    "normalize_terminology": RepairOperation.RENAME_TERM,
}
_SAFE_AUTOMATIC = {
    RepairOperation.MOVE_ELEMENT,
    RepairOperation.RESIZE_CONTAINER,
    RepairOperation.REPLACE_COLOR,
    RepairOperation.RENAME_TERM,
}


def inspection_result_id(result: InspectionResult) -> str:
    """Return a stable ID for a persisted inspection result."""
    payload = result.model_dump_json(exclude={"latency_ms", "cost"})
    return "inspection-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def actions_from_report(report: InspectionReport) -> list[RepairAction]:
    """Translate failed findings into proposals without mutating source."""
    actions: list[RepairAction] = []
    for finding in report.results:
        if finding.status != InspectionStatus.FAIL:
            continue
        hint = finding.repair_hint
        operation = RepairOperation.POLICY_EDIT
        targets = list(finding.element_ids)
        constraints: dict[str, Any] = {}
        reason = "Inspector supplied free-text or unsupported advice; policy must edit source."
        if isinstance(hint, RepairHint):
            operation = _HINT_OPERATIONS.get(hint.action, RepairOperation.POLICY_EDIT)
            targets = hint.targets or targets
            constraints = _normalize_constraints(operation, hint.parameters)
            reason = hint.explanation or reason
        elif isinstance(hint, str):
            constraints = {"suggestion": hint}
        if not targets:
            targets = ["slide-root"]
        actions.append(
            RepairAction(
                action_id=f"repair-{uuid.uuid4().hex[:12]}",
                operation=operation,
                target_ids=targets,
                constraints=constraints,
                source_inspection_ids=[inspection_result_id(finding)],
                before_artifact_id=report.artifact_id,
                policy_reason=reason
                if operation == RepairOperation.POLICY_EDIT
                else None,
            )
        )
    return actions


def _normalize_constraints(
    operation: RepairOperation, parameters: dict[str, Any]
) -> dict[str, Any]:
    constraints = dict(parameters)
    if operation == RepairOperation.REPLACE_COLOR:
        replacement = constraints.get("nearest_palette_color")
        if replacement is not None:
            constraints["replacement"] = replacement
    if operation == RepairOperation.MOVE_ELEMENT and "violated_edges" in constraints:
        margin = constraints.get("allowed_boundary_px")
        if margin is not None:
            for edge in constraints["violated_edges"]:
                constraints[edge] = margin
    return constraints


class DeterministicRepairer:
    """Apply only low-risk source edits; all other edits stay policy suggestions."""

    def apply(self, html_path: Path, action: RepairAction) -> RepairAction:
        if action.operation not in _SAFE_AUTOMATIC:
            return action.model_copy(
                update={
                    "status": RepairExecutionStatus.SUGGESTED,
                    "policy_reason": action.policy_reason
                    or "This operation may alter design intent and requires a policy edit.",
                }
            )
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        targets = [
            soup.select_one(f'[data-slidex-id="{_css_escape(target)}"]')
            for target in action.target_ids
        ]
        elements = [target for target in targets if isinstance(target, Tag)]
        if not elements:
            return action.model_copy(
                update={
                    "status": RepairExecutionStatus.REJECTED,
                    "policy_reason": "No target ID exists in the current source.",
                }
            )
        changed = self._apply_operation(elements, action)
        if not changed:
            return action.model_copy(
                update={
                    "status": RepairExecutionStatus.REJECTED,
                    "policy_reason": "Repair constraints are incomplete or unsafe.",
                }
            )
        html_path.write_text(str(soup), encoding="utf-8")
        return action.model_copy(update={"status": RepairExecutionStatus.APPLIED})

    def _apply_operation(self, elements: list[Tag], action: RepairAction) -> bool:
        if action.operation == RepairOperation.RENAME_TERM:
            canonical = action.constraints.get("canonical")
            variants = action.constraints.get("variants", [])
            if not isinstance(canonical, str) or not canonical or not variants:
                return False
            pattern = re.compile("|".join(re.escape(str(x)) for x in variants), re.I)
            for element in elements:
                for node in element.find_all(string=True):
                    node.replace_with(pattern.sub(canonical, str(node)))
            return True
        if action.operation == RepairOperation.REPLACE_COLOR:
            replacement = action.constraints.get(
                "replacement"
            ) or action.constraints.get("palette_color")
            original = action.constraints.get("color") or action.constraints.get(
                "original"
            )
            if not isinstance(replacement, str):
                return False
            for element in elements:
                style = str(element.get("style", ""))
                if original:
                    style = re.sub(
                        re.escape(str(original)), replacement, style, flags=re.I
                    )
                else:
                    style = _set_style(style, "color", replacement)
                element["style"] = style
            return True
        if action.operation == RepairOperation.MOVE_ELEMENT:
            updates = {
                key: action.constraints.get(key)
                for key in ("left", "top", "right", "bottom")
                if action.constraints.get(key) is not None
            }
            if not updates:
                return False
            for element in elements:
                style = str(element.get("style", ""))
                for key, value in updates.items():
                    style = _set_style(style, key, _css_length(value))
                element["style"] = style
            return True
        if action.operation == RepairOperation.RESIZE_CONTAINER:
            updates = {
                key: action.constraints.get(key)
                for key in ("width", "height", "max-width", "max-height")
                if action.constraints.get(key) is not None
            }
            if not updates:
                return False
            for element in elements:
                style = str(element.get("style", ""))
                for key, value in updates.items():
                    style = _set_style(style, key, _css_length(value))
                element["style"] = style
            return True
        return False


def bind_after_artifact(
    action: RepairAction,
    artifact_id: str,
    *,
    before_report: InspectionReport | None = None,
    after_report: InspectionReport | None = None,
) -> RepairAction:
    """Close lineage and attach the mandatory post-inspection defect delta."""
    if action.status != RepairExecutionStatus.APPLIED:
        raise ValueError("only an applied action can be bound to an after artifact")
    delta = (
        compare_reports(before_report, after_report)
        if before_report is not None and after_report is not None
        else []
    )
    return action.model_copy(
        update={"after_artifact_id": artifact_id, "defect_delta": delta}
    )


def compare_reports(
    before: InspectionReport, after: InspectionReport
) -> list[DefectTransition]:
    """Compute replayable per-class repair transitions."""
    before_status = _report_status(before)
    after_status = _report_status(after)
    before_severity = _report_severity(before)
    after_severity = _report_severity(after)
    transitions: list[DefectTransition] = []
    for defect_class in sorted(before_status.keys() | after_status.keys()):
        old = before_status.get(defect_class, InspectionStatus.NOT_APPLICABLE)
        new = after_status.get(defect_class, InspectionStatus.NOT_APPLICABLE)
        old_rank, new_rank = _status_rank(old), _status_rank(new)
        old_severity = before_severity.get(defect_class, 0)
        new_severity = after_severity.get(defect_class, 0)
        transition = (
            "improved"
            if new_rank < old_rank or (new_rank == old_rank and new_severity < old_severity)
            else "worsened"
            if new_rank > old_rank or (new_rank == old_rank and new_severity > old_severity)
            else "unchanged"
        )
        transitions.append(
            DefectTransition(
                defect_class=defect_class,
                before=old,
                after=new,
                before_severity=old_severity,
                after_severity=new_severity,
                transition=transition,
            )
        )
    return transitions


def append_repair_trajectory(path: Path, action: RepairAction) -> None:
    """Append an explicit action to a replayable JSONL trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(action.model_dump_json() + "\n")


def detect_policy_violations(
    artifact: SlideArtifact,
    *,
    required_text: Iterable[str] = (),
    minimum_font_px: float = 12,
) -> list[PolicyViolation]:
    """Detect hard evasions that can make a checker pass without fixing content."""
    violations: list[PolicyViolation] = []
    computed = (
        list(_flatten(artifact.computed_ir.elements)) if artifact.computed_ir else []
    )
    content = [item for item in computed if item.text.strip()]
    exempt = 0
    for item in content:
        style = item.computed_style
        display = style.get("display", "")
        visibility = style.get("visibility", "")
        opacity = _number(style.get("opacity"), 1)
        if (
            display == "none"
            or visibility == "hidden"
            or opacity <= 0.01
            or not item.visible
        ):
            violations.append(
                _violation(
                    artifact,
                    PolicyViolationCode.HIDDEN_CONTENT,
                    item.element_id,
                    detail="Content is hidden by display, visibility, or opacity.",
                )
            )
        if item.partially_outside_page and item.visible_bbox is None:
            violations.append(
                _violation(
                    artifact,
                    PolicyViolationCode.OFFSCREEN_CONTENT,
                    item.element_id,
                    detail="Content is fully positioned outside the slide viewport.",
                )
            )
        if item.client_width <= 0 or item.client_height <= 0:
            violations.append(
                _violation(
                    artifact,
                    PolicyViolationCode.ZERO_SIZE_CONTENT,
                    item.element_id,
                    detail="Content has a zero-size layout box.",
                )
            )
        if _number(style.get("fontSize"), 100) < minimum_font_px:
            violations.append(
                _violation(
                    artifact,
                    PolicyViolationCode.TINY_TEXT,
                    item.element_id,
                    detail=f"Text font size is below {minimum_font_px:g}px.",
                )
            )
        if item.semantic_role in {"decoration", "background"} or item.style.get(
            "allow_overlap"
        ) in {True, "true"}:
            exempt += 1
    if content and exempt / len(content) > 0.5:
        violations.append(
            _violation(
                artifact,
                PolicyViolationCode.EXEMPTION_ABUSE,
                *[item.element_id for item in content],
                detail="More than half of textual elements claim decorative/overlap exemptions.",
            )
        )
    declared_text = " ".join(
        item.text for item in _flatten(artifact.declared_ir.elements)
    ).casefold()
    for required in required_text:
        if required.strip() and required.casefold() not in declared_text:
            violations.append(
                _violation(
                    artifact,
                    PolicyViolationCode.REQUIRED_CONTENT_REMOVED,
                    detail=f"Required manuscript content is missing: {required!r}.",
                )
            )
    for item in _flatten(artifact.declared_ir.elements):
        if item.tag.lower() == "img" and (item.semantic_role or "").lower() in {
            "text",
            "body",
            "title",
        }:
            violations.append(
                _violation(
                    artifact,
                    PolicyViolationCode.TEXT_AS_IMAGE,
                    item.element_id,
                    detail="Important text is encoded as an image element.",
                )
            )
    return violations


def _violation(
    artifact: SlideArtifact, code: PolicyViolationCode, *ids: str, detail: str
) -> PolicyViolation:
    return PolicyViolation(
        code=code,
        slide_id=artifact.declared_ir.slide_id,
        element_ids=list(ids),
        detail=detail,
    )


def _flatten(elements: list[SlideElement]) -> list[SlideElement]:
    return [item for root in elements for item in [root, *_flatten(root.children)]]


def _set_style(style: str, key: str, value: str) -> str:
    declaration = re.compile(rf"(?i)(?:^|;)\s*{re.escape(key)}\s*:[^;]*")
    cleaned = declaration.sub("", style).strip(" ;")
    return f"{cleaned}; {key}: {value};".lstrip("; ")


def _css_length(value: Any) -> str:
    return f"{value}px" if isinstance(value, (int, float)) else str(value)


def _css_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _number(value: Any, default: float) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else default


def _report_status(report: InspectionReport) -> dict[DefectClass, InspectionStatus]:
    statuses: dict[DefectClass, InspectionStatus] = {}
    for result in report.results:
        current = statuses.get(result.defect_class)
        if current is None or _status_rank(result.status) > _status_rank(current):
            statuses[result.defect_class] = result.status
    return statuses


def _report_severity(report: InspectionReport) -> dict[DefectClass, float]:
    severity: dict[DefectClass, float] = {}
    for result in report.results:
        severity[result.defect_class] = max(
            severity.get(result.defect_class, 0),
            result.severity if result.status == InspectionStatus.FAIL else 0,
        )
    return severity


def _status_rank(status: InspectionStatus) -> int:
    return {
        InspectionStatus.PASS: 0,
        InspectionStatus.NOT_APPLICABLE: 0,
        InspectionStatus.DEFER: 1,
        InspectionStatus.FAIL: 2,
        InspectionStatus.ERROR: 3,
    }[status]
