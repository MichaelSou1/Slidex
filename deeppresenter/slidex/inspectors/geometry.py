"""Deterministic geometry inspectors for G1, G2, G3, G6, and G7."""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations

from deeppresenter.slidex.inspectors.base import result
from deeppresenter.slidex.models import (
    ComputedSlideElement,
    DefectClass,
    Evidence,
    EvidenceSource,
    InspectionResult,
    InspectionStatus,
    ObservedBoundingBox,
    RepairHint,
    SlideArtifact,
    SlideElement,
)


def _flatten(elements: list[SlideElement]) -> list[SlideElement]:
    return [
        element for root in elements for element in [root, *_flatten(root.children)]
    ]


def _box_intersection(
    a: ObservedBoundingBox, b: ObservedBoundingBox
) -> tuple[float, float, float, float] | None:
    left, top = max(a.x, b.x), max(a.y, b.y)
    right, bottom = (
        min(a.x + a.width, b.x + b.width),
        min(a.y + a.height, b.y + b.height),
    )
    return (
        (left, top, right - left, bottom - top)
        if right > left and bottom > top
        else None
    )


def _computed(artifact: SlideArtifact) -> list[ComputedSlideElement] | None:
    if artifact.computed_ir is None or not artifact.computed_ir.render_ready:
        return None
    return [
        item
        for item in _flatten(artifact.computed_ir.elements)
        if isinstance(item, ComputedSlideElement)
    ]


def _skip_decorative(element: ComputedSlideElement) -> bool:
    return element.semantic_role in {"background", "decoration"} or element.style.get(
        "allow_overlap"
    ) in {True, "true"}


@dataclass
class OverlapInspector:
    """G2 pairwise visible-box overlap with structural exemptions."""

    tolerance_px: float = 1
    min_area_px: float = 4
    name: str = "geometry.overlap"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.G2

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        started = time.perf_counter()
        elements = _computed(artifact)
        if elements is None:
            return [
                result(
                    self,
                    artifact,
                    InspectionStatus.DEFER,
                    confidence=0,
                    started_at=started,
                )
            ]
        failures: list[InspectionResult] = []
        for left, right in combinations(
            [
                e
                for e in elements
                if e.visible and e.visible_bbox and not _skip_decorative(e)
            ],
            2,
        ):
            if (
                left.parent_id == right.element_id
                or right.parent_id == left.element_id
                or (
                    left.parent_id
                    and left.parent_id == right.parent_id
                    and left.style.get("overlay_group")
                    == right.style.get("overlay_group")
                    and left.style.get("overlay_group")
                )
            ):
                continue
            intersection = _box_intersection(left.visible_bbox, right.visible_bbox)
            if not intersection:
                continue
            x, y, width, height = intersection
            area = width * height
            if (
                width <= self.tolerance_px
                or height <= self.tolerance_px
                or area < self.min_area_px
            ):
                continue
            smaller = max(
                1,
                min(
                    left.visible_bbox.width * left.visible_bbox.height,
                    right.visible_bbox.width * right.visible_bbox.height,
                ),
            )
            ratio = area / smaller
            ids = [left.element_id, right.element_id]
            failures.append(
                result(
                    self,
                    artifact,
                    InspectionStatus.FAIL,
                    severity=min(1, ratio),
                    evidence=[
                        Evidence(
                            source=EvidenceSource.COMPUTED_IR,
                            detail=f"intersection=({x:.2f},{y:.2f},{width:.2f},{height:.2f}), area={area:.2f}px², smaller_ratio={ratio:.4f}",
                            element_ids=ids,
                        )
                    ],
                    element_ids=ids,
                    repair_hint=RepairHint(
                        action="separate_elements",
                        targets=ids,
                        parameters={
                            "intersection_bbox": [x, y, width, height],
                            "area_px": area,
                        },
                    ),
                    started_at=started,
                )
            )
        return failures or [
            result(self, artifact, InspectionStatus.PASS, started_at=started)
        ]


@dataclass
class AlignmentInspector:
    """G3 sibling alignment outlier detection with minimum evidence."""

    tolerance_px: float = 2
    minimum_peers: int = 3
    name: str = "geometry.alignment"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.G3

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        started = time.perf_counter()
        elements = _computed(artifact)
        if elements is None:
            return [
                result(
                    self,
                    artifact,
                    InspectionStatus.DEFER,
                    confidence=0,
                    started_at=started,
                )
            ]
        failures: list[InspectionResult] = []
        groups: dict[tuple[str | None, str | None], list[ComputedSlideElement]] = {}
        for item in elements:
            if item.visible and item.bbox and not _skip_decorative(item):
                groups.setdefault((item.parent_id, item.semantic_role), []).append(item)
        applicable = False
        for peers in groups.values():
            if len(peers) < self.minimum_peers:
                continue
            applicable = True
            for edge, getter in {
                "left": lambda b: b.x,
                "right": lambda b: b.x + b.width,
                "center": lambda b: b.x + b.width / 2,
                "top": lambda b: b.y,
                "bottom": lambda b: b.y + b.height,
            }.items():
                values = sorted(getter(item.bbox) for item in peers)
                median = values[len(values) // 2]
                inliers = [
                    item
                    for item in peers
                    if abs(getter(item.bbox) - median) <= self.tolerance_px
                ]
                if len(inliers) < self.minimum_peers - 1:
                    continue
                for item in peers:
                    offset = getter(item.bbox) - median
                    if abs(offset) <= self.tolerance_px:
                        continue
                    refs = [peer.element_id for peer in inliers]
                    failures.append(
                        result(
                            self,
                            artifact,
                            InspectionStatus.FAIL,
                            severity=min(
                                1, abs(offset) / max(1, self.tolerance_px * 5)
                            ),
                            evidence=[
                                Evidence(
                                    source=EvidenceSource.COMPUTED_IR,
                                    detail=f"{edge} offset={offset:.2f}px from sibling median={median:.2f}px; references={refs}",
                                    element_ids=[item.element_id, *refs],
                                )
                            ],
                            element_ids=[item.element_id],
                            repair_hint=RepairHint(
                                action="align_edge",
                                targets=[item.element_id],
                                parameters={
                                    "edge": edge,
                                    "offset_px": offset,
                                    "reference_ids": refs,
                                },
                            ),
                            started_at=started,
                        )
                    )
                if failures:
                    break
        if failures:
            unique: dict[str, InspectionResult] = {
                item.element_ids[0]: item for item in failures
            }
            return list(unique.values())
        status = (
            InspectionStatus.PASS if applicable else InspectionStatus.NOT_APPLICABLE
        )
        return [result(self, artifact, status, started_at=started)]


@dataclass
class MarginInspector:
    """G6 safe-area violation on visible boxes."""

    safety_margin_px: float = 24
    name: str = "geometry.margin"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.G6

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        started = time.perf_counter()
        elements = _computed(artifact)
        if elements is None:
            return [
                result(
                    self,
                    artifact,
                    InspectionStatus.DEFER,
                    confidence=0,
                    started_at=started,
                )
            ]
        failures: list[InspectionResult] = []
        for item in elements:
            if (
                not item.visible
                or not item.visible_bbox
                or _skip_decorative(item)
                or item.style.get("full_bleed") in {True, "true"}
            ):
                continue
            box = item.visible_bbox
            boundaries = {
                "left": box.x,
                "top": box.y,
                "right": box.page_width - (box.x + box.width),
                "bottom": box.page_height - (box.y + box.height),
            }
            violated = {
                edge: distance
                for edge, distance in boundaries.items()
                if distance < self.safety_margin_px
            }
            if not violated:
                continue
            ids = [item.element_id]
            failures.append(
                result(
                    self,
                    artifact,
                    InspectionStatus.FAIL,
                    severity=min(
                        1,
                        max(
                            self.safety_margin_px - value for value in violated.values()
                        )
                        / max(1, self.safety_margin_px),
                    ),
                    evidence=[
                        Evidence(
                            source=EvidenceSource.COMPUTED_IR,
                            detail=f"safe margin={self.safety_margin_px}px, violations={violated}",
                            element_ids=ids,
                        )
                    ],
                    element_ids=ids,
                    repair_hint=RepairHint(
                        action="move_inside_safe_area",
                        targets=ids,
                        parameters={
                            "violated_edges": violated,
                            "allowed_boundary_px": self.safety_margin_px,
                        },
                    ),
                    started_at=started,
                )
            )
        return failures or [
            result(self, artifact, InspectionStatus.PASS, started_at=started)
        ]


@dataclass
class DeclaredOverflowInspector:
    """G1 source-declared text/container constraint violations."""

    name: str = "geometry.declared_overflow"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.G1

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        started = time.perf_counter()
        failures = []
        for item in _flatten(artifact.declared_ir.elements):
            declared = item.style.get("declared_overflow") or item.style.get(
                "constraint_violation"
            )
            if declared not in {True, "true", "text"}:
                continue
            failures.append(
                result(
                    self,
                    artifact,
                    InspectionStatus.FAIL,
                    severity=1,
                    evidence=[
                        Evidence(
                            source=EvidenceSource.DECLARED_IR,
                            detail="source declares a text/container constraint violation",
                            element_ids=[item.element_id],
                        )
                    ],
                    element_ids=[item.element_id],
                    repair_hint=RepairHint(
                        action="resize_declared_container", targets=[item.element_id]
                    ),
                    started_at=started,
                )
            )
        if failures:
            return failures
        has_constraint_evidence = any(
            item.style.get("declared_overflow") is not None
            or item.style.get("constraint_violation") is not None
            for item in _flatten(artifact.declared_ir.elements)
        )
        return [
            result(
                self,
                artifact,
                InspectionStatus.PASS
                if has_constraint_evidence
                else InspectionStatus.DEFER,
                confidence=1 if has_constraint_evidence else 0,
                evidence=(
                    [
                        Evidence(
                            source=EvidenceSource.DECLARED_IR,
                            detail="source explicitly declares no text/container constraint violation",
                        )
                    ]
                    if has_constraint_evidence
                    else [
                        Evidence(
                            source=EvidenceSource.DECLARED_IR,
                            detail="source has no explicit container-fit bookkeeping",
                        )
                    ]
                ),
                started_at=started,
            )
        ]


@dataclass
class RenderOverflowInspector:
    """G7 computed overflow; unexplained pixel anomalies remain neural work."""

    tolerance_px: float = 1
    name: str = "geometry.render_overflow"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.G7

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        started = time.perf_counter()
        elements = _computed(artifact)
        if elements is None:
            return [
                result(
                    self,
                    artifact,
                    InspectionStatus.DEFER,
                    confidence=0,
                    started_at=started,
                )
            ]
        failures = []
        for item in elements:
            overflow_x = item.scroll_width - item.client_width
            overflow_y = item.scroll_height - item.client_height
            child_boxes = [child.bbox for child in item.children if child.bbox]
            child_outside = (
                any(
                    box.x < item.bbox.x - self.tolerance_px
                    or box.y < item.bbox.y - self.tolerance_px
                    or box.x + box.width
                    > item.bbox.x + item.bbox.width + self.tolerance_px
                    or box.y + box.height
                    > item.bbox.y + item.bbox.height + self.tolerance_px
                    for box in child_boxes
                )
                if item.bbox
                else False
            )
            text_outside = (
                any(
                    box.x < item.bbox.x - self.tolerance_px
                    or box.y < item.bbox.y - self.tolerance_px
                    or box.x + box.width
                    > item.bbox.x + item.bbox.width + self.tolerance_px
                    or box.y + box.height
                    > item.bbox.y + item.bbox.height + self.tolerance_px
                    for box in item.text_bboxes
                )
                if item.bbox
                else False
            )
            if (
                overflow_x <= self.tolerance_px
                and overflow_y <= self.tolerance_px
                and not child_outside
                and not text_outside
                and not item.partially_outside_page
            ):
                continue
            ids = [item.element_id]
            detail = f"scroll overflow=({overflow_x:.2f},{overflow_y:.2f})px, child_outside={child_outside}, text_outside={text_outside}, page_outside={item.partially_outside_page}"
            failures.append(
                result(
                    self,
                    artifact,
                    InspectionStatus.FAIL,
                    severity=min(
                        1,
                        max(overflow_x, overflow_y, self.tolerance_px)
                        / max(item.client_width, item.client_height, 1),
                    ),
                    evidence=[
                        Evidence(
                            source=EvidenceSource.COMPUTED_IR,
                            detail=detail,
                            element_ids=ids,
                        )
                    ],
                    element_ids=ids,
                    repair_hint=RepairHint(
                        action="fit_rendered_content",
                        targets=ids,
                        parameters={
                            "overflow_x_px": overflow_x,
                            "overflow_y_px": overflow_y,
                            "clipped": item.clipped,
                        },
                    ),
                    started_at=started,
                )
            )
        if failures:
            return failures
        return [
            result(
                self,
                artifact,
                InspectionStatus.PASS,
                evidence=[
                    Evidence(
                        source=EvidenceSource.COMPUTED_IR,
                        detail="DOM geometry has no overflow; pixel-only anomalies are outside this inspector",
                    )
                ],
                started_at=started,
            )
        ]
