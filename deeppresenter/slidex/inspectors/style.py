"""Deterministic typography and brand-color inspectors."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from deeppresenter.slidex.inspectors.base import result
from deeppresenter.slidex.models import (
    ComputedSlideElement,
    DefectClass,
    Evidence,
    EvidenceSource,
    InspectionResult,
    InspectionStatus,
    RepairHint,
    SlideArtifact,
    SlideElement,
)


def _flatten(elements: list[SlideElement]) -> list[SlideElement]:
    return [item for root in elements for item in [root, *_flatten(root.children)]]


def _font_size(value: str | None) -> float | None:
    match = re.fullmatch(r"\s*([\d.]+)px\s*", value or "")
    return float(match.group(1)) if match else None


@dataclass
class TypographyInspector:
    """G4 same-role typography scale consistency across one or more slides."""

    size_tolerance_px: float = 1
    name: str = "style.typography"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.G4

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        return self.inspect_deck([artifact])

    def inspect_deck(self, artifacts: list[SlideArtifact]) -> list[InspectionResult]:
        started = time.perf_counter()
        if not artifacts or any(
            item.computed_ir is None or not item.computed_ir.render_ready
            for item in artifacts
        ):
            return (
                [
                    result(
                        self,
                        artifacts[0],
                        InspectionStatus.DEFER,
                        confidence=0,
                        started_at=started,
                    )
                ]
                if artifacts
                else []
            )
        occurrences: dict[
            str,
            list[tuple[SlideArtifact, ComputedSlideElement, tuple[float, str, str]]],
        ] = defaultdict(list)
        for artifact in artifacts:
            for element in _flatten(artifact.computed_ir.elements):
                if (
                    not isinstance(element, ComputedSlideElement)
                    or not element.visible
                    or element.semantic_role
                    in {None, "unknown", "background", "decoration"}
                    or element.style.get("intentional_emphasis") in {True, "true"}
                ):
                    continue
                size = _font_size(element.computed_style.get("fontSize"))
                if size is not None:
                    occurrences[element.semantic_role].append(
                        (
                            artifact,
                            element,
                            (
                                size,
                                element.computed_style.get("fontFamily", ""),
                                element.computed_style.get("fontWeight", ""),
                            ),
                        )
                    )
        failures: list[InspectionResult] = []
        applicable = False
        for role, values in occurrences.items():
            if len(values) < 2:
                continue
            applicable = True
            signatures = [signature for _, _, signature in values]
            expected = Counter(signatures).most_common(1)[0][0]
            for artifact, element, actual in values:
                if (
                    abs(actual[0] - expected[0]) <= self.size_tolerance_px
                    and actual[1:] == expected[1:]
                ):
                    continue
                failures.append(
                    result(
                        self,
                        artifact,
                        InspectionStatus.FAIL,
                        severity=min(
                            1,
                            abs(actual[0] - expected[0]) / max(expected[0], 1)
                            + (actual[1:] != expected[1:]) * 0.5,
                        ),
                        evidence=[
                            Evidence(
                                source=EvidenceSource.COMPUTED_IR,
                                detail=f"role={role}, expected(size,family,weight)={expected}, actual={actual}",
                                element_ids=[element.element_id],
                            )
                        ],
                        element_ids=[element.element_id],
                        repair_hint=RepairHint(
                            action="apply_typography_scale",
                            targets=[element.element_id],
                            parameters={
                                "role": role,
                                "font_size_px": expected[0],
                                "font_family": expected[1],
                                "font_weight": expected[2],
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
                artifacts[0],
                InspectionStatus.PASS
                if applicable
                else InspectionStatus.NOT_APPLICABLE,
                started_at=started,
            )
        ]


@dataclass
class BrandColorInspector:
    """G5 visible CSS colors compared in CIEDE2000 Lab space."""

    delta_e_threshold: float = 5
    name: str = "style.brand_color"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.G5

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        started = time.perf_counter()
        if artifact.computed_ir is None or not artifact.computed_ir.render_ready:
            return [
                result(
                    self,
                    artifact,
                    InspectionStatus.DEFER,
                    confidence=0,
                    started_at=started,
                )
            ]
        palette = [
            _parse_color(value)
            for value in artifact.declared_ir.theme_tokens.get("palette", [])
        ]
        palette = [color for color in palette if color is not None and color[3] > 0]
        if not palette:
            return [
                result(
                    self,
                    artifact,
                    InspectionStatus.DEFER,
                    confidence=0,
                    evidence=[
                        Evidence(
                            source=EvidenceSource.DECLARED_IR,
                            detail="trusted palette is absent",
                        )
                    ],
                    started_at=started,
                )
            ]
        failures = []
        for element in _flatten(artifact.computed_ir.elements):
            if (
                not isinstance(element, ComputedSlideElement)
                or not element.visible
                or element.semantic_role == "background"
            ):
                continue
            background = _parse_color(
                element.computed_style.get("backgroundColor")
            ) or (255, 255, 255, 1)
            for channel, key in {
                "text": "color",
                "background": "backgroundColor",
                "border": "borderColor",
            }.items():
                color = _parse_color(element.computed_style.get(key))
                if color is None or color[3] == 0:
                    continue
                visible = _composite(color, background) if color[3] < 1 else color
                distances = [
                    (
                        _ciede2000(_rgb_to_lab(visible), _rgb_to_lab(candidate)),
                        candidate,
                    )
                    for candidate in palette
                ]
                delta_e, nearest = min(distances, key=lambda item: item[0])
                if delta_e <= self.delta_e_threshold:
                    continue
                failures.append(
                    result(
                        self,
                        artifact,
                        InspectionStatus.FAIL,
                        severity=min(1, (delta_e - self.delta_e_threshold) / 30),
                        evidence=[
                            Evidence(
                                source=EvidenceSource.COMPUTED_IR,
                                detail=f"channel={channel}, visible={_hex(visible)}, nearest={_hex(nearest)}, delta_e_2000={delta_e:.3f}",
                                element_ids=[element.element_id],
                            )
                        ],
                        element_ids=[element.element_id],
                        repair_hint=RepairHint(
                            action="replace_color",
                            targets=[element.element_id],
                            parameters={
                                "channel": channel,
                                "nearest_palette_color": _hex(nearest),
                                "delta_e_2000": delta_e,
                            },
                        ),
                        started_at=started,
                    )
                )
        return failures or [
            result(self, artifact, InspectionStatus.PASS, started_at=started)
        ]


def _parse_color(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    value = value.strip().lower()
    if value == "transparent":
        return 0, 0, 0, 0
    if value.startswith("#"):
        raw = value[1:]
        if len(raw) in {3, 4}:
            raw = "".join(char * 2 for char in raw)
        if len(raw) in {6, 8}:
            return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4)) + (
                (int(raw[6:8], 16) / 255) if len(raw) == 8 else 1,
            )
    match = re.fullmatch(r"rgba?\(([^)]*)\)", value)
    if match:
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) in {3, 4}:
            rgb = tuple(
                float(part.rstrip("%")) * 2.55 if part.endswith("%") else float(part)
                for part in parts[:3]
            )
            return rgb + (float(parts[3]) if len(parts) == 4 else 1,)
    return None


def _composite(
    front: tuple[float, float, float, float], back: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    alpha = front[3]
    return tuple(front[i] * alpha + back[i] * (1 - alpha) for i in range(3)) + (1,)


def _hex(color: tuple[float, float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(value))):02x}" for value in color[:3])


def _rgb_to_lab(color: tuple[float, float, float, float]) -> tuple[float, float, float]:
    rgb = [value / 255 for value in color[:3]]
    linear = [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in rgb]
    x = (linear[0] * 0.4124 + linear[1] * 0.3576 + linear[2] * 0.1805) / 0.95047
    y = linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722
    z = (linear[0] * 0.0193 + linear[1] * 0.1192 + linear[2] * 0.9505) / 1.08883

    def transform(value: float) -> float:
        return value ** (1 / 3) if value > 0.008856 else 7.787 * value + 16 / 116

    return (
        116 * transform(y) - 16,
        500 * (transform(x) - transform(y)),
        200 * (transform(y) - transform(z)),
    )


def _ciede2000(
    lab1: tuple[float, float, float], lab2: tuple[float, float, float]
) -> float:
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cbar**7 / (cbar**7 + 25**7)))
    ap1 = (1 + g) * a1
    ap2 = (1 + g) * a2
    cp1 = math.hypot(ap1, b1)
    cp2 = math.hypot(ap2, b2)

    def hue(a: float, b: float) -> float:
        return (math.degrees(math.atan2(b, a)) + 360) % 360 if a or b else 0

    h1 = hue(ap1, b1)
    h2 = hue(ap2, b2)
    dl = l2 - l1
    dc = cp2 - cp1
    dh = h2 - h1
    if cp1 * cp2 == 0:
        dh = 0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dH = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dh / 2))
    lbar = (l1 + l2) / 2
    cpbar = (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        hbar = h1 + h2
    elif abs(h1 - h2) <= 180:
        hbar = (h1 + h2) / 2
    elif h1 + h2 < 360:
        hbar = (h1 + h2 + 360) / 2
    else:
        hbar = (h1 + h2 - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(hbar - 30))
        + 0.24 * math.cos(math.radians(2 * hbar))
        + 0.32 * math.cos(math.radians(3 * hbar + 6))
        - 0.20 * math.cos(math.radians(4 * hbar - 63))
    )
    sl = 1 + 0.015 * (lbar - 50) ** 2 / math.sqrt(20 + (lbar - 50) ** 2)
    sc = 1 + 0.045 * cpbar
    sh = 1 + 0.015 * cpbar * t
    rt = (
        -2
        * math.sqrt(cpbar**7 / (cpbar**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((hbar - 275) / 25) ** 2))))
    )
    return math.sqrt(
        (dl / sl) ** 2 + (dc / sc) ** 2 + (dH / sh) ** 2 + rt * (dc / sc) * (dH / sh)
    )
