from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deeppresenter.slidex.browser import BrowserObserver, extract_declared_ir
from deeppresenter.slidex.critic import SymbolicCritic
from deeppresenter.slidex.inspectors import (
    AlignmentInspector,
    BrandColorInspector,
    MarginInspector,
    OverlapInspector,
    RenderOverflowInspector,
    TerminologyInspector,
    TypographyInspector,
)
from deeppresenter.slidex.models import (
    ComputedSlideElement,
    ComputedSlideIR,
    DefectClass,
    DeclaredSlideIR,
    InspectionStatus,
    ObservedBoundingBox,
    Provenance,
    SlideArtifact,
    SlideElement,
)


def _box(x: float, y: float, width: float, height: float) -> ObservedBoundingBox:
    return ObservedBoundingBox(
        x=x, y=y, width=width, height=height, page_width=1280, page_height=720
    )


def _artifact(
    elements: list[ComputedSlideElement],
    declared: list[SlideElement] | None = None,
    palette: list[str] | None = None,
) -> SlideArtifact:
    ir = DeclaredSlideIR(
        slide_id="slide",
        page_width=1280,
        page_height=720,
        elements=declared or [],
        theme_tokens={"palette": palette or []},
    )
    computed = ComputedSlideIR(
        slide_id="slide",
        page_width=1280,
        page_height=720,
        elements=elements,
        browser="Chromium",
        browser_version="test",
    )
    return SlideArtifact(
        artifact_id="artifact",
        source_uri="slide.html",
        source_sha256=hashlib.sha256(b"slide").hexdigest(),
        declared_ir=ir,
        computed_ir=computed,
        provenance=Provenance(creation_action="test"),
    )


def _element(
    element_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    role: str = "body",
    style: dict[str, str] | None = None,
    **kwargs: object,
) -> ComputedSlideElement:
    box = _box(x, y, width, height)
    values = {
        "element_id": element_id,
        "tag": "div",
        "semantic_role": role,
        "bbox": box,
        "visible_bbox": box,
        "client_width": width,
        "client_height": height,
        "scroll_width": width,
        "scroll_height": height,
        "computed_style": style or {},
    }
    values.update(kwargs)
    return ComputedSlideElement.model_validate(values)


@pytest.mark.unit
def test_declared_ir_extracts_stable_structure_tokens_and_warnings() -> None:
    html = """<html><head><style>:root{--brand:#123456;--safe-left:24px} .grid{grid-template-columns:1fr 1fr;font-size:24px}</style></head><body data-slide-id='s1'><section data-slidex-id='root' data-slidex-role='content'><h1 data-slidex-id='title' data-slidex-role='title' style='left:40px;top:20px;width:300px;height:50px'>Title</h1><img src='https://example.com/a.png'></section><script>run()</script></body></html>"""
    ir = extract_declared_ir(html)
    assert ir.slide_id == "s1"
    assert ir.elements[0].children[0].element_id == "title"
    assert ir.elements[0].children[1].element_id.startswith("auto-")
    assert ir.theme_tokens["palette"] == ["#123456"]
    assert ir.expected_roles[ir.elements[0].children[1].element_id] == "unknown"
    assert any("dynamic scripts" in warning for warning in ir.warnings)
    assert any("remote resources" in warning for warning in ir.warnings)


@pytest.mark.unit
def test_overlap_clean_and_defective_pair() -> None:
    inspector = OverlapInspector()
    clean = _artifact(
        [_element("a", 30, 30, 100, 100), _element("b", 150, 30, 100, 100)]
    )
    defective = _artifact(
        [_element("a", 30, 30, 100, 100), _element("b", 120, 30, 100, 100)]
    )
    assert inspector.inspect(clean)[0].status == InspectionStatus.PASS
    failure = inspector.inspect(defective)[0]
    assert failure.status == InspectionStatus.FAIL
    assert failure.element_ids == ["a", "b"]
    assert failure.repair_hint.action == "separate_elements"


@pytest.mark.unit
def test_alignment_needs_sibling_evidence_and_locates_outlier() -> None:
    inspector = AlignmentInspector(tolerance_px=2)
    insufficient = _artifact(
        [_element("a", 40, 40, 100, 30), _element("b", 41, 80, 100, 30)]
    )
    defective = _artifact(
        [
            _element("a", 40, 40, 100, 30),
            _element("b", 40, 80, 100, 30),
            _element("c", 50, 120, 100, 30),
        ]
    )
    assert inspector.inspect(insufficient)[0].status == InspectionStatus.NOT_APPLICABLE
    assert inspector.inspect(defective)[0].element_ids == ["c"]


@pytest.mark.unit
def test_margin_and_render_overflow_boundaries() -> None:
    margin = MarginInspector(safety_margin_px=24)
    clean = _artifact([_element("a", 24, 24, 100, 100)])
    defective = _artifact([_element("a", 23, 24, 100, 100)])
    assert margin.inspect(clean)[0].status == InspectionStatus.PASS
    assert margin.inspect(defective)[0].status == InspectionStatus.FAIL
    overflowing = _element("text", 40, 40, 100, 30, scroll_width=120)
    result = RenderOverflowInspector().inspect(_artifact([overflowing]))[0]
    assert (
        result.defect_class == DefectClass.G7 and result.status == InspectionStatus.FAIL
    )


@pytest.mark.unit
def test_typography_brand_color_and_terminology() -> None:
    title1 = _element(
        "t1",
        40,
        40,
        300,
        60,
        role="title",
        style={
            "fontSize": "32px",
            "fontFamily": "Arial",
            "fontWeight": "700",
            "color": "rgb(18, 52, 86)",
            "backgroundColor": "rgba(0,0,0,0)",
            "borderColor": "rgba(0,0,0,0)",
        },
    )
    title2 = _element(
        "t2",
        40,
        120,
        300,
        60,
        role="title",
        style={
            "fontSize": "24px",
            "fontFamily": "Arial",
            "fontWeight": "700",
            "color": "rgb(255, 0, 0)",
            "backgroundColor": "rgba(0,0,0,0)",
            "borderColor": "rgba(0,0,0,0)",
        },
    )
    artifact = _artifact(
        [title1, title2],
        declared=[
            SlideElement(element_id="d1", tag="p", text="Large Language Model"),
            SlideElement(element_id="d2", tag="p", text="Large-Language Models"),
        ],
        palette=["#123456"],
    )
    assert TypographyInspector().inspect(artifact)[0].status == InspectionStatus.FAIL
    color_results = BrandColorInspector().inspect(artifact)
    assert any(
        item.element_ids == ["t2"] and item.status == InspectionStatus.FAIL
        for item in color_results
    )
    terminology = TerminologyInspector(similarity_threshold=0.7).inspect(artifact)
    assert terminology[0].status == InspectionStatus.FAIL
    assert terminology[0].repair_hint.action == "normalize_terminology"


@pytest.mark.unit
def test_symbolic_critic_converts_inspector_exception_to_error() -> None:
    class Broken:
        name = "broken"
        version = "1"
        defect_class = DefectClass.G2

        def inspect(self, artifact: SlideArtifact):
            raise RuntimeError("boom")

    report = SymbolicCritic([Broken()]).inspect(_artifact([]))
    assert report.results[0].status == InspectionStatus.ERROR
    assert report.summary["error"] == 1


@pytest.mark.browser
@pytest.mark.asyncio
async def test_single_browser_load_outputs_ir_png_pdf_and_overlay(
    tmp_path: Path,
) -> None:
    html = tmp_path / "slide.html"
    html.write_text(
        """<html><style>html,body{margin:0;width:1280px;height:720px}.box{position:absolute;left:40px;top:40px;width:100px;height:30px;overflow:hidden;white-space:nowrap}</style><body><div class='box' data-slidex-id='text' data-slidex-role='body'>A long line that overflows the fixed box</div></body></html>"""
    )
    observation = await BrowserObserver().observe(
        html, tmp_path / "out", debug_overlay=True
    )
    element = observation.computed_ir.elements[0]
    assert observation.computed_ir.render_ready
    assert element.scroll_width > element.client_width
    assert (
        observation.screenshot_path.exists()
        and observation.pdf_path.exists()
        and observation.overlay_path.exists()
    )
