import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from deeppresenter.slidex.export import (
    FinalExportService,
    LibreOfficeRenderer,
    RenderFidelityValidator,
    extract_html_text,
)
from deeppresenter.slidex.models import (
    DefectClass,
    ExportCommandRecord,
    FinalArtifactStatus,
    RendererInfo,
)


def _image(path: Path, color: str = "white", text: str = "Slidex") -> Path:
    image = Image.new("RGB", (1280, 720), color)
    ImageDraw.Draw(image).text((100, 100), text, fill="black")
    image.save(path)
    return path


@pytest.mark.unit
def test_multi_signal_fidelity_and_zero_signal_mutation(tmp_path: Path) -> None:
    clean = _image(tmp_path / "clean.png")
    same = _image(tmp_path / "same.png")
    changed = _image(tmp_path / "changed.png", color="black", text="Changed")
    renderer = RendererInfo(name="test", version="1")
    validator = RenderFidelityValidator(max_pixel_difference=0.01)
    report = validator.validate(
        [clean],
        [same],
        ["slide_01"],
        renderer,
        source_text=[["Slidex"]],
        pptx_text=[["Slidex"]],
    )
    assert not report.export_fidelity_failure
    assert report.page_results[0].passed
    zero = validator.mutation_result("m1", DefectClass.G1, clean, same, renderer)
    signal = validator.mutation_result("m2", DefectClass.G1, clean, changed, renderer)
    assert zero.zero_signal and not zero.include_in_training
    assert not signal.zero_signal and signal.include_in_training
    assert validator.survival_rates([zero, signal]) == {DefectClass.G1: 0.5}


@pytest.mark.unit
def test_page_count_and_text_are_independent_hard_fidelity_signals(
    tmp_path: Path,
) -> None:
    html = _image(tmp_path / "html.png")
    pptx = _image(tmp_path / "pptx.png")
    report = RenderFidelityValidator().validate(
        [html, html],
        [pptx],
        ["slide_01", "slide_02"],
        RendererInfo(name="test", version="1"),
        source_text=[["Required phrase"]],
        pptx_text=[["Other phrase"]],
    )
    assert report.export_fidelity_failure
    assert not report.page_count_matches
    assert report.page_results[0].missing_text == ["required"]


@pytest.mark.unit
def test_html_text_excludes_head_metadata(tmp_path: Path) -> None:
    html = tmp_path / "slide.html"
    html.write_text(
        "<html><head><title>Duplicate title</title></head>"
        "<body><h1>Visible title</h1></body></html>",
        encoding="utf-8",
    )
    assert extract_html_text(html) == ["Visible title"]


@pytest.mark.unit
def test_fidelity_allows_tokenization_and_requested_fonts(tmp_path: Path) -> None:
    html = _image(tmp_path / "html.png")
    pptx = _image(tmp_path / "pptx.png")
    report = RenderFidelityValidator().validate(
        [html],
        [pptx],
        ["slide_01"],
        RendererInfo(name="test", version="1"),
        source_text=[["检索增强生成（RAG）：三步工作流程"]],
        pptx_text=[["检索增强生成", "（RAG）", "：三步工作流程"]],
        font_substitutions=[["PingFang SC"]],
    )
    assert report.page_results[0].passed
    assert report.page_results[0].text_presence == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_soft_mode_requires_explicit_request(tmp_path: Path) -> None:
    html = tmp_path / "slide_01.html"
    html.write_text("<html><body>Slidex</body></html>")
    manifest = await FinalExportService().export(
        [html], tmp_path / "out.pptx", [], soft_mode=True, soft_mode_explicit=False
    )
    assert manifest.status == FinalArtifactStatus.INVALID_ARTIFACT
    assert manifest.hard_penalty
    assert not manifest.commands


@pytest.mark.unit
@pytest.mark.asyncio
async def test_export_manifest_keeps_validation_failure_command(
    monkeypatch, tmp_path: Path
) -> None:
    html = tmp_path / "slide_01.html"
    html.write_text("<html><body>Slidex</body></html>")
    record = ExportCommandRecord(
        executable="node",
        arguments=["html2pptx_cli.js", "--validate"],
        version="test",
        return_code=1,
        stderr="validation failed",
        duration_ms=1,
    )

    class Failure(RuntimeError):
        command = record

    async def fail(*args, **kwargs):
        raise Failure("validation failed")

    monkeypatch.setattr("deeppresenter.slidex.export.convert_html_to_pptx", fail)
    manifest = await FinalExportService().export([html], tmp_path / "out.pptx", [])
    assert manifest.status == FinalArtifactStatus.INVALID_ARTIFACT
    assert manifest.hard_penalty
    assert manifest.commands == [record]


@pytest.mark.export
@pytest.mark.asyncio
async def test_strict_export_rerender_and_traceable_manifest(tmp_path: Path) -> None:
    if not LibreOfficeRenderer().executable:
        pytest.skip("LibreOffice is unavailable")
    html = tmp_path / "slide_01.html"
    html.write_text(
        """<!doctype html><html><head><style>
        html,body{width:1280px;height:720px;margin:0;overflow:hidden;background:#fff}
        h1{position:absolute;left:80px;top:60px;font:700 48px Arial;color:#111}
        </style></head><body><h1 data-slidex-id='title'>Slidex fidelity</h1></body></html>"""
    )
    from deeppresenter.slidex.browser import BrowserObserver

    observation = await BrowserObserver().observe(
        html, tmp_path / "html_render", slide_id="slide_01"
    )
    manifest = await FinalExportService().export(
        [html],
        tmp_path / "out.pptx",
        [observation.screenshot_path],
        source_artifact_ids=["source-artifact"],
        critic_report_uris=["artifact://episode/report/report.json"],
    )
    manifest_path = FinalExportService.save_manifest(
        manifest, tmp_path / "export_manifest.json"
    )
    restored = json.loads(manifest_path.read_text())
    assert (tmp_path / "out.pptx").exists()
    assert restored["source_artifact_ids"] == ["source-artifact"]
    assert len(restored["commands"]) == 2
    assert restored["commands"][0]["stdout"] is not None
    assert restored["output_files"]["pptx_render_slide_01"]["sha256"]
    assert manifest.status == FinalArtifactStatus.PPTX_RENDER_VALIDATED
    assert manifest.fidelity_report
    assert not manifest.fidelity_report.export_fidelity_failure
    assert not manifest.hard_penalty

@pytest.mark.asyncio
async def test_native_pptx_validation_is_backend_independent(tmp_path: Path) -> None:
    from deeppresenter.slidex.export import FinalExportService, pptx_to_slide_artifacts
    from deeppresenter.slidex.models import ExportCommandRecord, RendererInfo

    pptx = tmp_path / "native.pptx"
    with zipfile.ZipFile(pptx, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            '<p:sld xmlns:p="p" xmlns:a="a"><a:t>Native deck</a:t></p:sld>',
        )
    page = tmp_path / "page.png"
    Image.new("RGB", (1280, 720), "white").save(page)
    pdf = tmp_path / "native.pdf"
    pdf.write_bytes(b"pdf")

    class Renderer:
        def info(self) -> RendererInfo:
            return RendererInfo(name="fake", version="1")

        async def render(self, _pptx: Path, _output: Path):
            return pdf, [page], ExportCommandRecord(
                executable="fake",
                version="1",
                return_code=0,
                duration_ms=1,
            )

    service = FinalExportService(renderer=Renderer())
    manifest = await service.validate_pptx(pptx, expected_page_count=1)
    assert manifest.status == FinalArtifactStatus.PPTX_RENDER_VALIDATED
    artifacts = pptx_to_slide_artifacts(
        pptx, [page], manifest.fidelity_report.renderer
    )
    assert artifacts[0].declared_ir.elements[0].text == "Native deck"
    assert artifacts[0].trust.value == "image_only"
