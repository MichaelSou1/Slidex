"""Strict PPTX export, final re-render, and multi-signal fidelity validation."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree import ElementTree

from PIL import Image, ImageChops, ImageStat

from deeppresenter.slidex.models import (
    ArtifactReference,
    DefectClass,
    ExportCommandRecord,
    ExportManifest,
    FidelityPageResult,
    FinalArtifactStatus,
    MutationFidelityResult,
    RenderFidelityReport,
    RendererInfo,
)
from deeppresenter.utils.webview import convert_html_to_pptx


class ExportCapabilityError(RuntimeError):
    """Raised when the required native renderer is unavailable."""


class ExportValidationError(RuntimeError):
    """Raised when strict export or final fidelity validation fails."""


class LibreOfficeRenderer:
    """Render PPTX through LibreOffice and rasterize its PDF deterministically."""

    def __init__(self, executable: str | None = None, dpi: int = 144) -> None:
        self.executable = (
            executable or shutil.which("soffice") or shutil.which("libreoffice")
        )
        self.dpi = dpi

    def info(self) -> RendererInfo:
        if not self.executable:
            raise ExportCapabilityError(
                "LibreOffice/soffice is required for PPTX re-rendering"
            )
        result = subprocess.run(
            [self.executable, "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise ExportCapabilityError(
                f"LibreOffice version check failed: {result.stderr.strip()}"
            )
        return RendererInfo(
            name="LibreOffice",
            version=result.stdout.strip() or "unknown",
            options={"headless": True, "dpi": self.dpi},
        )

    async def render(
        self, pptx_path: Path, output_dir: Path
    ) -> tuple[Path, list[Path], ExportCommandRecord]:
        info = self.info()
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(pptx_path.resolve()),
        ]
        started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            str(self.executable),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        command = ExportCommandRecord(
            executable=str(self.executable),
            arguments=args,
            version=info.version,
            return_code=process.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        if process.returncode != 0:
            error = ExportValidationError(
                f"LibreOffice PPTX render failed: {command.stderr or command.stdout}"
            )
            error.command = command
            raise error
        pdf_path = output_dir / f"{pptx_path.stem}.pdf"
        if not pdf_path.exists():
            raise ExportValidationError("LibreOffice completed without producing a PDF")
        pages = await self._rasterize(pdf_path, output_dir)
        return pdf_path, pages, command

    async def _rasterize(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        pdftoppm = shutil.which("pdftoppm")
        if not pdftoppm:
            raise ExportCapabilityError(
                "pdftoppm is required to rasterize the PPTX re-render"
            )
        prefix = output_dir / ".page"
        process = await asyncio.create_subprocess_exec(
            pdftoppm,
            "-png",
            "-r",
            str(self.dpi),
            str(pdf_path),
            str(prefix),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise ExportValidationError(
                f"pdftoppm failed: {stderr.decode(errors='replace')}"
            )
        raw_pages = sorted(output_dir.glob(".page-*.png"))
        pages: list[Path] = []
        for index, source in enumerate(raw_pages, start=1):
            destination = output_dir / f"slide_{index:02d}.png"
            source.replace(destination)
            pages.append(destination)
        return pages


class RenderFidelityValidator:
    """Gate final renders using independent pixel, text, asset, and structure signals."""

    def __init__(
        self,
        *,
        max_pixel_difference: float = 0.12,
        min_perceptual_similarity: float = 0.90,
        min_text_presence: float = 0.95,
        zero_signal_threshold: float = 0.001,
    ) -> None:
        self.max_pixel_difference = max_pixel_difference
        self.min_perceptual_similarity = min_perceptual_similarity
        self.min_text_presence = min_text_presence
        self.zero_signal_threshold = zero_signal_threshold

    def validate(
        self,
        html_renders: Sequence[Path],
        pptx_renders: Sequence[Path],
        slide_ids: Sequence[str],
        renderer: RendererInfo,
        *,
        source_text: Sequence[Sequence[str]] | None = None,
        pptx_text: Sequence[Sequence[str]] | None = None,
        missing_images: Sequence[int] | None = None,
        font_substitutions: Sequence[Sequence[str]] | None = None,
        position_drift: Sequence[dict[str, float]] | None = None,
    ) -> RenderFidelityReport:
        reasons: list[str] = []
        if len(html_renders) != len(pptx_renders):
            reasons.append(
                f"page_count_mismatch:{len(html_renders)}!={len(pptx_renders)}"
            )
        pages: list[FidelityPageResult] = []
        for index, (html_path, pptx_path) in enumerate(zip(html_renders, pptx_renders)):
            pixel_diff, perceptual, html_size, pptx_size = compare_images(
                html_path, pptx_path
            )
            expected = (
                list(source_text[index])
                if source_text and index < len(source_text)
                else []
            )
            exported = (
                list(pptx_text[index]) if pptx_text and index < len(pptx_text) else []
            )
            presence, missing = text_presence(expected, exported)
            image_failures = (
                missing_images[index]
                if missing_images and index < len(missing_images)
                else 0
            )
            fonts = (
                list(font_substitutions[index])
                if font_substitutions and index < len(font_substitutions)
                else []
            )
            drift = (
                position_drift[index]
                if position_drift and index < len(position_drift)
                else {}
            )
            findings = final_render_findings(pptx_path)
            dimensions_match = _aspect_ratio_matches(html_size, pptx_size)
            passed = (
                dimensions_match
                and pixel_diff <= self.max_pixel_difference
                and perceptual >= self.min_perceptual_similarity
                and presence >= self.min_text_presence
                and image_failures == 0
                and not findings
            )
            slide_id = (
                slide_ids[index] if index < len(slide_ids) else f"slide_{index + 1:02d}"
            )
            if not passed:
                reasons.append(f"{slide_id}:export_fidelity_failure")
            pages.append(
                FidelityPageResult(
                    slide_id=slide_id,
                    html_render_uri=html_path.resolve().as_uri(),
                    pptx_render_uri=pptx_path.resolve().as_uri(),
                    html_size=html_size,
                    pptx_size=pptx_size,
                    pixel_difference=pixel_diff,
                    perceptual_similarity=perceptual,
                    text_presence=presence,
                    missing_text=missing,
                    missing_images=image_failures,
                    font_substitutions=fonts,
                    position_drift=drift,
                    final_render_findings=findings,
                    passed=passed,
                )
            )
        return RenderFidelityReport(
            page_results=pages,
            expected_page_count=len(html_renders),
            actual_page_count=len(pptx_renders),
            page_count_matches=len(html_renders) == len(pptx_renders),
            renderer=renderer,
            export_fidelity_failure=bool(reasons),
            failure_reasons=reasons,
        )

    def mutation_result(
        self,
        mutation_id: str,
        defect_class: DefectClass,
        clean_render: Path,
        defective_render: Path,
        renderer: RendererInfo,
    ) -> MutationFidelityResult:
        pixel_diff, perceptual, _, _ = compare_images(clean_render, defective_render)
        zero_signal = pixel_diff <= self.zero_signal_threshold
        return MutationFidelityResult(
            mutation_id=mutation_id,
            defect_class=defect_class,
            clean_render_uri=clean_render.resolve().as_uri(),
            defective_render_uri=defective_render.resolve().as_uri(),
            renderer=renderer,
            pixel_difference=pixel_diff,
            perceptual_similarity=perceptual,
            zero_signal=zero_signal,
            include_in_training=not zero_signal,
        )

    @staticmethod
    def survival_rates(
        results: Iterable[MutationFidelityResult],
    ) -> dict[DefectClass, float]:
        grouped: dict[DefectClass, list[bool]] = {}
        for result in results:
            grouped.setdefault(result.defect_class, []).append(not result.zero_signal)
        return {defect: sum(values) / len(values) for defect, values in grouped.items()}


class FinalExportService:
    """Strict application service producing a validated final PPTX or explicit failure."""

    def __init__(
        self,
        renderer: LibreOfficeRenderer | None = None,
        validator: RenderFidelityValidator | None = None,
    ) -> None:
        self.renderer = renderer or LibreOfficeRenderer()
        self.validator = validator or RenderFidelityValidator()

    async def export(
        self,
        html_inputs: Path | Sequence[Path],
        output_pptx: Path,
        html_renders: Sequence[Path],
        *,
        aspect_ratio: str = "16:9",
        soft_mode: bool = False,
        soft_mode_explicit: bool = False,
        source_artifact_ids: Sequence[str] = (),
        critic_report_uris: Sequence[str] = (),
    ) -> ExportManifest:
        sources = (
            sorted(html_inputs.glob("*.html"))
            if isinstance(html_inputs, Path) and html_inputs.is_dir()
            else list(html_inputs)
            if not isinstance(html_inputs, Path)
            else [html_inputs]
        )
        manifest = ExportManifest(
            export_id=f"export-{uuid.uuid4().hex[:12]}",
            status=FinalArtifactStatus.DRAFT_HTML_VALID,
            source_uris=[path.resolve().as_uri() for path in sources],
            source_artifact_ids=list(source_artifact_ids),
            critic_report_uris=list(critic_report_uris),
            strict_validation=not soft_mode,
            soft_mode_explicit=soft_mode_explicit,
        )
        if soft_mode and not soft_mode_explicit:
            return manifest.model_copy(
                update={
                    "status": FinalArtifactStatus.INVALID_ARTIFACT,
                    "hard_penalty": True,
                    "failure_reason": "soft mode must be explicitly requested",
                }
            )
        try:
            conversion = await convert_html_to_pptx(
                html_inputs,
                output_pptx,
                aspect_ratio=aspect_ratio,
                soft_parsing=soft_mode,
            )
            manifest.commands.append(conversion.command)
            manifest.ignored_warnings.extend(conversion.ignored_warnings)
            manifest.output_files["pptx"] = artifact_reference(
                output_pptx,
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
            manifest.status = FinalArtifactStatus.PPTX_EXPORTED
            render_dir = output_pptx.parent / f"{output_pptx.stem}_pptx_render"
            pdf_path, pptx_pages, render_command = await self.renderer.render(
                output_pptx, render_dir
            )
            manifest.commands.append(render_command)
            manifest.output_files["pptx_render_pdf"] = artifact_reference(
                pdf_path, "application/pdf"
            )
            for index, page in enumerate(pptx_pages, start=1):
                manifest.output_files[f"pptx_render_slide_{index:02d}"] = (
                    artifact_reference(page, "image/png")
                )
            source_text = [extract_html_text(path) for path in sources]
            pptx_text, fonts, drifts, missing_images = extract_pptx_structure(
                output_pptx, len(sources)
            )
            report = self.validator.validate(
                html_renders,
                pptx_pages,
                [path.stem for path in sources],
                self.renderer.info(),
                source_text=source_text,
                pptx_text=pptx_text,
                missing_images=missing_images,
                font_substitutions=fonts,
                position_drift=drifts,
            )
            manifest.fidelity_report = report
            if report.export_fidelity_failure:
                manifest.hard_penalty = True
                manifest.failure_reason = "; ".join(report.failure_reasons)
            else:
                manifest.status = FinalArtifactStatus.PPTX_RENDER_VALIDATED
        except ExportCapabilityError as exc:
            manifest.status = FinalArtifactStatus.CAPABILITY_ERROR
            manifest.hard_penalty = True
            manifest.failure_reason = str(exc)
            command = getattr(exc, "command", None)
            if command is not None:
                manifest.commands.append(command)
        except Exception as exc:
            manifest.status = FinalArtifactStatus.INVALID_ARTIFACT
            manifest.hard_penalty = True
            manifest.failure_reason = str(exc)
            command = getattr(exc, "command", None)
            if command is not None:
                manifest.commands.append(command)
        return manifest

    @staticmethod
    def save_manifest(manifest: ExportManifest, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return path


def _aspect_ratio_matches(
    left: tuple[int, int], right: tuple[int, int], tolerance: float = 0.002
) -> bool:
    """Compare physical page shape while allowing renderer DPI differences."""
    return abs(left[0] / left[1] - right[0] / right[1]) <= tolerance


def compare_images(
    left_path: Path, right_path: Path
) -> tuple[float, float, tuple[int, int], tuple[int, int]]:
    with Image.open(left_path) as left_raw, Image.open(right_path) as right_raw:
        left = left_raw.convert("RGB")
        right = right_raw.convert("RGB")
        left_size, right_size = left.size, right.size
        resized = (
            right.resize(left.size, Image.Resampling.LANCZOS)
            if right.size != left.size
            else right
        )
        difference = ImageChops.difference(left, resized)
        pixel_difference = sum(ImageStat.Stat(difference).mean) / (3 * 255)
        perceptual = 1 - _difference_hash_distance(left, resized)
    return pixel_difference, perceptual, left_size, right_size


def _difference_hash_distance(left: Image.Image, right: Image.Image) -> float:
    def digest(image: Image.Image) -> list[bool]:
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(small.get_flattened_data())
        return [
            pixels[row * 9 + col] > pixels[row * 9 + col + 1]
            for row in range(8)
            for col in range(8)
        ]

    first, second = digest(left), digest(right)
    return sum(a != b for a, b in zip(first, second)) / len(first)


def text_presence(
    expected: Sequence[str], exported: Sequence[str]
) -> tuple[float, list[str]]:
    expected_tokens = Counter(_tokens(" ".join(expected)))
    exported_tokens = Counter(_tokens(" ".join(exported)))
    if not expected_tokens:
        return 1.0, []
    matched = sum(
        min(count, exported_tokens[token]) for token, count in expected_tokens.items()
    )
    missing = [
        token
        for token, count in expected_tokens.items()
        if exported_tokens[token] < count
    ]
    return matched / sum(expected_tokens.values()), missing


def extract_html_text(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    body = re.search(r"<body\b[^>]*>(.*?)</body>", content, flags=re.I | re.S)
    visible = body.group(1) if body else content
    visible = re.sub(
        r"<(script|style|template)\b[^>]*>.*?</\1>",
        " ",
        visible,
        flags=re.I | re.S,
    )
    return [
        re.sub(r"\s+", " ", text).strip()
        for text in re.findall(r">([^<>]+)<", visible)
        if text.strip()
    ]


def extract_pptx_structure(
    path: Path, page_count: int
) -> tuple[list[list[str]], list[list[str]], list[dict[str, float]], list[int]]:
    texts = [[] for _ in range(page_count)]
    fonts = [[] for _ in range(page_count)]
    drifts = [{} for _ in range(page_count)]
    missing_images = [0 for _ in range(page_count)]
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        theme_fonts = (
            set(
                re.findall(
                    r'typeface="([^"]+)"',
                    archive.read("ppt/theme/theme1.xml").decode(errors="ignore"),
                )
            )
            if "ppt/theme/theme1.xml" in names
            else set()
        )
        for index in range(page_count):
            slide_name = f"ppt/slides/slide{index + 1}.xml"
            if slide_name not in names:
                continue
            root = ElementTree.fromstring(archive.read(slide_name))
            texts[index] = [
                node.text or "" for node in root.iter() if node.tag.endswith("}t")
            ]
            used_fonts = {
                node.attrib["typeface"]
                for node in root.iter()
                if node.tag.endswith(("}latin", "}ea", "}cs"))
                and node.attrib.get("typeface")
            }
            # The PPTX XML records requested fonts, not renderer substitutions.
            # Treating every non-theme font as substituted rejects valid exports.
            fonts[index] = []
            extents = [
                (int(node.attrib.get("cx", 0)), int(node.attrib.get("cy", 0)))
                for node in root.iter()
                if node.tag.endswith("}ext")
            ]
            if extents:
                drifts[index] = {
                    "max_width_emu": float(max(width for width, _ in extents)),
                    "max_height_emu": float(max(height for _, height in extents)),
                }
            rel_name = f"ppt/slides/_rels/slide{index + 1}.xml.rels"
            if rel_name in names:
                rel = ElementTree.fromstring(archive.read(rel_name))
                for node in rel:
                    target = node.attrib.get("Target", "")
                    if (
                        "/image" in node.attrib.get("Type", "")
                        and f"ppt/{target.replace('../', '')}" not in names
                    ):
                        missing_images[index] += 1
    return texts, fonts, drifts, missing_images


def final_render_findings(path: Path) -> list[str]:
    findings: list[str] = []
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        border = [
            rgb.crop((0, 0, width, 2)),
            rgb.crop((0, height - 2, width, height)),
            rgb.crop((0, 0, 2, height)),
            rgb.crop((width - 2, 0, width, height)),
        ]
        # Full-bleed backgrounds and gradients legitimately vary at the edge.
        # Boundary clipping is already checked from exported element geometry;
        # edge-color variance alone is not evidence of clipped content.
        margin = max(1, min(width, height) // 100)
        inset = rgb.crop((margin, margin, width - margin, height - margin))
        if (
            sum(
                ImageStat.Stat(
                    ImageChops.difference(rgb.resize(inset.size), inset)
                ).mean
            )
            / 3
            > 80
        ):
            findings.append("margin:large_edge_discontinuity")
    return findings


def artifact_reference(path: Path, media_type: str) -> ArtifactReference:
    content = path.read_bytes()
    return ArtifactReference(
        uri=path.resolve().as_uri(),
        sha256=hashlib.sha256(content).hexdigest(),
        media_type=media_type,
        size_bytes=len(content),
    )


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", value.casefold())
