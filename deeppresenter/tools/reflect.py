import base64
import hashlib
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP
from mcp.types import ImageContent

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.browser import BrowserObserver, extract_declared_ir
from deeppresenter.slidex.critic import HybridCritic, persist_report
from deeppresenter.slidex.models import (
    InspectionContext,
    Provenance,
    RenderArtifact,
    RendererInfo,
    SlideArtifact,
)
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.log import info, set_logger
from deeppresenter.utils.webview import (
    PlaywrightConverter,
    convert_html_to_pptx,
    playwright_lifespan,
)
from pptagent.model_utils import _get_lid_model

mcp = FastMCP("Slidex", lifespan=playwright_lifespan)
CONFIG = DeepPresenterConfig.load_from_file(os.getenv("CONFIG_FILE"))
LID_MODEL = _get_lid_model()
REFLECTIVE_DESIGN = CONFIG.design_agent.is_multimodal and CONFIG.heavy_reflect


@mcp.tool()
async def inspect_slide(
    html_file: str,
    aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
) -> str:
    """Run the frozen Slidex hybrid critic and return its structured report."""
    html_path = Path(html_file).absolute()
    assert html_path.is_file() and html_path.suffix == ".html", (
        f"HTML path {html_path} does not exist or is not an HTML file"
    )
    await convert_html_to_pptx(html_path, aspect_ratio=aspect_ratio)
    output_dir = html_path.parent / ".slidex" / html_path.stem
    observation = await BrowserObserver().observe(
        html_path, output_dir, slide_id=html_path.stem, debug_overlay=True
    )
    declared = extract_declared_ir(
        html_path,
        slide_id=html_path.stem,
        global_css=html_path.parent / "global.css"
        if (html_path.parent / "global.css").exists()
        else None,
    )
    source = html_path.read_bytes()
    provenance = Provenance(
        creation_action="inspect_slide",
        versions={"taxonomy": CONFIG.slidex.taxonomy_version},
    )
    artifact = SlideArtifact(
        artifact_id="pending",
        source_uri=html_path.as_uri(),
        source_sha256=hashlib.sha256(source).hexdigest(),
        declared_ir=declared,
        computed_ir=observation.computed_ir,
        renders=[
            RenderArtifact(
                kind="html",
                uri=observation.screenshot_path.as_uri(),
                sha256=hashlib.sha256(
                    observation.screenshot_path.read_bytes()
                ).hexdigest(),
                width=int(observation.computed_ir.page_width),
                height=int(observation.computed_ir.page_height),
                renderer=RendererInfo(
                    name=observation.computed_ir.browser,
                    version=observation.computed_ir.browser_version,
                ),
            )
        ],
        provenance=provenance,
    )
    store = ArtifactStore(output_dir / "artifacts")
    episode = store.create_episode(
        versions={
            "taxonomy": CONFIG.slidex.taxonomy_version,
            "router": CONFIG.slidex.router_version,
        }
    )
    manifest = store.write_artifact(
        episode.episode_id,
        {
            f"source/{html_path.name}": html_path,
            "renders/render.png": observation.screenshot_path,
            "ir/declared.json": declared.model_dump_json(indent=2),
            "ir/computed.json": observation.computed_ir.model_dump_json(indent=2),
        },
        provenance,
        artifact,
    )
    artifact = artifact.model_copy(update={"artifact_id": manifest.artifact_id})
    report = await HybridCritic(
        CONFIG.slidex,
        critic_model=CONFIG.critic_model,
        semantic_model=CONFIG.semantic_model,
    ).inspect(
        InspectionContext(
            artifact=artifact, render_path=str(observation.screenshot_path)
        )
    )
    report_uri = persist_report(
        store, episode.episode_id, report, parent_artifact_id=manifest.artifact_id
    )
    return report.model_copy(update={"report_uri": report_uri}).model_dump_json(
        indent=2
    )


@mcp.tool()
async def render_slide(
    html_file: str,
    aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
) -> ImageContent:
    """Render a visual preview without making an inspection verdict."""
    html_path = Path(html_file).absolute()
    assert html_path.is_file() and html_path.suffix == ".html"
    pdf_path = Path(tempfile.mkdtemp()) / "slide.pdf"
    async with PlaywrightConverter() as converter:
        image_dir = await converter.convert_to_pdf([html_path], pdf_path, aspect_ratio)
    image_data = (image_dir / "slide_01.jpg").read_bytes()
    return ImageContent(
        type="image",
        data=f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}",
        mimeType="image/jpeg",
    )


@mcp.tool()
def inspect_manuscript(md_file: str) -> dict:
    """
    Inspect the markdown manuscript for general statistics and image asset validation.
    Args:
        md_file (str): The path to the markdown file
    """
    md_path = Path(md_file)
    assert md_path.exists(), f"file does not exist: {md_file}"
    assert md_file.lower().endswith(".md"), f"file is not a markdown file: {md_file}"

    with open(md_file, encoding="utf-8") as f:
        markdown = f.read()

    pages = [p for p in markdown.split("\n---\n") if p.strip()]
    result = defaultdict(list)
    result["num_pages"] = len(pages)
    label = LID_MODEL.predict(markdown[:1000].replace("\n", " "))
    result["language"] = label[0][0].replace("__label__", "")

    seen_images = set()
    for match in re.finditer(r"!\[(.*?)\]\((.*?)\)", markdown):
        label, path = match.group(1), match.group(2)
        path = path.split()[0].strip("\"'")

        if path in seen_images:
            continue
        seen_images.add(path)

        if re.match(r"https?://", path):
            result["warnings"].append(
                f"External link detected: {match.group(0)}, consider downloading to local storage."
            )
            continue

        if not (md_path.parent / path).exists() and not Path(path).exists():
            result["warnings"].append(f"Image file does not exist: {path}")

        if not label.strip():
            result["warnings"].append(f"Image {path} is missing alt text.")

        count = markdown.count(path)
        if count > 1:
            result["warnings"].append(
                f"Image {path} used {count} times in the whole presentation manuscript."
            )

    if len(result["warnings"]) == 0:
        result["success"].append(
            "Image asset validation passed: all referenced images exist."
        )

    return result


if __name__ == "__main__":
    work_dir = Path(os.environ["WORKSPACE"])
    assert work_dir.exists(), f"Workspace {work_dir} does not exist."
    os.chdir(work_dir)
    set_logger(f"task-{work_dir.stem}", work_dir / ".history" / "task.log")

    if REFLECTIVE_DESIGN:
        info("Reflective Design is enabled.")

    mcp.run(show_banner=False)
