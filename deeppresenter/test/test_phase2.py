import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.browser import deterministic_fallback_id, validate_element_ids
from deeppresenter.slidex.models import (
    BoundingBox,
    ComputedSlideIR,
    DeclaredSlideIR,
    DefectClass,
    EpisodeManifest,
    InspectionReport,
    InspectionResult,
    InspectionStatus,
    Provenance,
    RewardBreakdown,
    SlideArtifact,
    SlideElement,
    TrajectoryStep,
)
from deeppresenter.utils.config import DeepPresenterConfig


def _declared_ir() -> DeclaredSlideIR:
    return DeclaredSlideIR(
        slide_id="slide-1",
        page_width=1280,
        page_height=720,
        elements=[
            SlideElement(
                element_id="title",
                tag="h1",
                semantic_role="title",
                text="Slidex",
                bbox=BoundingBox(
                    x=40,
                    y=30,
                    width=400,
                    height=80,
                    page_width=1280,
                    page_height=720,
                ),
            )
        ],
    )


@pytest.mark.unit
def test_phase2_models_round_trip() -> None:
    reward = RewardBreakdown(
        hard_constraints={"no_overflow": True},
        soft_scores={"layout": 0.9},
        aggregate=0.9,
        reward_version="1.0",
    )
    step = TrajectoryStep(step_index=0, action={"type": "generate"}, reward=reward)
    episode = EpisodeManifest(episode_id="episode", workspace_uri="file:///tmp/episode", steps=[step])
    inspection = InspectionReport(
        artifact_id="artifact",
        slide_id="slide-1",
        router_version="1.0",
        taxonomy_version="1.0",
        results=[
            InspectionResult(
                defect_class=DefectClass.G1,
                status=InspectionStatus.PASS,
                severity=0,
                confidence=1,
                inspector_version="geometry/1.0",
            )
        ],
    )

    for model in (_declared_ir(), reward, step, episode, inspection):
        assert type(model).model_validate_json(model.model_dump_json()) == model


@pytest.mark.unit
def test_phase2_rejects_invalid_schema_and_geometry() -> None:
    with pytest.raises(ValidationError, match="bounding box exceeds page bounds"):
        BoundingBox(x=1200, y=0, width=100, height=10, page_width=1280, page_height=720)
    with pytest.raises(ValidationError, match="duplicate element ID"):
        DeclaredSlideIR(
            slide_id="slide-1",
            page_width=1280,
            page_height=720,
            elements=[SlideElement(element_id="same", tag="div"), SlideElement(element_id="same", tag="p")],
        )
    with pytest.raises(ValidationError):
        EpisodeManifest.model_validate({"schema_version": "1.0", "episode_id": "old", "workspace_uri": "file:///tmp/old"})
    with pytest.raises(ValidationError):
        InspectionResult(
            defect_class="G1",
            status="unknown",
            severity=0,
            confidence=1,
            inspector_version="test",
        )


@pytest.mark.unit
def test_stable_id_helpers() -> None:
    assert deterministic_fallback_id("html/body/div[1]") == deterministic_fallback_id("html/body/div[1]")
    assert validate_element_ids(["title", ""]) == ["element 1 is missing data-slidex-id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_element_ids(["title", "title"])


@pytest.mark.unit
def test_artifact_store_writes_atomically_and_verifies_hashes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    episode = store.create_episode("episode-1", versions={"taxonomy": "1.0"})
    source = b"<html><body><h1 data-slidex-id='title'>Slidex</h1></body></html>"
    declared = _declared_ir()
    computed = ComputedSlideIR(
        slide_id="slide-1",
        page_width=1280,
        page_height=720,
        elements=[],
        browser="Chromium",
        browser_version="test",
    )
    artifact = SlideArtifact(
        artifact_id="pending",
        source_uri="source/slide.html",
        source_sha256=hashlib.sha256(source).hexdigest(),
        declared_ir=declared,
        computed_ir=computed,
        provenance=Provenance(
            creation_action="generate",
            model="test-model",
            versions={"html2pptx": "test", "libreoffice": "test"},
        ),
    )
    manifest = store.write_artifact(
        episode.episode_id,
        {"source/slide.html": source, "ir/declared.json": declared.model_dump_json()},
        artifact.provenance,
        artifact,
    )

    assert store.verify_artifact(episode.episode_id, manifest.artifact_id)
    artifact_dir = tmp_path / "store" / episode.episode_id / "artifacts" / manifest.artifact_id
    assert not list(artifact_dir.parent.glob(".tmp-*"))
    saved = json.loads((artifact_dir / "manifest.json").read_text())
    for name, reference in saved["files"].items():
        assert hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest() == reference["sha256"]


@pytest.mark.unit
def test_config_has_strict_slidex_defaults_and_rejects_unknown_fields(tmp_path: Path) -> None:
    mcp = tmp_path / "mcp.json"
    mcp.write_text("[]")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
offline_mode: true
mcp_config_file: {mcp}
research_agent: &model
  base_url: http://localhost:1/v1
  model: test
  api_key: test
design_agent: *model
long_context_model: *model
"""
    )
    parsed = DeepPresenterConfig.load_from_file(str(config))
    assert parsed.slidex.strict_export is True
    assert parsed.slidex.reference_policy == "on_defer"

    config.write_text(config.read_text() + "unknown_critical_field: true\n")
    with pytest.raises(ValidationError, match="unknown_critical_field"):
        DeepPresenterConfig.load_from_file(str(config))
