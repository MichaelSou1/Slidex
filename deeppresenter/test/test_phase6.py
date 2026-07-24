import hashlib
import json
from pathlib import Path

import pytest

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.critic import HybridCritic, persist_report
from deeppresenter.slidex.models import (
    ArtifactTrust,
    ComputedSlideElement,
    ComputedSlideIR,
    DeclaredSlideIR,
    DefectClass,
    InspectionContext,
    InspectionStatus,
    Provenance,
    SlideArtifact,
    SlideElement,
)
from deeppresenter.slidex.router import (
    EvidenceAvailability,
    FrozenCriticRouter,
    FrozenRouterConfig,
)
from deeppresenter.utils.config import SlidexConfig


def _artifact(
    *, overflow: bool = False, trust: ArtifactTrust = ArtifactTrust.TRUSTED_SOURCE
) -> SlideArtifact:
    elements = [
        SlideElement(
            element_id="title", tag="h1", semantic_role="title", text="Revenue"
        ),
        SlideElement(
            element_id="body", tag="p", semantic_role="body", text="Revenue grew"
        ),
    ]
    computed = [
        ComputedSlideElement(
            element_id="title",
            tag="h1",
            semantic_role="title",
            text="Revenue",
            bbox={
                "x": 40,
                "y": 40,
                "width": 400,
                "height": 60,
                "page_width": 1280,
                "page_height": 720,
            },
            visible_bbox={
                "x": 40,
                "y": 40,
                "width": 400,
                "height": 60,
                "page_width": 1280,
                "page_height": 720,
            },
            client_width=400,
            client_height=60,
            scroll_width=400,
            scroll_height=60,
        ),
        ComputedSlideElement(
            element_id="body",
            tag="p",
            semantic_role="body",
            text="Revenue grew",
            bbox={
                "x": 40,
                "y": 140,
                "width": 500,
                "height": 100,
                "page_width": 1280,
                "page_height": 720,
            },
            visible_bbox={
                "x": 40,
                "y": 140,
                "width": 500,
                "height": 100,
                "page_width": 1280,
                "page_height": 720,
            },
            client_width=500,
            client_height=100,
            scroll_width=520 if overflow else 500,
            scroll_height=100,
        ),
    ]
    return SlideArtifact(
        artifact_id="artifact",
        source_uri="slide.html",
        source_sha256=hashlib.sha256(b"slide").hexdigest(),
        declared_ir=DeclaredSlideIR(
            slide_id="slide", page_width=1280, page_height=720, elements=elements
        ),
        computed_ir=ComputedSlideIR(
            slide_id="slide",
            page_width=1280,
            page_height=720,
            elements=computed,
            browser="Chromium",
            browser_version="1",
        ),
        provenance=Provenance(creation_action="test"),
        trust=trust,
    )


@pytest.mark.unit
def test_frozen_router_mapping_and_hash_are_stable() -> None:
    config = FrozenRouterConfig()
    router = FrozenCriticRouter(config)
    context = InspectionContext(artifact=_artifact(), render_path="render.png")
    expected = {
        DefectClass.G2: "geometry.overlap",
        DefectClass.G3: "geometry.alignment",
        DefectClass.G4: "style.typography",
        DefectClass.G5: "style.brand_color",
        DefectClass.G6: "geometry.margin",
        DefectClass.S3: "terminology",
        DefectClass.S1: "title-body-mismatch",
        DefectClass.S4: "density",
        DefectClass.S2: "deck-semantic",
        DefectClass.S5: "deck-semantic",
        DefectClass.S6: "image-text-contradiction",
        DefectClass.G1: "geometry.declared_overflow",
        DefectClass.G7: "geometry.render_overflow",
    }
    assert {item.value: config.config_hash for item in DefectClass}
    assert len(config.config_hash) == 64
    for defect_class, first in expected.items():
        decision = router.route(
            defect_class,
            EvidenceAvailability.from_context(context),
            ArtifactTrust.TRUSTED_SOURCE,
        )
        assert decision.stages[0].inspector == first
        assert decision.reason


@pytest.mark.unit
def test_image_only_explicitly_downgrades_capability() -> None:
    context = InspectionContext(
        artifact=_artifact(trust=ArtifactTrust.IMAGE_ONLY), render_path="render.png"
    )
    decision = FrozenCriticRouter().route(
        DefectClass.G2,
        EvidenceAvailability.from_context(context),
        ArtifactTrust.IMAGE_ONLY,
    )
    assert not decision.stages
    assert decision.missing_evidence == ["trusted_native_ir"]
    assert "Image-only" in (decision.capability_limit or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_critic_without_models_keeps_symbolic_and_defers_neural() -> None:
    report = await HybridCritic(SlidexConfig()).inspect(
        InspectionContext(artifact=_artifact(overflow=True))
    )
    g7 = [item for item in report.results if item.defect_class == DefectClass.G7]
    s1 = [item for item in report.results if item.defect_class == DefectClass.S1]
    assert g7[0].status == InspectionStatus.FAIL
    assert s1[0].status == InspectionStatus.DEFER
    assert report.summary["fail"] >= 1
    assert report.summary["defer"] >= 1
    assert report.router_hash and len(report.routes) == len(DefectClass)


@pytest.mark.unit
def test_trusted_native_result_has_deterministic_priority_without_dropping_conflict() -> (
    None
):
    from deeppresenter.slidex.critic import _conflicts, _resolve_status
    from deeppresenter.slidex.models import InspectionResult

    results = [
        InspectionResult(
            defect_class=DefectClass.G7,
            status=InspectionStatus.PASS,
            severity=0,
            confidence=1,
            inspector_name="geometry.render_overflow",
            inspector_version="1.0",
        ),
        InspectionResult(
            defect_class=DefectClass.G7,
            status=InspectionStatus.FAIL,
            severity=1,
            confidence=0.8,
            inspector_name="unresolved-render-anomaly",
            inspector_version="atomic-neural/1.0",
        ),
    ]
    assert _conflicts(results, ArtifactTrust.TRUSTED_SOURCE) == [DefectClass.G7]
    assert (
        _resolve_status(results, ArtifactTrust.TRUSTED_SOURCE)[DefectClass.G7]
        == InspectionStatus.PASS
    )
    assert (
        _resolve_status(results, ArtifactTrust.RECOVERED)[DefectClass.G7]
        == InspectionStatus.FAIL
    )


@pytest.mark.unit
def test_report_is_persisted_and_retrievable(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    episode = store.create_episode("episode")
    artifact = _artifact()
    report = __import__("asyncio").run(
        HybridCritic(SlidexConfig()).inspect(InspectionContext(artifact=artifact))
    )
    uri = persist_report(
        store, episode.episode_id, report, parent_artifact_id=artifact.artifact_id
    )
    _, location = uri.split("://", 1)
    _, report_artifact_id, name = location.split("/", 2)
    restored = json.loads(store.read_artifact_file("episode", report_artifact_id, name))
    assert restored["router_hash"] == report.router_hash
    assert restored["summary"] == report.summary
