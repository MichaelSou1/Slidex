"""Bridge frozen intrinsic critic controls to real OpenAI-compatible models.

This module turns a frozen ``EvaluationCase`` (a mutated PPTX plus its
rendered PNGs, produced by :mod:`deeppresenter.eval.pipeline`) into the
``SlideArtifact`` / ``InspectionContext`` objects that
:mod:`deeppresenter.slidex.critic` already knows how to consume, and exposes
four ``CriticCall`` implementations for the frozen controls defined in
``deeppresenter/eval/executors.py``:

- ``whole_rubric``: single VLM call over the complete rubric (``C0``/``C0_x10``).
- ``atomic``: single-defect atomic VLM query with no clean reference (``ATOMIC``/``VLM_ONLY``).
- ``reference``: order-controlled pairwise comparison against the clean twin.
- ``hybrid``: the frozen symbolic-neural-reference router (``FROZEN_HYBRID``).

All four share the same underlying rendered evidence so intrinsic accuracy
differences reflect the inspection mechanism, not different inputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from deeppresenter.slidex.critic import HybridCritic, _definition
from deeppresenter.slidex.export import RendererInfo, extract_pptx_structure, pptx_to_slide_artifacts
from deeppresenter.slidex.inspectors import AtomicNeuralClient
from deeppresenter.slidex.models import (
    ArtifactTrust,
    AtomicVerdict,
    DeclaredSlideIR,
    DefectClass,
    InspectionContext,
    InspectionStatus,
    Provenance,
    RenderArtifact,
    SlideArtifact,
)
from deeppresenter.utils.config import DeepPresenterConfig, LLM

from .models import Arm, EvaluationCase, Outcome
from .executors import CriticObservation

_RENDERER = RendererInfo(name="libreoffice", version="frozen-eval", options={})
_SLIDE_PART_RE = re.compile(r"slide(\d+)\.xml$")


def slide_index_from_part(slide_part: str | None) -> int:
    """Return the zero-based render index encoded by a PPTX slide XML part."""
    if not slide_part:
        return 0
    match = _SLIDE_PART_RE.search(slide_part)
    return int(match.group(1)) - 1 if match else 0


def pptx_slide_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(
            1
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )


def deck_outline(path: Path) -> list[str]:
    """First non-empty text run per slide, used as a narrative-order proxy."""
    count = pptx_slide_count(path)
    texts, _, _, _ = extract_pptx_structure(path, count)
    return [next((t for t in slide_texts if t.strip()), "") for slide_texts in texts]


def _usage_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get("total_tokens")
    return int(value) if isinstance(value, (int, float)) else 0


_OUTCOME_FROM_STATUS = {
    InspectionStatus.PASS: Outcome.PASS,
    InspectionStatus.FAIL: Outcome.FAIL,
    InspectionStatus.DEFER: Outcome.DEFER,
    InspectionStatus.NOT_APPLICABLE: Outcome.NOT_APPLICABLE,
    InspectionStatus.ERROR: Outcome.ERROR,
}


def _with_rate_limit_retry(call: Any, retries: int = 5, base_delay: float = 3.0) -> Any:
    """Retry a CriticCall on shared-quota 429s; other errors surface immediately."""

    async def wrapped(case: EvaluationCase, prompt: str, condition: str) -> CriticObservation:
        last: CriticObservation | None = None
        for attempt in range(retries):
            observation = await call(case, prompt, condition)
            is_rate_limited = observation.outcome == Outcome.ERROR and "RateLimitError" in str(
                observation.raw_output
            )
            if not is_rate_limited:
                return observation
            last = observation
            await asyncio.sleep(base_delay * (attempt + 1))
        assert last is not None
        return last

    return wrapped


class CaseArtifactBuilder:
    """Cache PPTX -> SlideArtifact conversion keyed by (uri, cache_dir)."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._artifacts: dict[str, list[SlideArtifact]] = {}
        self._outlines: dict[str, list[str]] = {}

    def _resolve(self, uri: str) -> Path:
        path = Path(uri)
        return path if path.is_absolute() else self.cache_dir / uri

    def artifacts(self, pptx_uri: str, render_uris: list[str]) -> list[SlideArtifact]:
        key = pptx_uri
        if key not in self._artifacts:
            pptx_path = self._resolve(pptx_uri)
            renders = [self._resolve(uri) for uri in render_uris]
            self._artifacts[key] = pptx_to_slide_artifacts(pptx_path, renders, _RENDERER)
        return self._artifacts[key]

    def outline(self, pptx_uri: str) -> list[str]:
        if pptx_uri not in self._outlines:
            self._outlines[pptx_uri] = deck_outline(self._resolve(pptx_uri))
        return self._outlines[pptx_uri]

    def render_path(self, uri: str) -> str:
        return self._resolve(uri).resolve().as_uri().replace("file://", "")

    def declared_only_artifact(self, pptx_uri: str) -> SlideArtifact:
        """Build a render-free artifact for deck-scope (S2) native-IR evidence."""
        pptx_path = self._resolve(pptx_uri)
        source_hash = hashlib.sha256(pptx_path.read_bytes()).hexdigest()
        return SlideArtifact(
            artifact_id=f"native-declared-{source_hash[:12]}",
            source_uri=pptx_path.resolve().as_uri(),
            source_sha256=source_hash,
            declared_ir=DeclaredSlideIR(
                slide_id="deck",
                page_width=1,
                page_height=1,
                elements=[],
            ),
            provenance=Provenance(creation_action="validate_native_pptx"),
            trust=ArtifactTrust.IMAGE_ONLY,
            missing_bookkeeping=["computed_ir", "native_element_geometry", "render"],
        )

    def image_only_artifact(self, image_uri: str) -> SlideArtifact:
        """Build an artifact for open-world image-only corpora (e.g. SlideAudit).

        There is no PPTX/native IR at all here: ``image_uri`` points directly
        at an externally-sourced slide image, so this never opens it as a
        zip/PPTX archive (unlike :meth:`declared_only_artifact`, which is for
        Zenodo10K deck-scope S2 cases that do have a real PPTX, just no
        per-slide render). ``renders[0]`` carries the image path itself so
        VLM inspectors can look at it directly; native predicates are
        unavailable and must be reported as capability downgrades.
        """
        image_path = self._resolve(image_uri)
        source_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        with Image.open(image_path) as img:
            width, height = img.size
        return SlideArtifact(
            artifact_id=f"native-image-only-{source_hash[:12]}",
            source_uri=image_path.resolve().as_uri(),
            source_sha256=source_hash,
            declared_ir=DeclaredSlideIR(
                slide_id="slide",
                page_width=width,
                page_height=height,
                elements=[],
            ),
            renders=[
                RenderArtifact(
                    kind="pdf",
                    uri=image_path.resolve().as_uri(),
                    sha256=source_hash,
                    width=width,
                    height=height,
                    renderer=RendererInfo(
                        name="external-source", version="image-only", options={}
                    ),
                )
            ],
            provenance=Provenance(creation_action="load_open_world_image"),
            trust=ArtifactTrust.IMAGE_ONLY,
            missing_bookkeeping=[
                "computed_ir",
                "native_element_geometry",
                "declared_ir_content",
                "trusted_native_ir",
            ],
        )


def build_context(case: EvaluationCase, builder: CaseArtifactBuilder) -> InspectionContext:
    """Construct the target/reference InspectionContext for one frozen case.

    ``case.labels`` is empty for negative (no-defect) cases, notably in the
    SlideAudit open-world corpus, so the label must never be dereferenced
    unconditionally here; it is only meaningful for the native-IR S2 branch
    at the end of this function.
    """
    if case.metadata.get("evidence_condition") == "image_only":
        # Open-world image-only corpora (e.g. SlideAudit): input_uri is a
        # real external slide image, never a PPTX/zip archive, so this must
        # not go through pptx_to_slide_artifacts()/deck_outline() at all.
        target = builder.image_only_artifact(case.input_uri)
        reference = (
            builder.image_only_artifact(case.clean_reference_uri)
            if case.clean_reference_uri
            else None
        )
        return InspectionContext(
            artifact=target,
            render_path=target.renders[0].uri.replace("file://", "") if target.renders else None,
            reference_artifact_id=reference.artifact_id if reference else None,
            reference_artifact=reference,
            reference_render_path=(
                reference.renders[0].uri.replace("file://", "")
                if reference and reference.renders
                else None
            ),
            deck_outline=[],
        )
    defective_renders = list(case.metadata.get("defective_render_uris", []))
    clean_renders = list(case.metadata.get("clean_render_uris", []))
    if not defective_renders:
        # Deck-scope defects (S2) carry no per-slide render; only native IR
        # evidence (here, the deck outline) is trustworthy for these cases.
        target = builder.declared_only_artifact(case.input_uri)
        reference = (
            builder.declared_only_artifact(case.clean_reference_uri)
            if case.clean_reference_uri
            else None
        )
        return InspectionContext(
            artifact=target,
            render_path=None,
            reference_artifact_id=reference.artifact_id if reference else None,
            reference_artifact=reference,
            reference_render_path=None,
            deck_outline=builder.outline(case.input_uri),
        )
    slide_part = case.metadata.get("target_slide_part")
    index = slide_index_from_part(slide_part)
    target_artifacts = builder.artifacts(case.input_uri, defective_renders)
    reference_artifacts = (
        builder.artifacts(case.clean_reference_uri, clean_renders)
        if case.clean_reference_uri
        else []
    )
    index = min(index, len(target_artifacts) - 1) if target_artifacts else 0
    target = target_artifacts[index] if target_artifacts else None
    reference = (
        reference_artifacts[index]
        if reference_artifacts and index < len(reference_artifacts)
        else None
    )
    if target is None:
        raise ValueError(f"case {case.case_id} has no target slide artifact")
    label = case.labels[0] if case.labels else None
    outline = (
        builder.outline(case.input_uri)
        if label is not None and label.defect_class == DefectClass.S2
        else []
    )
    return InspectionContext(
        artifact=target,
        render_path=target.renders[0].uri.replace("file://", "") if target.renders else None,
        reference_artifact_id=reference.artifact_id if reference else None,
        reference_artifact=reference,
        reference_render_path=(
            reference.renders[0].uri.replace("file://", "")
            if reference and reference.renders
            else None
        ),
        deck_outline=outline,
    )


class RealCriticExecutors:
    """Real-model CriticCall implementations sharing one frozen configuration."""

    def __init__(self, config: DeepPresenterConfig, cache_dir: Path) -> None:
        if config.critic_model is None:
            raise ValueError("config.critic_model is required for real critic execution")
        self.config = config
        self.builder = CaseArtifactBuilder(cache_dir)
        self.critic_model: LLM = config.critic_model
        self._hybrid = HybridCritic(config.slidex, critic_model=config.critic_model)

    async def whole_rubric(
        self, case: EvaluationCase, prompt: str, condition: str
    ) -> CriticObservation:
        """Single VLM call over the full rubric; the model must self-localize the defect."""
        started = time.perf_counter()
        context = build_context(case, self.builder)
        if not case.labels:
            raise NotImplementedError(
                f"whole_rubric() requires a single target defect class and does not support "
                "negative (no-label) cases such as SlideAudit true negatives; "
                "only the hybrid()/frozen_hybrid arm has whole-taxonomy negative-case semantics."
            )
        client = AtomicNeuralClient(self.critic_model, require_multimodal=True)
        render = context.render_path
        try:
            verdict, record = await client.inspect(
                case.labels[0].defect_class,
                "Any presentation-quality defect visible in this slide, using your full judgment.",
                {"note": "whole-rubric single-defect-free control"},
                image_paths=[render] if render else [],
            )
        except Exception as exc:  # capability/parse failures must be visible, not silent passes
            return CriticObservation(
                outcome=Outcome.ERROR,
                raw_output=str(exc),
                latency_seconds=time.perf_counter() - started,
            )
        return CriticObservation(
            outcome=_OUTCOME_FROM_STATUS[InspectionStatus(verdict.verdict)],
            defects=[case.labels[0].defect_class.value] if verdict.verdict == "fail" else [],
            confidence=verdict.confidence,
            element_id=verdict.element_ids[0] if verdict.element_ids else None,
            raw_output=verdict.model_dump(mode="json"),
            tokens=_usage_tokens(record.usage),
            latency_seconds=time.perf_counter() - started,
        )

    async def atomic(
        self, case: EvaluationCase, prompt: str, condition: str
    ) -> CriticObservation:
        """Single-defect atomic query without a clean reference (frozen negative control)."""
        started = time.perf_counter()
        context = build_context(case, self.builder)
        if not case.labels:
            raise NotImplementedError(
                f"atomic() requires a single target defect class and does not support "
                "negative (no-label) cases such as SlideAudit true negatives; "
                "only the hybrid()/frozen_hybrid arm has whole-taxonomy negative-case semantics."
            )
        defect = case.labels[0].defect_class
        client = AtomicNeuralClient(self.critic_model, require_multimodal=True)
        try:
            verdict, record = await client.inspect(
                defect,
                _definition(defect),
                {"target_slide_id": context.artifact.declared_ir.slide_id},
                image_paths=[context.render_path] if context.render_path else [],
            )
        except Exception as exc:
            return CriticObservation(
                outcome=Outcome.ERROR,
                raw_output=str(exc),
                latency_seconds=time.perf_counter() - started,
            )
        return CriticObservation(
            outcome=_OUTCOME_FROM_STATUS[InspectionStatus(verdict.verdict)],
            defects=[defect.value] if verdict.verdict == "fail" else [],
            confidence=verdict.confidence,
            element_id=verdict.element_ids[0] if verdict.element_ids else None,
            raw_output=verdict.model_dump(mode="json"),
            tokens=_usage_tokens(record.usage),
            latency_seconds=time.perf_counter() - started,
        )

    async def reference(
        self, case: EvaluationCase, prompt: str, condition: str
    ) -> CriticObservation:
        """Order-controlled pairwise comparison against the clean twin render."""
        from deeppresenter.slidex.inspectors import ReferenceInspector

        started = time.perf_counter()
        context = build_context(case, self.builder)
        if not case.labels:
            raise NotImplementedError(
                f"reference() requires a single target defect class and does not support "
                "negative (no-label) cases such as SlideAudit true negatives; "
                "only the hybrid()/frozen_hybrid arm has whole-taxonomy negative-case semantics."
            )
        defect = case.labels[0].defect_class
        client = AtomicNeuralClient(self.critic_model, require_multimodal=True)
        inspector = ReferenceInspector(client)
        result = await inspector.inspect(context, defect, _definition(defect))
        return CriticObservation(
            outcome=_OUTCOME_FROM_STATUS[result.status],
            defects=[defect.value] if result.status == InspectionStatus.FAIL else [],
            confidence=result.confidence,
            raw_output=[evidence.detail for evidence in result.evidence],
            latency_seconds=time.perf_counter() - started,
        )

    async def hybrid(
        self, case: EvaluationCase, prompt: str, condition: str
    ) -> CriticObservation:
        """Run the frozen HybridCritic and report the case's target defect class.

        ``case.labels`` is empty for negative (no-defect) cases -- notably in
        the open-world SlideAudit corpus -- which have no single "target"
        defect class to look up in ``report.resolved_status``. For those
        cases the correct arm-level verdict is derived across the *entire*
        taxonomy the router just inspected: any FAIL anywhere is a false
        positive (outcome FAIL); otherwise the case is a true negative
        (outcome PASS), unless the router could only DEFER on every class.
        """
        started = time.perf_counter()
        context = build_context(case, self.builder)
        report = await self._hybrid.inspect(context)
        capability_downgrade = report.capability_limits[0] if report.capability_limits else None
        if not case.labels:
            statuses = list(report.resolved_status.values())
            if InspectionStatus.FAIL in statuses:
                status = InspectionStatus.FAIL
            elif statuses and all(status == InspectionStatus.DEFER for status in statuses):
                status = InspectionStatus.DEFER
            else:
                status = InspectionStatus.PASS
            defects = [
                defect_class.value
                for defect_class, defect_status in report.resolved_status.items()
                if defect_status == InspectionStatus.FAIL
            ]
            confidences = [r.confidence for r in report.results if r.confidence is not None]
            confidence = max(confidences) if confidences else None
            return CriticObservation(
                outcome=_OUTCOME_FROM_STATUS[status],
                defects=defects,
                confidence=confidence,
                raw_output=report.model_dump(mode="json"),
                latency_seconds=time.perf_counter() - started,
                capability_downgrade=capability_downgrade,
            )
        defect = case.labels[0].defect_class
        status = report.resolved_status.get(defect, InspectionStatus.DEFER)
        matches = [r for r in report.results if r.defect_class == defect]
        confidence = matches[0].confidence if matches else None
        return CriticObservation(
            outcome=_OUTCOME_FROM_STATUS[status],
            defects=[defect.value] if status == InspectionStatus.FAIL else [],
            confidence=confidence,
            raw_output=report.model_dump(mode="json"),
            latency_seconds=time.perf_counter() - started,
            capability_downgrade=capability_downgrade,
        )


    async def mismatched_router(
        self, case: EvaluationCase, prompt: str, condition: str
    ) -> CriticObservation:
        """Negative control: compare the target slide against an unrelated slide."""
        from deeppresenter.slidex.inspectors import ReferenceInspector

        started = time.perf_counter()
        context = build_context(case, self.builder)
        if not case.labels:
            raise NotImplementedError(
                f"mismatched_router() requires a single target defect class and does not support "
                "negative (no-label) cases such as SlideAudit true negatives; "
                "only the hybrid()/frozen_hybrid arm has whole-taxonomy negative-case semantics."
            )
        defect = case.labels[0].defect_class
        mismatched_index = (slide_index_from_part(case.metadata.get("target_slide_part")) + 1) % max(
            1, len(self.builder.artifacts(case.input_uri, case.metadata.get("defective_render_uris", [])))
        )
        renders = case.metadata.get("defective_render_uris", [])
        mismatched_context = context.model_copy(
            update={
                "reference_render_path": self.builder.render_path(renders[mismatched_index])
                if renders
                else context.reference_render_path,
            }
        )
        client = AtomicNeuralClient(self.critic_model, require_multimodal=True)
        inspector = ReferenceInspector(client)
        result = await inspector.inspect(
            mismatched_context, defect, _definition(defect), run_clean_control=False
        )
        return CriticObservation(
            outcome=_OUTCOME_FROM_STATUS[result.status],
            defects=[defect.value] if result.status == InspectionStatus.FAIL else [],
            confidence=result.confidence,
            raw_output=[evidence.detail for evidence in result.evidence],
            latency_seconds=time.perf_counter() - started,
        )

    def call_for_arm(self, arm: Arm) -> Any:
        """Return the CriticCall bound to one frozen experimental arm."""
        mapping = {
            Arm.C0: self.whole_rubric,
            Arm.C0_X10: self.whole_rubric,
            Arm.C0_PLUS: self.whole_rubric,
            Arm.ATOMIC: self.atomic,
            Arm.VLM_ONLY: self.atomic,
            Arm.REFERENCE_DISABLED: self.atomic,
            Arm.FROZEN_HYBRID: self.hybrid,
            Arm.MISMATCHED_ROUTER: self.mismatched_router,
        }
        if arm not in mapping:
            raise ValueError(f"no real executor registered for arm {arm}")
        return _with_rate_limit_retry(mapping[arm])


def build_real_executors(cache_dir: Path, config_path: str | None = None) -> RealCriticExecutors:
    """Load the frozen Phase 13 model configuration and wire real critic calls."""
    config = DeepPresenterConfig.load_from_file(
        config_path or "deeppresenter/config.yaml.example"
    )
    return RealCriticExecutors(config, cache_dir)
