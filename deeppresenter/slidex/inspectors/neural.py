"""Atomic, stateless neural inspectors for semantic and unresolved render defects."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import mimetypes
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from deeppresenter.slidex.models import (
    AtomicVerdict,
    ComputedSlideElement,
    DefectClass,
    Evidence,
    EvidenceSource,
    InspectionContext,
    InspectionResult,
    InspectionStatus,
    NeuralCallRecord,
    PairwiseVerdict,
    RepairHint,
    SlideArtifact,
    SlideElement,
)
from deeppresenter.utils.config import LLM

PROMPT_VERSION = "atomic-neural/1.0"
_JSON_INSTRUCTION = "Return only an object matching the supplied JSON schema."


class NeuralCapabilityError(RuntimeError):
    """Raised when a configured provider cannot satisfy required evidence/schema."""


# Most gateways reject request bodies above a few MB; some source decks render
# at unexpectedly high pixel dimensions (e.g. poster-sized aspect ratios at
# fixed DPI), producing multi-megabyte PNGs that a naive base64 encode would
# balloon past the gateway limit. This is a transport-layer safeguard only:
# it never touches the frozen render artifacts on disk, only the bytes sent
# to the model for this one call.
_MAX_ENCODED_IMAGE_BYTES = 3_500_000


def _encode_image_for_transport(path: str) -> tuple[str, str]:
    """Return (base64, media_type), downscaling oversized renders before encoding.

    PNG is lossless, so pixel-dimension downscaling alone barely shrinks dense
    renders (photographic backgrounds, gradients). Once the losslessly encoded
    render exceeds the transport guard, this falls back to re-encoding as JPEG
    with progressively lower resolution/quality, which is sufficient for a VLM
    to still judge geometry/semantic defects. The scale/quality ladder is fixed
    (not adaptive to content), so repeated calls on the same frozen render are
    deterministic.
    """
    data = Path(path).read_bytes()
    media_type = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(data).decode()
    if len(encoded) <= _MAX_ENCODED_IMAGE_BYTES:
        return encoded, media_type
    image = Image.open(io.BytesIO(data))
    image.load()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    candidate = encoded
    for scale, quality in (
        (1.0, 90),
        (0.75, 88),
        (0.5, 85),
        (0.35, 82),
        (0.25, 80),
        (0.18, 78),
        (0.12, 75),
    ):
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        resized = image if scale == 1.0 else image.resize((width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=quality, optimize=True)
        candidate = base64.b64encode(buffer.getvalue()).decode()
        if len(candidate) <= _MAX_ENCODED_IMAGE_BYTES:
            return candidate, "image/jpeg"
    # Last resort: return the smallest attempt even if still above the guard;
    # the transport error (if any) will surface as a normal ERROR outcome.
    return candidate, "image/jpeg"


class AtomicNeuralClient:
    """Issue independent structured requests and retain replay metadata."""

    def __init__(
        self,
        model: LLM,
        *,
        require_multimodal: bool = False,
        max_concurrent: int = 4,
        cache_results: bool = False,
    ) -> None:
        if require_multimodal and model.is_multimodal is not True:
            raise NeuralCapabilityError("critic model does not declare image support")
        if hasattr(model, "require_capabilities"):
            try:
                model.require_capabilities("structured_output")
                if require_multimodal:
                    model.require_capabilities("vision")
            except Exception as exc:
                raise NeuralCapabilityError(str(exc)) from exc
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.model = model
        self.records: list[NeuralCallRecord] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache_results = cache_results
        self._cache: dict[str, tuple[AtomicVerdict, NeuralCallRecord]] = {}

    async def inspect(
        self,
        defect_class: DefectClass,
        definition: str,
        evidence: dict[str, Any],
        *,
        image_paths: Sequence[str] = (),
        condition: str | None = None,
    ) -> tuple[AtomicVerdict, NeuralCallRecord]:
        prompt = self._atomic_prompt(defect_class, definition, evidence)
        image_hashes = [
            hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in image_paths
        ]
        cache_key = hashlib.sha256(
            json.dumps(
                [defect_class.value, prompt, image_hashes, condition],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = self._cache.get(cache_key) if self._cache_results else None
        if cached is not None:
            return cached
        content = self._content(prompt, image_paths)
        async with self._semaphore:
            started = time.perf_counter()
            response = await self.model.run(
                [{"role": "user", "content": content}],
                response_format=AtomicVerdict,
                retry_times=1,
            )
        latency = (time.perf_counter() - started) * 1000
        raw = response.choices[0].message.content or ""
        verdict = AtomicVerdict.model_validate_json(raw)
        record = self._record(defect_class, prompt, response, raw, latency, condition)
        self.records.append(record)
        if self._cache_results:
            self._cache[cache_key] = (verdict, record)
        return verdict, record

    async def compare(
        self,
        defect_class: DefectClass,
        definition: str,
        left_path: str,
        right_path: str,
        *,
        condition: str,
    ) -> tuple[PairwiseVerdict, NeuralCallRecord]:
        prompt = (
            f"Protocol {PROMPT_VERSION}. Compare only defect {defect_class.value}.\n"
            f"Operational definition: {definition}\n"
            "Image 1 is LEFT and image 2 is RIGHT. Decide which has less of this defect. "
            "Use tie when observably equivalent and defer when evidence is insufficient. "
            "Do not assess general visual quality. " + _JSON_INSTRUCTION
        )
        async with self._semaphore:
            started = time.perf_counter()
            response = await self.model.run(
                [
                    {
                        "role": "user",
                        "content": self._content(prompt, [left_path, right_path]),
                    }
                ],
                response_format=PairwiseVerdict,
                retry_times=1,
            )
        latency = (time.perf_counter() - started) * 1000
        raw = response.choices[0].message.content or ""
        verdict = PairwiseVerdict.model_validate_json(raw)
        record = self._record(defect_class, prompt, response, raw, latency, condition)
        self.records.append(record)
        return verdict, record

    def build_content(
        self, prompt: str, image_paths: Sequence[str]
    ) -> str | list[dict[str, Any]]:
        """Public wrapper so other critics can reuse this client's multimodal encoding."""
        return self._content(prompt, image_paths)

    def _content(
        self, prompt: str, image_paths: Sequence[str]
    ) -> str | list[dict[str, Any]]:
        if not image_paths:
            return prompt
        if self.model.is_multimodal is not True:
            raise NeuralCapabilityError("provider does not support image input")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            encoded, media_type = _encode_image_for_transport(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        return content

    @staticmethod
    def _atomic_prompt(
        defect_class: DefectClass, definition: str, evidence: dict[str, Any]
    ) -> str:
        return (
            f"Protocol {PROMPT_VERSION}. Inspect exactly one defect class: {defect_class.value}.\n"
            f"Operational definition: {definition}\n"
            "Verdict must be pass, fail, or defer. Incompleteness alone is not contradiction. "
            "Localize with supplied element IDs and/or normalized bounding boxes. Evidence must "
            "state observable facts. A repair suggestion does not affect the verdict. Do not "
            "assess general quality. Mutation metadata and labels are intentionally absent.\n"
            f"Evidence:\n{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}\n{_JSON_INSTRUCTION}"
        )

    def _record(
        self,
        defect_class: DefectClass,
        prompt: str,
        response: Any,
        raw: str,
        latency_ms: float,
        condition: str | None,
    ) -> NeuralCallRecord:
        call = getattr(self.model, "last_call", None) or {}
        endpoint = self.model._endpoints[0]
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        return NeuralCallRecord(
            defect_class=defect_class,
            endpoint_identifier=call.get("endpoint_identifier")
            or self.model.identifier
            or endpoint.base_url
            or "openai-default",
            model=call.get("model") or endpoint.model,
            sampling_parameters=call.get("sampling_parameters")
            or endpoint.sampling_parameters,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=raw,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            condition=condition,
        )


class AtomicInspector:
    """Base class converting strict atomic verdicts into inspection results."""

    defect_class: DefectClass
    name = "atomic-neural"
    version = PROMPT_VERSION
    definition: str

    def __init__(self, client: AtomicNeuralClient) -> None:
        self.client = client

    async def inspect(self, context: InspectionContext) -> InspectionResult:
        started = time.perf_counter()
        try:
            prepared = self.prepare(context)
            if isinstance(prepared, InspectionResult):
                return prepared
            evidence, images = prepared
            verdict, record = await self.client.inspect(
                self.defect_class, self.definition, evidence, image_paths=images
            )
            return self._result(context.artifact, verdict, record, started)
        except Exception as exc:
            return self._error(context.artifact, exc, started)

    def prepare(
        self, context: InspectionContext
    ) -> tuple[dict[str, Any], list[str]] | InspectionResult:
        raise NotImplementedError

    def _result(
        self,
        artifact: SlideArtifact,
        verdict: AtomicVerdict,
        record: NeuralCallRecord,
        started: float,
    ) -> InspectionResult:
        status = InspectionStatus(verdict.verdict)
        evidence = [
            Evidence(
                source=self.evidence_source,
                detail=item,
                element_ids=verdict.element_ids,
            )
            for item in verdict.evidence
        ]
        hint = None
        if verdict.repair_suggestion:
            hint = RepairHint(
                action="policy_edit",
                targets=verdict.element_ids,
                explanation=verdict.repair_suggestion,
            )
        if status == InspectionStatus.DEFER:
            hint = RepairHint(
                action="provide_evidence", explanation=verdict.defer_reason
            )
        usage = record.usage
        cost = float(usage.get("cost", 0) or 0)
        return InspectionResult(
            defect_class=self.defect_class,
            status=status,
            severity=verdict.severity,
            confidence=verdict.confidence,
            evidence=evidence,
            element_ids=verdict.element_ids,
            repair_hint=hint,
            inspector_name=self.name,
            inspector_version=self.version,
            input_hash=hashlib.sha256(artifact.model_dump_json().encode()).hexdigest(),
            latency_ms=(time.perf_counter() - started) * 1000,
            cost=cost,
        )

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.RENDER

    def _defer(self, artifact: SlideArtifact, reason: str) -> InspectionResult:
        return InspectionResult(
            defect_class=self.defect_class,
            status=InspectionStatus.DEFER,
            severity=0,
            confidence=1,
            evidence=[Evidence(source=self.evidence_source, detail=reason)],
            repair_hint=RepairHint(action="provide_evidence", explanation=reason),
            inspector_name=self.name,
            inspector_version=self.version,
        )

    def _error(
        self, artifact: SlideArtifact, exc: Exception, started: float
    ) -> InspectionResult:
        action = (
            "configure_provider_capability"
            if isinstance(exc, NeuralCapabilityError)
            else "retry_inspection"
        )
        return InspectionResult(
            defect_class=self.defect_class,
            status=InspectionStatus.ERROR,
            severity=0,
            confidence=0,
            evidence=[
                Evidence(
                    source=self.evidence_source, detail=f"{type(exc).__name__}: {exc}"
                )
            ],
            repair_hint=RepairHint(action=action, explanation=str(exc)),
            inspector_name=self.name,
            inspector_version=self.version,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class TitleBodyMismatchInspector(AtomicInspector):
    defect_class = DefectClass.S1
    name = "title-body-mismatch"
    definition = "Fail only when body claims contradict or clearly concern a different topic than the title; missing detail alone passes."

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.DECLARED_IR

    def prepare(self, context: InspectionContext) -> tuple[dict[str, Any], list[str]]:
        elements = _flatten(context.artifact.declared_ir.elements)
        titles = [
            _element_payload(item) for item in elements if item.semantic_role == "title"
        ]
        bodies = [
            _element_payload(item)
            for item in elements
            if item.semantic_role in {"body", "content", "caption", "claim"}
        ]
        images = [context.render_path] if context.render_path else []
        return {"titles": titles, "bodies": bodies}, images


class DensityInspector(AtomicInspector):
    defect_class = DefectClass.S4
    name = "density"
    definition = "Classify only over-packed or under-packed information density relative to the page role; intentional minimal title slides pass."

    def __init__(
        self,
        client: AtomicNeuralClient,
        *,
        over_chars: int = 900,
        over_elements: int = 35,
        under_chars: int = 20,
    ) -> None:
        super().__init__(client)
        self.over_chars = over_chars
        self.over_elements = over_elements
        self.under_chars = under_chars

    async def inspect(self, context: InspectionContext) -> InspectionResult:
        statistics = _density_statistics(context.artifact)
        if (
            statistics["character_count"] > self.over_chars
            or statistics["element_count"] > self.over_elements
        ):
            verdict = AtomicVerdict(
                verdict="fail",
                severity=1,
                confidence=1,
                evidence=[f"Deterministic density threshold exceeded: {statistics}"],
                element_ids=statistics["element_ids"],
                repair_suggestion="Reduce or split dense content.",
            )
            fake = NeuralCallRecord(
                defect_class=self.defect_class,
                endpoint_identifier="deterministic",
                model="none",
                latency_ms=0,
                raw_response=verdict.model_dump_json(),
                prompt_hash="0" * 64,
            )
            return self._result(context.artifact, verdict, fake, time.perf_counter())
        return await super().inspect(context)

    def prepare(self, context: InspectionContext) -> tuple[dict[str, Any], list[str]]:
        stats = _density_statistics(context.artifact)
        stats["possible_under_packed"] = stats["character_count"] < self.under_chars
        return stats, [context.render_path] if context.render_path else []


class ImageTextContradictionInspector(AtomicInspector):
    defect_class = DefectClass.S6
    name = "image-text-contradiction"
    definition = "For exactly one supplied image and adjacent caption or claim, fail only when visible image content contradicts that text. Defer when the image cannot establish the claim."

    def __init__(self, client: AtomicNeuralClient, image_id: str, text_id: str) -> None:
        super().__init__(client)
        self.image_id = image_id
        self.text_id = text_id

    def prepare(
        self, context: InspectionContext
    ) -> tuple[dict[str, Any], list[str]] | InspectionResult:
        if not context.render_path:
            return self._defer(context.artifact, "render_required")
        elements = {
            item.element_id: item
            for item in _flatten(context.artifact.declared_ir.elements)
        }
        if self.image_id not in elements or self.text_id not in elements:
            return self._defer(context.artifact, "image_claim_pair_missing")
        return {
            "image_element_id": self.image_id,
            "text": _element_payload(elements[self.text_id]),
        }, [context.render_path]


class RenderAnomalyInspector(AtomicInspector):
    defect_class = DefectClass.G7
    name = "unresolved-render-anomaly"
    definition = "Determine only whether content crosses or is clipped by its specified container boundary in the rendered slide."

    def __init__(self, client: AtomicNeuralClient, target_id: str) -> None:
        super().__init__(client)
        self.target_id = target_id

    def prepare(
        self, context: InspectionContext
    ) -> tuple[dict[str, Any], list[str]] | InspectionResult:
        if context.artifact.computed_ir:
            computed = {
                item.element_id: item
                for item in _flatten(context.artifact.computed_ir.elements)
            }
            target = computed.get(self.target_id)
            if target and (
                target.clipped
                or target.partially_outside_page
                or target.scroll_width > target.client_width
                or target.scroll_height > target.client_height
            ):
                return InspectionResult(
                    defect_class=self.defect_class,
                    status=InspectionStatus.FAIL,
                    severity=1,
                    confidence=1,
                    evidence=[
                        Evidence(
                            source=EvidenceSource.COMPUTED_IR,
                            detail="DOM geometry deterministically establishes overflow",
                            element_ids=[self.target_id],
                        )
                    ],
                    element_ids=[self.target_id],
                    repair_hint=RepairHint(
                        action="resize_container", targets=[self.target_id]
                    ),
                    inspector_name=self.name,
                    inspector_version=self.version,
                )
        if not context.render_path:
            return self._defer(context.artifact, "render_required")
        return {
            "target_element_id": self.target_id,
            "instruction": "Use the target bounding box visible in the supplied overlay.",
        }, [context.render_path]


class RenderOnlyGeometryInspector(AtomicInspector):
    """Open-world image-only downgrade for symbolic-only defect classes.

    Geometry/style defects (G1-G6) and deck-level semantics (S2, S3, S5)
    normally require computed_ir or native XML that image-only artifacts do
    not carry. Rather than defer unconditionally, this issues one atomic VLM
    query against the rendered pixels so an image-only pipeline still yields
    a verdict, at the coarser precision the frozen router already discloses
    via capability_limit.
    """

    name = "render-only-geometry"

    def __init__(self, client: AtomicNeuralClient, defect_class: DefectClass, definition: str) -> None:
        super().__init__(client)
        self.defect_class = defect_class
        self.definition = definition

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.RENDER

    def prepare(
        self, context: InspectionContext
    ) -> tuple[dict[str, Any], list[str]] | InspectionResult:
        if not context.render_path:
            return self._defer(context.artifact, "render_required")
        return (
            {"note": "Image-only artifact; judge from rendered pixels alone."},
            [context.render_path],
        )


_RENDER_ONLY_DEFINITIONS: dict[DefectClass, str] = {
    DefectClass.G1: "text or content that overflows, is clipped by, or does not fit within its visible container",
    DefectClass.G2: "two or more elements that visibly overlap each other in a way that obscures content",
    DefectClass.G3: "elements that are visibly misaligned relative to other elements sharing the same row or column",
    DefectClass.G4: "text rendered at a visibly inconsistent or clashing font size relative to sibling text",
    DefectClass.G5: "a color that is visibly off-palette or clashes with the slide's established color scheme",
    DefectClass.G6: "content that visibly crosses or sits flush against the slide's safe margin/edge",
    DefectClass.S2: "an ordered sequence of slides whose visible narrative order contains a clear break",
    DefectClass.S3: "inconsistent terminology or labeling for what is visibly the same concept across slides",
    DefectClass.S5: "a visibly missing step or section that the slide's own structure implies should be present",
}


class DeckSemanticInspector(AtomicInspector):
    name = "deck-semantic"

    def __init__(
        self,
        client: AtomicNeuralClient,
        defect_class: Literal[DefectClass.S2, DefectClass.S5],
    ) -> None:
        if defect_class not in {DefectClass.S2, DefectClass.S5}:
            raise ValueError("deck semantic inspector supports only S2 or S5")
        super().__init__(client)
        self.defect_class = defect_class
        self.definition = (
            "Fail only when the ordered slide titles/summaries contain a clear, high-confidence "
            "narrative-order break (not merely a terse or minimal transition). When uncertain, "
            "prefer pass or defer over fail."
            if defect_class == DefectClass.S2
            else (
                "Fail only when a required step explicitly implied by the task description is "
                "absent from the slides. If no 'approved_outline' key is present in evidence, "
                "no fixed structure was approved for this deck: judge solely against the task "
                "description's own implied steps, and the absence of that key is not itself "
                "evidence of a missing step. This check is about missing content only: extra "
                "slides beyond what the task lists (e.g. an added intro or conclusion slide) "
                "are never a failure on their own and must not be flagged or recommended for "
                "removal; only report a fail when a required step is truly absent. A slide "
                "count that already satisfies any page-count range stated in the task is never "
                "itself evidence of missing depth or content -- do not fail on reasoning like "
                "'N pages may indicate insufficient depth' when N is within the requested range; "
                "judge only whether a specific, named step or section is absent."
            )
        )

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.DECK_TEXT

    def prepare(
        self, context: InspectionContext
    ) -> tuple[dict[str, Any], list[str]] | InspectionResult:
        if (
            self.defect_class == DefectClass.S5
            and not context.task
            and not context.approved_outline
        ):
            return InspectionResult(
                defect_class=self.defect_class,
                status=InspectionStatus.NOT_APPLICABLE,
                severity=0,
                confidence=1,
                evidence=[
                    Evidence(
                        source=EvidenceSource.DECK_TEXT,
                        detail="Task has no approved fixed logical structure",
                    )
                ],
                inspector_name=self.name,
                inspector_version=self.version,
            )
        # Omit outline/approved_outline keys entirely when empty rather than
        # sending an empty-list literal: an empty-list value in the evidence
        # JSON reads to the model as an observed "structure is missing" signal
        # and biases S5 toward a false fail on every deck (no E2E caller sets
        # a planner-approved outline today, so this previously fired on 100%
        # of decks). Presence/absence of the key is the intended signal now.
        evidence: dict[str, Any] = {"slide_summaries": context.slide_summaries}
        if context.deck_outline:
            evidence["outline"] = context.deck_outline
        if context.approved_outline:
            evidence["approved_outline"] = context.approved_outline
        if context.task:
            evidence["task"] = context.task
        return evidence, []


def _flatten(elements: Sequence[SlideElement | ComputedSlideElement]) -> list[Any]:
    result: list[Any] = []
    for element in elements:
        result.append(element)
        result.extend(_flatten(element.children))
    return result


def _element_payload(element: SlideElement) -> dict[str, str | None]:
    return {
        "element_id": element.element_id,
        "role": element.semantic_role,
        "text": element.text,
    }


def _density_statistics(artifact: SlideArtifact) -> dict[str, Any]:
    elements = _flatten(
        artifact.computed_ir.elements
        if artifact.computed_ir
        else artifact.declared_ir.elements
    )
    text_elements = [item for item in elements if item.text.strip()]
    area = artifact.declared_ir.page_width * artifact.declared_ir.page_height
    occupied = sum(item.bbox.width * item.bbox.height for item in elements if item.bbox)
    font_sizes = []
    for item in elements:
        raw = (getattr(item, "computed_style", {}) or item.style).get("fontSize", "")
        try:
            font_sizes.append(float(str(raw).removesuffix("px")))
        except ValueError:
            pass
    return {
        "character_count": sum(len(item.text) for item in text_elements),
        "element_count": len(elements),
        "occupied_area_ratio": min(1.0, occupied / area),
        "whitespace_ratio": max(0.0, 1 - occupied / area),
        "minimum_font_size_px": min(font_sizes) if font_sizes else None,
        "element_ids": [item.element_id for item in text_elements],
        "page_role": next(
            (
                item.semantic_role
                for item in elements
                if item.semantic_role in {"title-slide", "section", "content"}
            ),
            None,
        ),
    }
