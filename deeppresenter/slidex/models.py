"""Versioned domain models for Slidex artifacts and inspection results."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "2.0"


class SlidexModel(BaseModel):
    """Strict base model shared by persisted Slidex schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DefectClass(StrEnum):
    """Version 1 defect taxonomy; new values require a taxonomy version bump."""

    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"
    G6 = "G6"
    G7 = "G7"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"


class RepairOperation(StrEnum):
    """Supported source-level repair operations."""

    MOVE_ELEMENT = "move_element"
    RESIZE_CONTAINER = "resize_container"
    REDUCE_TEXT = "reduce_text"
    CHANGE_FONT_SIZE = "change_font_size"
    REPLACE_COLOR = "replace_color"
    RENAME_TERM = "rename_term"
    POLICY_EDIT = "policy_edit"


class RepairExecutionStatus(StrEnum):
    APPLIED = "applied"
    SUGGESTED = "suggested"
    REJECTED = "rejected"


class PolicyViolationCode(StrEnum):
    HIDDEN_CONTENT = "hidden_content"
    OFFSCREEN_CONTENT = "offscreen_content"
    ZERO_SIZE_CONTENT = "zero_size_content"
    TEXT_AS_IMAGE = "text_as_image"
    TINY_TEXT = "tiny_text"
    EXEMPTION_ABUSE = "exemption_abuse"
    REQUIRED_CONTENT_REMOVED = "required_content_removed"
    PATH_ESCAPE = "path_escape"
    INVALID_ACTION = "invalid_action"


class InspectionStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    DEFER = "defer"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class EvidenceSource(StrEnum):
    DECLARED_IR = "declared_ir"
    COMPUTED_IR = "computed_ir"
    RENDER = "render"
    CLEAN_REFERENCE = "clean_reference"
    DECK_TEXT = "deck_text"


class BoundingBox(SlidexModel):
    """A box in a page-relative coordinate system."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    unit: Literal["px", "pt", "in", "mm"] = "px"
    coordinate_system: Literal["top_left"] = "top_left"

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox:
        if (
            self.x + self.width > self.page_width
            or self.y + self.height > self.page_height
        ):
            raise ValueError("bounding box exceeds page bounds")
        return self


class SlideElement(SlidexModel):
    """A stable, recursively nested slide element."""

    element_id: str = Field(min_length=1, pattern=r"^\S+$")
    tag: str = Field(min_length=1)
    element_type: str | None = None
    semantic_role: str | None = None
    text: str = ""
    bbox: BoundingBox | None = None
    style: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None
    children: list[SlideElement] = Field(default_factory=list)


class DeclaredSlideIR(SlidexModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    slide_id: str = Field(min_length=1)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    unit: Literal["px", "pt", "in", "mm"] = "px"
    elements: list[SlideElement] = Field(default_factory=list)
    containers: list[str] = Field(default_factory=list)
    theme_tokens: dict[str, Any] = Field(default_factory=dict)
    expected_roles: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_elements(self) -> DeclaredSlideIR:
        _validate_element_tree(self.elements)
        return self


class ObservedBoundingBox(SlidexModel):
    """A browser-observed box which may extend beyond the slide viewport."""

    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    unit: Literal["px"] = "px"
    coordinate_system: Literal["top_left"] = "top_left"


class ComputedSlideElement(SlideElement):
    bbox: ObservedBoundingBox | None = None
    computed_style: dict[str, str] = Field(default_factory=dict)
    client_width: float = Field(ge=0, default=0)
    client_height: float = Field(ge=0, default=0)
    scroll_width: float = Field(ge=0, default=0)
    scroll_height: float = Field(ge=0, default=0)
    text_bboxes: list[ObservedBoundingBox] = Field(default_factory=list)
    visible_bbox: ObservedBoundingBox | None = None
    visible: bool = True
    partially_outside_page: bool = False
    clipped: bool = False
    stacking_order: int = 0
    font_fallback: list[str] = Field(default_factory=list)
    image: dict[str, Any] = Field(default_factory=dict)


class ComputedSlideIR(SlidexModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    slide_id: str = Field(min_length=1)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    elements: list[ComputedSlideElement] = Field(default_factory=list)
    browser: str
    browser_version: str
    warnings: list[str] = Field(default_factory=list)
    console_errors: list[str] = Field(default_factory=list)
    page_errors: list[str] = Field(default_factory=list)
    resource_errors: list[str] = Field(default_factory=list)
    render_ready: bool = True

    @model_validator(mode="after")
    def validate_elements(self) -> ComputedSlideIR:
        _validate_element_tree(self.elements)
        return self


class RendererInfo(SlidexModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class RenderArtifact(SlidexModel):
    kind: Literal["html", "pdf", "pptx_rerender"]
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    renderer: RendererInfo


class FinalArtifactStatus(StrEnum):
    """Lifecycle state of a generated presentation deliverable."""

    DRAFT_HTML_VALID = "draft_html_valid"
    PPTX_EXPORTED = "pptx_exported"
    PPTX_RENDER_VALIDATED = "pptx_render_validated"
    INVALID_ARTIFACT = "invalid_artifact"
    CAPABILITY_ERROR = "capability_error"


class ExportCommandRecord(SlidexModel):
    """Replay metadata for one external export command."""

    executable: str
    arguments: list[str] = Field(default_factory=list)
    version: str
    return_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = Field(ge=0)


class FidelityPageResult(SlidexModel):
    """Multi-signal comparison between one HTML render and PPTX re-render."""

    slide_id: str
    html_render_uri: str
    pptx_render_uri: str
    html_size: tuple[int, int]
    pptx_size: tuple[int, int]
    pixel_difference: float = Field(ge=0, le=1)
    perceptual_similarity: float = Field(ge=0, le=1)
    text_presence: float = Field(ge=0, le=1)
    missing_text: list[str] = Field(default_factory=list)
    missing_images: int = Field(ge=0, default=0)
    font_substitutions: list[str] = Field(default_factory=list)
    position_drift: dict[str, float] = Field(default_factory=dict)
    final_render_findings: list[str] = Field(default_factory=list)
    passed: bool


class RenderFidelityReport(SlidexModel):
    """Deck-level export fidelity gate; no single similarity decides validity."""

    page_results: list[FidelityPageResult] = Field(default_factory=list)
    expected_page_count: int = Field(ge=0)
    actual_page_count: int = Field(ge=0)
    page_count_matches: bool
    renderer: RendererInfo
    export_fidelity_failure: bool
    failure_reasons: list[str] = Field(default_factory=list)


class ExportManifest(SlidexModel):
    """Auditable final-deliverable manifest for export and re-render."""

    schema_version: Literal["2.0"] = SCHEMA_VERSION
    export_id: str = Field(min_length=1)
    status: FinalArtifactStatus
    source_uris: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    critic_report_uris: list[str] = Field(default_factory=list)
    strict_validation: bool = True
    soft_mode_explicit: bool = False
    ignored_warnings: list[str] = Field(default_factory=list)
    commands: list[ExportCommandRecord] = Field(default_factory=list)
    output_files: dict[str, ArtifactReference] = Field(default_factory=dict)
    fidelity_report: RenderFidelityReport | None = None
    hard_penalty: bool = False
    failure_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GroundingStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


class GroundingFinding(SlidexModel):
    claim: str = Field(min_length=1)
    slide_id: str = Field(min_length=1)
    status: GroundingStatus
    evidence: list[str] = Field(default_factory=list)
    source_uris: list[str] = Field(default_factory=list)


class GroundingReport(SlidexModel):
    findings: list[GroundingFinding] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)
    supported_rate: float = Field(ge=0, le=1)
    contradiction_rate: float = Field(ge=0, le=1)
    unsupported_rate: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)


class MutationFidelityResult(SlidexModel):
    """Observable survival of one mutation after the final renderer."""

    mutation_id: str
    defect_class: DefectClass
    clean_render_uri: str
    defective_render_uri: str
    renderer: RendererInfo
    pixel_difference: float = Field(ge=0, le=1)
    perceptual_similarity: float = Field(ge=0, le=1)
    zero_signal: bool
    include_in_training: bool
    label_source: Literal["final_render"] = "final_render"


class Provenance(SlidexModel):
    parent_artifact_id: str | None = None
    creation_action: str = Field(min_length=1)
    model: str | None = None
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)


class ArtifactTrust(StrEnum):
    TRUSTED_SOURCE = "trusted_source"
    PARTIAL_NATIVE = "partial_native"
    RECOVERED = "recovered"
    IMAGE_ONLY = "image_only"


class SlideArtifact(SlidexModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    source_uri: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    declared_ir: DeclaredSlideIR
    computed_ir: ComputedSlideIR | None = None
    renders: list[RenderArtifact] = Field(default_factory=list)
    assets: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance
    trust: ArtifactTrust = ArtifactTrust.TRUSTED_SOURCE
    missing_bookkeeping: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_slide_identity(self) -> SlideArtifact:
        if self.computed_ir and self.computed_ir.slide_id != self.declared_ir.slide_id:
            raise ValueError("declared and computed IR slide IDs differ")
        return self


class DefectTransition(SlidexModel):
    """Before/after state for one defect class across a repair."""

    defect_class: DefectClass
    before: InspectionStatus
    after: InspectionStatus
    before_severity: float = Field(default=0, ge=0, le=1)
    after_severity: float = Field(default=0, ge=0, le=1)
    transition: Literal["improved", "unchanged", "worsened"]


class RepairAction(SlidexModel):
    """Auditable proposal or execution of one localized source edit."""

    action_id: str = Field(min_length=1)
    operation: RepairOperation
    target_ids: list[str] = Field(min_length=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    source_inspection_ids: list[str] = Field(min_length=1)
    before_artifact_id: str = Field(min_length=1)
    after_artifact_id: str | None = None
    status: RepairExecutionStatus = RepairExecutionStatus.SUGGESTED
    policy_reason: str | None = None
    defect_delta: list[DefectTransition] = Field(default_factory=list)


class PolicyViolation(SlidexModel):
    """A hard anti-reward-hacking finding."""

    code: PolicyViolationCode
    slide_id: str
    element_ids: list[str] = Field(default_factory=list)
    detail: str = Field(min_length=1)
    severity: float = Field(default=1, ge=0, le=1)


class RepairHint(SlidexModel):
    """Machine-readable repair operation emitted by deterministic inspectors."""

    action: str = Field(min_length=1)
    targets: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None


class Evidence(SlidexModel):
    source: EvidenceSource
    detail: str = Field(min_length=1)
    element_ids: list[str] = Field(default_factory=list)
    artifact_uri: str | None = None


class InspectionResult(SlidexModel):
    defect_class: DefectClass
    status: InspectionStatus
    severity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    repair_hint: RepairHint | str | None = None
    inspector_name: str | None = None
    input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    latency_ms: float = Field(ge=0, default=0)
    cost: float = Field(ge=0, default=0)
    inspector_version: str = Field(min_length=1)
    raw_output_uri: str | None = None


class RouteRecord(SlidexModel):
    defect_class: DefectClass
    stages: list[str] = Field(default_factory=list)
    reason: str
    missing_evidence: list[str] = Field(default_factory=list)
    capability_limit: str | None = None


class InspectionReport(SlidexModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    artifact_id: str
    slide_id: str
    results: list[InspectionResult] = Field(default_factory=list)
    summary: dict[str, int | float | str] = Field(default_factory=dict)
    router_version: str
    taxonomy_version: str
    router_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    routes: list[RouteRecord] = Field(default_factory=list)
    conflicts: list[DefectClass] = Field(default_factory=list)
    resolved_status: dict[DefectClass, InspectionStatus] = Field(default_factory=dict)
    capability_limits: list[str] = Field(default_factory=list)
    report_uri: str | None = None
    policy_violations: list[PolicyViolation] = Field(default_factory=list)


class DeckInspectionReport(SlidexModel):
    """Final deck gate with page-local and cross-page findings."""

    deck_id: str = Field(min_length=1)
    page_reports: dict[str, InspectionReport] = Field(default_factory=dict)
    deck_results: list[InspectionResult] = Field(default_factory=list)
    affected_slide_ids: list[str] = Field(default_factory=list)
    policy_violations: list[PolicyViolation] = Field(default_factory=list)
    hard_failures: int = Field(ge=0)
    export_allowed: bool
    override_reason: str | None = None
    taxonomy_version: str
    router_version: str


class RewardBreakdown(SlidexModel):
    """Legacy scalar reward view retained for API compatibility."""

    schema_version: Literal["2.0"] = SCHEMA_VERSION
    hard_constraints: dict[str, bool] = Field(default_factory=dict)
    soft_scores: dict[str, float] = Field(default_factory=dict)
    cost_penalty: float = Field(ge=0, default=0)
    aggregate: float
    gating_reason: str | None = None
    reward_version: str


class ArtifactReference(SlidexModel):
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str | None = None
    size_bytes: int = Field(ge=0)


class PolicyCallRecord(SlidexModel):
    """Auditable outbound policy call attached to one environment step."""

    endpoint_identifier: str
    provider: Literal["openai", "litellm"]
    model: str
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    finish_reasons: list[str | None] = Field(default_factory=list)
    reasoning: list[str | None] = Field(default_factory=list)
    tool_calls: list[list[dict[str, Any]]] = Field(default_factory=list)
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt: int = Field(ge=1)


class TrajectoryStep(SlidexModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = Field(ge=0)
    action: dict[str, Any]
    observation: dict[str, Any] = Field(default_factory=dict)
    policy_call: PolicyCallRecord | None = None
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    reward: RewardBreakdown | None = None
    strict_validation: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EpisodeManifest(SlidexModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    workspace_uri: str
    status: Literal["active", "completed", "failed"] = "active"
    artifact_ids: list[str] = Field(default_factory=list)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    versions: dict[str, str] = Field(default_factory=dict)


class ArtifactManifest(SlidexModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    artifact_id: str
    files: dict[str, ArtifactReference] = Field(default_factory=dict)
    provenance: Provenance
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _validate_element_tree(elements: list[SlideElement]) -> None:
    seen: set[str] = set()

    def visit(element: SlideElement, expected_parent: str | None) -> None:
        if element.element_id in seen:
            raise ValueError(f"duplicate element ID: {element.element_id}")
        if element.parent_id != expected_parent:
            raise ValueError(
                f"invalid parent for {element.element_id}: expected {expected_parent!r}"
            )
        seen.add(element.element_id)
        for child in element.children:
            visit(child, element.element_id)

    for root in elements:
        visit(root, None)


class NormalizedBoundingBox(SlidexModel):
    """A localization box normalized to the [0, 1] slide coordinate space."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> NormalizedBoundingBox:
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("normalized bounding box exceeds page bounds")
        return self


class InspectionContext(SlidexModel):
    """Explicit evidence available to neural inspectors."""

    artifact: SlideArtifact
    render_path: str | None = None
    reference_artifact_id: str | None = None
    reference_artifact: SlideArtifact | None = None
    reference_render_path: str | None = None
    deck_outline: list[str] = Field(default_factory=list)
    slide_summaries: dict[str, str] = Field(default_factory=dict)
    approved_outline: list[str] = Field(default_factory=list)
    task: str | None = None


class AtomicVerdict(SlidexModel):
    """Strict response schema shared by all single-defect model calls."""

    verdict: Literal["pass", "fail", "defer"]
    severity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(min_length=1)
    element_ids: list[str] = Field(default_factory=list)
    bboxes: list[NormalizedBoundingBox] = Field(default_factory=list)
    repair_suggestion: str | None = None
    defer_reason: str | None = None

    @model_validator(mode="after")
    def validate_defer_reason(self) -> AtomicVerdict:
        if self.verdict == "defer" and not self.defer_reason:
            raise ValueError("defer_reason is required for a deferred verdict")
        return self


class PairwiseVerdict(SlidexModel):
    verdict: Literal["left", "right", "tie", "defer"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(min_length=1)
    defer_reason: str | None = None


class NeuralCallRecord(SlidexModel):
    """Replay metadata for one stateless OpenAI-compatible request."""

    defect_class: DefectClass
    endpoint_identifier: str
    model: str
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    raw_response: str
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition: str | None = None


class AttributionLabel(StrEnum):
    IMAGE_SUFFICIENT = "image_sufficient"
    STRUCTURE_RESCUED = "structure_rescued"
    FORMAT_SUPPRESSED = "format_suppressed"
    REFERENCE_ASSISTED = "reference_assisted"
    UNRESOLVED = "unresolved"


class FailureAttribution(SlidexModel):
    defect_class: DefectClass
    label: AttributionLabel
    conditions: dict[str, AtomicVerdict] = Field(default_factory=dict)
    whole_rubric: list[AtomicVerdict] = Field(default_factory=list)
    records: list[NeuralCallRecord] = Field(default_factory=list)
    explanation: str
