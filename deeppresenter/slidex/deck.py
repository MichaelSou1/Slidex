"""Deck-level inspection, incremental re-checking, and final export gate."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urlparse

from deeppresenter.slidex.critic import HybridCritic
from deeppresenter.slidex.inspectors.style import TypographyInspector
from deeppresenter.slidex.inspectors.terminology import TerminologyInspector
from deeppresenter.slidex.models import (
    DeckInspectionReport,
    DefectClass,
    InspectionContext,
    InspectionReport,
    InspectionResult,
    InspectionStatus,
    SlideArtifact,
)
from deeppresenter.slidex.repair import detect_policy_violations
from deeppresenter.utils.config import SlidexConfig

# S2/S5 evidence is collected once per page (each page's InspectionContext
# carries the same deck-wide task/slide_summaries), but they judge the deck
# as a whole rather than any single page. `_semantic_deck_results` folds the
# per-page duplicates down to one representative result per class; downstream
# hard-failure/affected-slide accounting must treat page_reports entries for
# these classes as raw evidence only, not as independently countable defects.
_DECK_LEVEL_CLASSES = {DefectClass.S2, DefectClass.S5}

# S5 ("a visibly missing step or section") is judged by an LLM on coarse
# deck-level evidence (task text + slide summaries) and has shown a
# reproducible hallucination mode: judging a page count that already
# satisfies the task's own requested range as "insufficient depth". Keep
# collecting and reporting S5 findings for visibility, but do not let them
# alone block export the way deterministic/geometry hard findings do.
_SOFT_DEFECT_CLASSES = {DefectClass.S5}


class DeckInspector:
    """Coordinate page and cross-page checks without indiscriminate regeneration."""

    def __init__(self, critic: HybridCritic, config: SlidexConfig) -> None:
        self.critic = critic
        self.config = config

    async def inspect(
        self,
        artifacts: list[SlideArtifact],
        *,
        previous: DeckInspectionReport | None = None,
        changed_slide_ids: Iterable[str] | None = None,
        approved_outline: list[str] | None = None,
        task: str | None = None,
        override_reason: str | None = None,
    ) -> DeckInspectionReport:
        if not artifacts:
            raise ValueError("deck inspection requires at least one slide")
        by_id = {item.declared_ir.slide_id: item for item in artifacts}
        changed = set(changed_slide_ids or by_id)
        if not changed <= by_id.keys():
            raise ValueError("changed_slide_ids contains an unknown slide")
        page_reports = dict(previous.page_reports) if previous else {}
        summaries = {
            slide_id: " ".join(
                element.text for element in artifact.declared_ir.elements
            )
            for slide_id, artifact in by_id.items()
        }
        for slide_id in by_id:
            if slide_id not in changed:
                continue
            artifact = by_id[slide_id]
            page_reports[slide_id] = await self.critic.inspect(
                InspectionContext(
                    artifact=artifact,
                    render_path=_local_render_path(artifact),
                    slide_summaries=summaries,
                    approved_outline=approved_outline or [],
                    deck_outline=approved_outline or [],
                    task=task,
                )
            )

        deck_results = self._deterministic_deck_results(artifacts)
        deck_results.extend(
            self._semantic_deck_results(page_reports, _DECK_LEVEL_CLASSES)
        )
        violations = [
            violation
            for artifact in artifacts
            for violation in detect_policy_violations(artifact)
        ]
        # Deck-level defect classes (S2/S5) are routed per page for evidence
        # collection, but a single deck-level defect must only count once.
        # `_semantic_deck_results` already picks one representative result per
        # class into `deck_results`; page_reports still carry the raw
        # per-page duplicates for inspection/debugging, so they must be
        # excluded here to avoid counting the same deck-level defect once per
        # slide.
        hard_failures = (
            sum(
                result.status in {InspectionStatus.FAIL, InspectionStatus.ERROR}
                and result.severity > 0
                for report in page_reports.values()
                for result in report.results
                if result.defect_class not in _DECK_LEVEL_CLASSES
            )
            + sum(
                result.status in {InspectionStatus.FAIL, InspectionStatus.ERROR}
                and result.severity > 0
                for result in deck_results
                if result.defect_class not in _SOFT_DEFECT_CLASSES
            )
            + len(violations)
        )
        affected = sorted(
            {
                slide_id
                for slide_id, report in page_reports.items()
                if any(
                    _is_hard(item)
                    for item in report.results
                    if item.defect_class not in _DECK_LEVEL_CLASSES
                )
            }
            | {
                artifact.declared_ir.slide_id
                for artifact in artifacts
                if any(
                    element_id in _element_ids(artifact)
                    for result in deck_results
                    for element_id in result.element_ids
                )
            }
            | (
                set(by_id)
                if any(
                    _is_hard(result) and not result.element_ids
                    for result in deck_results
                    if result.defect_class not in _SOFT_DEFECT_CLASSES
                )
                else set()
            )
            | {violation.slide_id for violation in violations}
        )
        return DeckInspectionReport(
            deck_id=_deck_id(artifacts),
            page_reports=page_reports,
            deck_results=deck_results,
            affected_slide_ids=affected,
            policy_violations=violations,
            hard_failures=hard_failures,
            export_allowed=hard_failures == 0 or bool(override_reason),
            override_reason=override_reason,
            taxonomy_version=self.config.taxonomy_version,
            router_version=self.config.router_version,
        )

    def _deterministic_deck_results(
        self, artifacts: list[SlideArtifact]
    ) -> list[InspectionResult]:
        results: list[InspectionResult] = []
        results.extend(TerminologyInspector().inspect_deck(artifacts))
        results.extend(
            TypographyInspector(self.config.typography_tolerance_px).inspect_deck(
                artifacts
            )
        )
        return results

    @staticmethod
    def _semantic_deck_results(
        reports: dict[str, InspectionReport], classes: set[DefectClass]
    ) -> list[InspectionResult]:
        selected: dict[DefectClass, InspectionResult] = {}
        for report in reports.values():
            for result in report.results:
                if result.defect_class not in classes:
                    continue
                current = selected.get(result.defect_class)
                if current is None or _status_priority(
                    result.status
                ) > _status_priority(current.status):
                    selected[result.defect_class] = result
        return list(selected.values())


def enforce_export_gate(report: DeckInspectionReport) -> None:
    """Block finalization unless all hard findings pass or an override is audited."""
    if not report.export_allowed:
        slides = ", ".join(report.affected_slide_ids) or "unknown"
        raise RuntimeError(
            f"export blocked by {report.hard_failures} hard findings; affected slides: {slides}"
        )


def _deck_id(artifacts: list[SlideArtifact]) -> str:
    value = "\n".join(
        f"{item.declared_ir.slide_id}:{item.artifact_id}" for item in artifacts
    )
    return "deck-" + hashlib.sha256(value.encode()).hexdigest()[:16]


def _element_ids(artifact: SlideArtifact) -> set[str]:
    def walk(elements):
        return {
            item.element_id
            for root in elements
            for item in [root, *walk_list(root.children)]
        }

    def walk_list(elements):
        return [item for root in elements for item in [root, *walk_list(root.children)]]

    return walk(artifact.declared_ir.elements)


def _is_hard(result: InspectionResult) -> bool:
    return result.status == InspectionStatus.ERROR or (
        result.status == InspectionStatus.FAIL and result.severity > 0
    )


def _status_priority(status: InspectionStatus) -> int:
    return {
        InspectionStatus.ERROR: 4,
        InspectionStatus.FAIL: 3,
        InspectionStatus.DEFER: 2,
        InspectionStatus.PASS: 1,
        InspectionStatus.NOT_APPLICABLE: 0,
    }[status]


def _local_render_path(artifact: SlideArtifact) -> str | None:
    if not artifact.renders:
        return None
    uri = artifact.renders[-1].uri
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return str(Path(unquote(parsed.path)))
    path = Path(uri)
    return str(path) if path.is_file() else None
