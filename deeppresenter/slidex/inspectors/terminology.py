"""Explainable deterministic terminology-variant detection (S3)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from deeppresenter.slidex.inspectors.base import result
from deeppresenter.slidex.models import (
    DefectClass,
    Evidence,
    EvidenceSource,
    InspectionResult,
    InspectionStatus,
    RepairHint,
    SlideArtifact,
    SlideElement,
)


def _flatten(elements: list[SlideElement]) -> list[SlideElement]:
    return [item for root in elements for item in [root, *_flatten(root.children)]]


def normalize_term(term: str) -> str:
    """Normalize case, width, hyphens, plural suffixes, and whitespace."""
    import unicodedata

    value = unicodedata.normalize("NFKC", term).casefold()
    value = re.sub(r"[‐‑‒–—−_]", "-", value)
    value = re.sub(r"\s+", " ", value).strip()
    words = [
        word[:-1]
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss")
        else word
        for word in value.split()
    ]
    return " ".join(words)


@dataclass
class TerminologyInspector:
    """S3 candidate conflicts; aliases and glossary remain authoritative."""

    glossary: dict[str, str] = field(default_factory=dict)
    accepted_aliases: dict[str, set[str]] = field(default_factory=dict)
    similarity_threshold: float = 0.88
    name: str = "terminology.consistency"
    version: str = "1.0"
    defect_class: DefectClass = DefectClass.S3

    def inspect(self, artifact: SlideArtifact) -> list[InspectionResult]:
        return self.inspect_deck([artifact])

    def inspect_deck(self, artifacts: list[SlideArtifact]) -> list[InspectionResult]:
        started = time.perf_counter()
        if not artifacts:
            return []
        occurrences: dict[str, list[tuple[SlideArtifact, SlideElement, str]]] = {}
        for artifact in artifacts:
            for element in _flatten(artifact.declared_ir.elements):
                for term in re.findall(
                    r"\b(?:[A-Z][A-Za-z0-9]*(?:[- ][A-Za-z0-9]+)+|[A-Za-z]+-[A-Za-z]+|[A-Z]{2,})\b",
                    element.text,
                ):
                    occurrences.setdefault(term, []).append((artifact, element, term))
        terms = sorted(occurrences)
        failures: list[InspectionResult] = []
        used: set[frozenset[str]] = set()
        for index, left in enumerate(terms):
            for right in terms[index + 1 :]:
                left_norm, right_norm = normalize_term(left), normalize_term(right)
                canonical = self.glossary.get(left_norm) or self.glossary.get(
                    right_norm
                )
                aliases = self.accepted_aliases.get(canonical or "", set())
                if {left_norm, right_norm} <= {
                    normalize_term(alias) for alias in aliases
                }:
                    continue
                similarity = SequenceMatcher(None, left_norm, right_norm).ratio()
                if left_norm != right_norm and similarity < self.similarity_threshold:
                    continue
                pair = frozenset({left, right})
                if pair in used or left == right:
                    continue
                used.add(pair)
                all_occurrences = occurrences[left] + occurrences[right]
                ids = [element.element_id for _, element, _ in all_occurrences]
                suggestion = canonical or max(
                    (left, right),
                    key=lambda value: (len(occurrences[value]), -len(value)),
                )
                evidence = [
                    {
                        "term": term,
                        "slide_id": artifact.declared_ir.slide_id,
                        "element_id": element.element_id,
                        "context": element.text,
                    }
                    for artifact, element, term in all_occurrences
                ]
                failures.append(
                    result(
                        self,
                        artifacts[0],
                        InspectionStatus.FAIL,
                        severity=min(
                            1,
                            max(0.2, 1 - similarity)
                            if left_norm != right_norm
                            else 0.35,
                        ),
                        confidence=0.8,
                        evidence=[
                            Evidence(
                                source=EvidenceSource.DECK_TEXT,
                                detail=f"candidate variants={left!r}/{right!r}; normalized={left_norm!r}/{right_norm!r}; similarity={similarity:.3f}; occurrences={evidence}",
                                element_ids=ids,
                            )
                        ],
                        element_ids=ids,
                        repair_hint=RepairHint(
                            action="normalize_terminology",
                            targets=ids,
                            parameters={
                                "canonical": suggestion,
                                "variants": [left, right],
                                "occurrences": evidence,
                            },
                        ),
                        started_at=started,
                    )
                )
        return failures or [
            result(self, artifacts[0], InspectionStatus.PASS, started_at=started)
        ]
