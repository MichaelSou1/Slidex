"""Deterministic claim-to-source grounding checks for generated presentations."""

from __future__ import annotations

import re
from collections.abc import Sequence

from deeppresenter.slidex.models import (
    GroundingFinding,
    GroundingReport,
    GroundingStatus,
)


class GroundingEvaluator:
    """Verify citation-like and numeric claims against collected source text.

    This evaluator intentionally stays conservative: it marks directly matched
    evidence as supported and otherwise reports insufficient evidence. It never
    turns absence of evidence into a fabricated contradiction.
    """

    def evaluate(
        self,
        slide_text: Sequence[Sequence[str]],
        sources: dict[str, str],
    ) -> GroundingReport:
        corpus = {uri: _normalize(text) for uri, text in sources.items() if text.strip()}
        findings: list[GroundingFinding] = []
        for index, page in enumerate(slide_text, start=1):
            for claim in _claims(page):
                tokens = _significant_tokens(claim)
                numbers = set(_numbers(claim))
                matches = [
                    uri
                    for uri, text in corpus.items()
                    if _supported(tokens, numbers, text)
                ]
                contradictions = [
                    uri
                    for uri, text in corpus.items()
                    if _contradicted(tokens, numbers, text)
                ]
                status = (
                    GroundingStatus.SUPPORTED
                    if matches
                    else GroundingStatus.CONTRADICTED
                    if contradictions
                    else GroundingStatus.NOT_ENOUGH_EVIDENCE
                )
                findings.append(
                    GroundingFinding(
                        claim=claim,
                        slide_id=f"slide_{index:02d}",
                        status=status,
                        evidence=[claim] if matches or contradictions else [],
                        source_uris=matches or contradictions,
                    )
                )
        total = len(findings)
        supported = sum(item.status == GroundingStatus.SUPPORTED for item in findings)
        contradicted = sum(item.status == GroundingStatus.CONTRADICTED for item in findings)
        unsupported = total - supported - contradicted
        return GroundingReport(
            findings=findings,
            source_count=len(corpus),
            supported_rate=supported / total if total else 0,
            contradiction_rate=contradicted / total if total else 0,
            unsupported_rate=unsupported / total if total else 0,
            coverage=(supported + contradicted) / total if total else 0,
        )


def _claims(page: Sequence[str]) -> list[str]:
    claims: list[str] = []
    for block in page:
        for item in re.split(r"[\n。！？!?;；]+", block):
            value = item.strip(" -•\t")
            if len(value) >= 8 and (_numbers(value) or len(_significant_tokens(value)) >= 4):
                claims.append(value)
    return claims


def _supported(tokens: set[str], numbers: set[str], source: str) -> bool:
    if numbers and not numbers <= set(_numbers(source)):
        return False
    source_tokens = _significant_tokens(source)
    if not tokens:
        return bool(numbers)
    return len(tokens & source_tokens) / len(tokens) >= 0.7


def _contradicted(tokens: set[str], numbers: set[str], source: str) -> bool:
    if not numbers:
        return False
    source_tokens = _significant_tokens(source)
    overlap = len(tokens & source_tokens) / len(tokens) if tokens else 0
    return overlap >= 0.7 and not numbers <= set(_numbers(source))


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[\w\u4e00-\u9fff.%+-]+", value.casefold()))


def _significant_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", value.casefold())
        if len(token) > 1 and token not in {"the", "and", "with", "from", "that", "this"}
    }


def _numbers(value: str) -> list[str]:
    return re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?%?", value)
