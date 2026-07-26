"""Intrinsic critic baselines and frozen failure-attribution controls."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import Arm, CaseResult, EvaluationCase, EvaluationRun, Outcome


class CriticObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: Outcome
    defects: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    element_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    raw_output: Any = None
    tokens: int = 0
    cost: float = 0
    latency_seconds: float = 0
    capability_downgrade: str | None = None


CriticCall = Callable[[EvaluationCase, str, str], Awaitable[CriticObservation]]


_PROMPTS = {
    Arm.C0: "Inspect this slide using the complete presentation-quality rubric. Return structured verdicts.",
    Arm.C0_X10: "Inspect this slide using the complete presentation-quality rubric. Return structured verdicts.",
    Arm.C0_PLUS: "Inspect this slide using the complete rubric, paying special attention to {defect}.",
    Arm.ATOMIC: "Decide only whether defect {defect} exists. Cite visible evidence and localize it.",
    Arm.VLM_ONLY: "Decide only whether defect {defect} exists from rendered pixels. Cite and localize evidence.",
    Arm.FROZEN_HYBRID: "Inspect only defect {defect} using the frozen symbolic-neural-reference route and cite the selected evidence.",
    Arm.REFERENCE_DISABLED: "Decide only whether defect {defect} exists without a clean reference.",
    Arm.MISMATCHED_ROUTER: "Inspect {defect} using deliberately mismatched evidence as a negative control.",
}


def prompt_hash(arm: Arm, defect: str = "") -> str:
    prompt = _PROMPTS.get(arm, "symbolic frozen predicates").format(defect=defect)
    return hashlib.sha256(prompt.encode()).hexdigest()


async def run_intrinsic_case(
    case: EvaluationCase, run: EvaluationRun, call: CriticCall
) -> CaseResult:
    """Run one frozen control while preserving abstentions and total budget."""
    defect = case.labels[0].defect_class if case.labels else "any"
    prompt = _PROMPTS.get(run.arm, "symbolic frozen predicates").format(defect=defect)
    repeats = 10 if run.arm is Arm.C0_X10 else 1
    observations = [
        await call(
            case,
            prompt,
            ("BA" if (int(case.case_id[:8], 16) + index) % 2 else "AB"),
        )
        for index in range(repeats)
    ]
    terminal = [
        item for item in observations if item.outcome in {Outcome.PASS, Outcome.FAIL}
    ]
    if not terminal:
        selected = observations[0]
    elif repeats == 1:
        selected = terminal[0]
    else:
        failures = sum(item.outcome is Outcome.FAIL for item in terminal)
        selected = terminal[0].model_copy(
            update={
                "outcome": Outcome.FAIL
                if failures > len(terminal) / 2
                else Outcome.PASS
            }
        )
    return CaseResult(
        case_id=case.case_id,
        outcome=selected.outcome,
        predicted_defects=selected.defects,
        confidence=selected.confidence,
        element_id=selected.element_id,
        bbox=selected.bbox,
        raw_output=[item.raw_output for item in observations],
        tokens=sum(item.tokens for item in observations),
        cost=sum(item.cost for item in observations),
        latency_seconds=sum(item.latency_seconds for item in observations),
        error=None,
        capability_downgrade=selected.capability_downgrade,
    )
