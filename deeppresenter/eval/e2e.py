"""Paired three-arm E2E orchestration from one immutable first artifact."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from .models import CaseResult, EvaluationCase, EvaluationRun
from .runner import validate_paired_runs

Generate = Callable[[EvaluationCase, int], Awaitable[str]]
Repair = Callable[[EvaluationCase, EvaluationRun], Awaitable[CaseResult]]


async def run_paired_task(
    case: EvaluationCase,
    seed: int,
    runs: list[EvaluationRun],
    generate: Generate,
    repair: Repair,
) -> list[CaseResult]:
    """Generate exactly once, then fork no-critic/generic/hybrid with equal budgets."""
    initial_artifact = await generate(case, seed)
    for run in runs:
        run.initial_artifact_id = initial_artifact
        run.metadata["task_id"] = case.case_id
    validate_paired_runs(runs)
    return [await repair(case, run) for run in runs]


def validate_medium_budget(
    manifest_cases: list[EvaluationCase], seeds: tuple[int, ...]
) -> None:
    sealed = [case for case in manifest_cases if case.split.value == "sealed_test"]
    if len(sealed) != 100 or len(seeds) != 3:
        raise ValueError(
            "confirmatory E2E requires exactly 100 sealed tasks and 3 fixed seeds"
        )
    categories: dict[str, int] = {}
    for case in sealed:
        category = str(case.metadata.get("task_type", ""))
        categories[category] = categories.get(category, 0) + 1
    expected = {"academic": 25, "business": 25, "product": 25, "teaching": 25}
    if categories != expected:
        raise ValueError(f"sealed task strata must equal {expected}")
