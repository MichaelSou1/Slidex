"""Resumable paired-arm execution with immutable per-case records."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from .io import content_hash, write_immutable
from .models import (
    Arm,
    BenchmarkManifest,
    CaseResult,
    EvaluationCase,
    EvaluationResult,
    EvaluationRun,
    Outcome,
)

CaseExecutor = Callable[[EvaluationCase, EvaluationRun], Awaitable[CaseResult]]
_FAILED = {Outcome.ERROR, Outcome.TIMEOUT, Outcome.MISSING}


class EvaluationRunner:
    def __init__(self, output_dir: Path, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.output_dir = output_dir
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run(
        self,
        manifest: BenchmarkManifest,
        run: EvaluationRun,
        executor: CaseExecutor,
        *,
        rerun_failed: bool = False,
    ) -> EvaluationResult:
        run_dir = self.output_dir / run.run_id
        write_immutable(run_dir / "run.json", run)

        async def execute(case: EvaluationCase) -> CaseResult:
            path = run_dir / "cases" / f"{case.case_id}.json"
            if path.exists():
                existing = CaseResult.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if not rerun_failed or existing.outcome not in _FAILED:
                    return existing
                path.unlink()
            async with self.semaphore:
                try:
                    result = await executor(case, run)
                except TimeoutError as error:
                    result = CaseResult(
                        case_id=case.case_id, outcome=Outcome.TIMEOUT, error=str(error)
                    )
                except Exception as error:
                    result = CaseResult(
                        case_id=case.case_id,
                        outcome=Outcome.ERROR,
                        error=f"{type(error).__name__}: {error}",
                    )
            if result.case_id != case.case_id:
                raise ValueError("executor returned a result for another case")
            write_immutable(path, result)
            return result

        results = await asyncio.gather(*(execute(case) for case in manifest.cases))
        final = EvaluationResult(
            run=run, results=sorted(results, key=lambda item: item.case_id)
        )
        final.immutable_hash = content_hash(
            final.model_dump(exclude={"immutable_hash"}, mode="json")
        )
        write_immutable(run_dir / "result.json", final)
        return final


def validate_paired_runs(runs: list[EvaluationRun]) -> None:
    """Ensure three E2E arms differ only by critic mechanism."""
    grouped: dict[tuple[str, int], list[EvaluationRun]] = {}
    for run in runs:
        grouped.setdefault((run.metadata.get("task_id", ""), run.seed), []).append(run)
    expected = {Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID}
    for key, group in grouped.items():
        if {run.arm for run in group} != expected:
            raise ValueError(f"incomplete paired arms: {key}")
        if len({run.initial_artifact_id for run in group}) != 1:
            raise ValueError(f"arms do not share initial artifact: {key}")
        if len({(run.max_repairs, run.model_budget) for run in group}) != 1:
            raise ValueError(f"arms do not share repair/model budget: {key}")
