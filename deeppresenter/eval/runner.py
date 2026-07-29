"""Resumable paired-arm execution with immutable per-case records."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from .integrity import validate_lineage
from .io import content_hash, file_hash, write_immutable
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
        result_path = run_dir / "result.json"
        if rerun_failed and result_path.exists():
            # Resuming previously-failed cases legitimately changes result.json;
            # only the per-case records under cases/ need to stay append-only.
            result_path.write_text(
                json.dumps(
                    final.model_dump(mode="json"),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            write_immutable(result_path, final)
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


def replay_case(
    manifest: BenchmarkManifest,
    result: EvaluationResult,
    case_id: str,
    cache_root: Path,
) -> dict[str, object]:
    """Verify all immutable inputs and lineage needed to replay one case."""
    case = next((item for item in manifest.cases if item.case_id == case_id), None)
    case_result = next(
        (item for item in result.results if item.case_id == case_id), None
    )
    if case is None or case_result is None:
        raise ValueError(f"unknown or missing case record: {case_id}")
    failures = validate_lineage(case, cache_root, require_files=True)
    input_path = cache_root / case.input_uri
    if not input_path.exists() or file_hash(input_path) != case.content_sha256:
        failures.append(f"case input mismatch:{case_id}")
    return {
        "case_id": case_id,
        "replayable": not failures,
        "failures": sorted(set(failures)),
        "manifest_hash": manifest.manifest_hash,
        "run_hash": result.immutable_hash,
        "run_id": result.run.run_id,
        "arm": result.run.arm.value,
        "seed": result.run.seed,
    }


async def replay_case_live(
    manifest: BenchmarkManifest,
    result: EvaluationResult,
    case_id: str,
    cache_root: Path,
    executor: CaseExecutor,
) -> dict[str, object]:
    """Re-execute one case against a real executor and diff it against the frozen record.

    This extends :func:`replay_case` (which only checks immutable lineage) by
    actually invoking the same critic executor used for the original run and
    comparing the newly produced :class:`CaseResult` against the frozen one.
    Sampling is pinned to temperature 0 in the frozen model configuration, so
    a genuine implementation/environment mismatch shows up as an outcome or
    predicted-defect drift rather than as expected sampling noise.
    """
    static_report = replay_case(manifest, result, case_id, cache_root)
    if not static_report["replayable"]:
        return {**static_report, "live_replay_attempted": False}
    case = next(item for item in manifest.cases if item.case_id == case_id)
    frozen_result = next(item for item in result.results if item.case_id == case_id)
    live_result = await executor(case, result.run)
    drift: list[str] = []
    if live_result.outcome != frozen_result.outcome:
        drift.append(
            f"outcome mismatch: frozen={frozen_result.outcome.value} "
            f"live={live_result.outcome.value}"
        )
    if set(live_result.predicted_defects) != set(frozen_result.predicted_defects):
        drift.append(
            "predicted_defects mismatch: "
            f"frozen={sorted(frozen_result.predicted_defects)} "
            f"live={sorted(live_result.predicted_defects)}"
        )
    return {
        **static_report,
        "live_replay_attempted": True,
        "live_replay_matches": not drift,
        "live_replay_drift": drift,
        "live_outcome": live_result.outcome.value,
        "frozen_outcome": frozen_result.outcome.value,
    }
