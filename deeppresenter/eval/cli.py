"""Typer commands for reproducible offline evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from .io import capture_environment, content_hash, git_commit
from .models import (
    Arm,
    BenchmarkManifest,
    CaseResult,
    EvaluationRun,
    Outcome,
    Suite,
    default_cache_dir,
)
from .prepare import prepare_manifest
from .runner import EvaluationRunner
from .summarize import summarize_run

app = typer.Typer(
    help="Prepare, run, and summarize frozen Slidex evaluations.", no_args_is_help=True
)


@app.command()
def prepare(
    spec: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    destination = output or default_cache_dir() / "manifests" / f"{spec.stem}.json"
    manifest = prepare_manifest(spec, destination)
    typer.echo(
        json.dumps({"manifest": str(destination), "hash": manifest.manifest_hash})
    )


@app.command(name="run")
def run_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    suite: Annotated[Suite, typer.Option()],
    arm: Annotated[Arm, typer.Option()],
    seed: Annotated[int, typer.Option()] = 0,
    concurrency: Annotated[int, typer.Option(min=1)] = 1,
    rerun_failed: Annotated[bool, typer.Option()] = False,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Run harness records. Real executors are selected by configured plugins; default is explicit missing."""
    manifest = BenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    run_id = content_hash(
        {"manifest": manifest.manifest_hash, "suite": suite, "arm": arm, "seed": seed}
    )[:24]
    record = EvaluationRun(
        run_id=run_id,
        suite=suite,
        arm=arm,
        seed=seed,
        manifest_hash=manifest.manifest_hash or "unfrozen",
        config_hash="unconfigured",
        environment=capture_environment(),
        git_commit=git_commit(),
    )

    async def unavailable(case, run):
        return CaseResult(
            case_id=case.case_id,
            outcome=Outcome.MISSING,
            error="No evaluation executor configured",
        )

    result = asyncio.run(
        EvaluationRunner(output_dir or default_cache_dir() / "runs", concurrency).run(
            manifest, record, unavailable, rerun_failed=rerun_failed
        )
    )
    typer.echo(json.dumps({"run_id": run_id, "result_hash": result.immutable_hash}))


@app.command(name="summarize")
def summarize_command(
    manifest: Annotated[Path, typer.Argument(exists=True, readable=True)],
    result: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    report = summarize_run(manifest, result, output)
    typer.echo(json.dumps(report["metrics"], sort_keys=True))
