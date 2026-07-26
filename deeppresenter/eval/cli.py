"""Typer commands for reproducible offline evaluation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .io import capture_environment, content_hash, git_commit, write_immutable
from .models import (
    Arm,
    BenchmarkManifest,
    CaseResult,
    EvaluationRun,
    ModelRecord,
    Outcome,
    Suite,
    default_cache_dir,
)
from .pipeline import DatasetPipeline, PreparationStage, run_render_stage
from .prepare import prepare_manifest
from .preregister import freeze_preregistration
from .runner import EvaluationRunner, replay_case
from .summarize import summarize_paired_runs, summarize_run
from .slideaudit import (
    SLIDEAUDIT_DATASET_ID,
    build_slideaudit_manifest,
    freeze_slideaudit_sample,
)
from .zenodo10k import ZENODO10K_REVISION, freeze_zenodo10k_sample

app = typer.Typer(
    help="Prepare, run, and summarize frozen Slidex evaluations.", no_args_is_help=True
)


@app.command()
def prepare(
    spec: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path | None, typer.Option()] = None,
    freeze: Annotated[
        bool,
        typer.Option(
            help="Run confirmatory dataset gates before accepting the manifest."
        ),
    ] = False,
    benchmark_kind: Annotated[str, typer.Option()] = "controlled_pairs",
) -> None:
    destination = output or default_cache_dir() / "manifests" / f"{spec.stem}.json"
    manifest = prepare_manifest(spec, destination)
    if freeze:
        from .integrity import freeze_gate

        report = freeze_gate(manifest, spec.parent, benchmark_kind)
        audit_path = default_cache_dir() / "audits" / f"{spec.stem}-freeze.json"
        write_immutable(audit_path, report)
        if not report["passed"]:
            raise typer.BadParameter(f"dataset freeze gates failed; see {audit_path}")
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
        judge_model=ModelRecord(
            provider="ark",
            model=os.getenv("ARK_JUDGE_MODEL", "doubao-seed-2-1-turbo-260628"),
            capabilities={"text": True, "vision": True, "structured_output": True},
        ),
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


@app.command(name="summarize-paired")
def summarize_paired_command(
    manifest: Annotated[Path, typer.Argument(exists=True, readable=True)],
    result: Annotated[list[Path], typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Summarize matched no-critic, generic, and hybrid E2E runs."""
    report = summarize_paired_runs(manifest, result, output)
    typer.echo(json.dumps(report["paired_effects"], sort_keys=True))


@app.command(name="replay")
def replay_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    result_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    case_id: Annotated[str, typer.Option()],
    cache_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Verify the immutable artifact lineage required to replay one case."""
    from .models import EvaluationResult

    manifest = BenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    result = EvaluationResult.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    report = replay_case(manifest, result, case_id, cache_dir or default_cache_dir())
    typer.echo(json.dumps(report, sort_keys=True))
    if not report["replayable"]:
        raise typer.Exit(code=1)


@app.command(name="dataset-stage")
def dataset_stage(
    stage: Annotated[PreparationStage, typer.Argument()],
    pipeline_id: Annotated[str, typer.Option()],
    cache_dir: Annotated[Path | None, typer.Option()] = None,
    source_spec: Annotated[
        Path | None, typer.Option(exists=True, readable=True)
    ] = None,
    annotations: Annotated[
        Path | None, typer.Option(exists=True, readable=True)
    ] = None,
    allowed_license: Annotated[list[str] | None, typer.Option()] = None,
    variants: Annotated[int, typer.Option(min=1)] = 1,
    dpi: Annotated[int, typer.Option(min=72)] = 144,
    render_concurrency: Annotated[int, typer.Option(min=1, max=8)] = 2,
    benchmark_id: Annotated[str, typer.Option()] = "slidex-phase13",
    revision: Annotated[str, typer.Option()] = "v1",
    config_hash: Annotated[list[str] | None, typer.Option(help="KEY=SHA256")] = None,
    benchmark_kind: Annotated[str, typer.Option()] = "controlled_pairs",
    scope: Annotated[str, typer.Option()] = "full_taxonomy",
    require_files: Annotated[bool, typer.Option()] = True,
) -> None:
    """Run exactly one resumable dataset stage; previous stages are mandatory."""
    pipeline = DatasetPipeline(cache_dir or default_cache_dir())
    if stage is PreparationStage.ACQUIRE:
        if source_spec is None:
            raise typer.BadParameter("--source-spec is required for acquire")
        result = pipeline.acquire(pipeline_id, source_spec, set(allowed_license or []))
    elif stage is PreparationStage.NORMALIZE:
        result = pipeline.normalize(pipeline_id)
    elif stage is PreparationStage.CLUSTER:
        result = pipeline.cluster(pipeline_id)
    elif stage is PreparationStage.SPLIT:
        result = pipeline.split(pipeline_id)
    elif stage is PreparationStage.MUTATE:
        result = pipeline.mutate(pipeline_id, variants=variants)
    elif stage is PreparationStage.RENDER:
        result = run_render_stage(
            pipeline, pipeline_id, dpi, concurrency=render_concurrency
        )
    elif stage is PreparationStage.VALIDATE:
        result = pipeline.validate(pipeline_id)
    elif stage is PreparationStage.ANNOTATE:
        if annotations is None:
            raise typer.BadParameter("--annotations is required for annotate")
        result = pipeline.annotate(pipeline_id, annotations)
    elif stage is PreparationStage.FREEZE:
        hashes = {}
        for item in config_hash or []:
            if "=" not in item:
                raise typer.BadParameter("--config-hash must use KEY=SHA256")
            key, value = item.split("=", 1)
            hashes[key] = value
        result = pipeline.freeze(pipeline_id, benchmark_id, revision, hashes, scope=scope)
    else:
        result = pipeline.audit(
            pipeline_id, benchmark_kind, require_files=require_files
        )
    typer.echo(json.dumps(result, sort_keys=True))


@app.command(name="preregister")
def preregister_command(
    output: Annotated[Path, typer.Option()],
    config_hash: Annotated[list[str], typer.Option(help="KEY=SHA256")],
) -> None:
    hashes = {}
    for item in config_hash:
        if "=" not in item:
            raise typer.BadParameter("--config-hash must use KEY=SHA256")
        key, value = item.split("=", 1)
        hashes[key] = value
    record = freeze_preregistration(output, hashes)
    typer.echo(json.dumps({"path": str(output), "hash": record.preregistration_hash}))


@app.command(name="zenodo10k-freeze")
def zenodo10k_freeze_command(
    cache_dir: Annotated[Path | None, typer.Option()] = None,
    revision: Annotated[str, typer.Option()] = ZENODO10K_REVISION,
    sample_size: Annotated[int, typer.Option(min=1)] = 60,
    seed: Annotated[int, typer.Option()] = 13,
    allowed_license: Annotated[list[str] | None, typer.Option()] = None,
    min_bytes: Annotated[int, typer.Option(min=1)] = 100_000,
    max_bytes: Annotated[int, typer.Option(min=1)] = 8 * 1024 * 1024,
    hf_endpoint: Annotated[str | None, typer.Option()] = None,
    workers: Annotated[int, typer.Option(min=1, max=16)] = 4,
) -> None:
    """Freeze a deterministic, license-filtered Zenodo10K PPTX sample."""
    result = freeze_zenodo10k_sample(
        cache_dir or default_cache_dir(),
        revision=revision,
        sample_size=sample_size,
        seed=seed,
        allowed_licenses=tuple(allowed_license or ["cc-by-4.0"]),
        min_bytes=min_bytes,
        max_bytes=max_bytes,
        endpoint=hf_endpoint,
        workers=workers,
    )
    typer.echo(
        json.dumps(
            {
                "frozen_sources": len(result["sources"]),
                "revision": result["revision"],
                "freeze_hash": result["freeze_hash"],
            },
            sort_keys=True,
        )
    )


@app.command(name="slideaudit-freeze")
def slideaudit_freeze_command(
    repo_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    cache_dir: Annotated[Path | None, typer.Option()] = None,
    license: Annotated[str, typer.Option()] = "cc-by-4.0",
    allowed_license: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    """Freeze the SlideAudit image-only corpus from a local git clone.

    ``repo_dir`` must be a clone of https://github.com/zhuohaouw/SlideAudit
    (``git clone https://github.com/zhuohaouw/SlideAudit.git``); this command
    never fetches network data itself, keeping acquisition explicit.
    """
    output_dir = cache_dir or default_cache_dir()
    frozen = freeze_slideaudit_sample(
        output_dir,
        repo_dir,
        license=license,
        allowed_licenses=tuple(allowed_license or ["cc-by-4.0"]),
    )
    manifest_path = output_dir / "manifests" / f"slideaudit-{frozen['revision']}.json"
    manifest = build_slideaudit_manifest(frozen, output_dir, manifest_path)
    typer.echo(
        json.dumps(
            {
                "dataset_id": SLIDEAUDIT_DATASET_ID,
                "revision": frozen["revision"],
                "slide_count": frozen["slide_count"],
                "freeze_hash": frozen["freeze_hash"],
                "manifest": str(manifest_path),
                "manifest_hash": manifest.manifest_hash,
            },
            sort_keys=True,
        )
    )
