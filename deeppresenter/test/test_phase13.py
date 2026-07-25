"""Phase 13 reproducible evaluation harness tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from deeppresenter.eval.io import write_immutable
from deeppresenter.eval.metrics import (
    binary_metrics,
    calibration,
    holm_adjust,
    weighted_kappa,
)
from deeppresenter.eval.models import (
    Arm,
    BenchmarkManifest,
    CaseResult,
    EvaluationCase,
    EvaluationRun,
    Outcome,
    SourceRecord,
    Split,
    Suite,
)
from deeppresenter.eval.prepare import (
    deterministic_case_id,
    deterministic_split,
    prepare_manifest,
    require_nonzero_pixel_difference,
)
from deeppresenter.eval.runner import EvaluationRunner, validate_paired_runs
from deeppresenter.eval.summarize import summarize_run

pytestmark = pytest.mark.unit


def source(path: Path) -> SourceRecord:
    path.write_text("open fixture", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SourceRecord(
        source_id="source",
        url="https://example.org/open",
        license="CC-BY-4.0",
        revision="v1",
        sha256=digest,
        acquired_at=datetime.now(UTC),
        local_path=path.name,
        redistributable=True,
    )


def test_deterministic_ids_split_and_leakage() -> None:
    assert deterministic_case_id("deck", "G1", 0) == deterministic_case_id(
        "deck", "G1", 0
    )
    assert deterministic_split("cluster") == deterministic_split("cluster")
    base = dict(
        source_id="source", input_uri="a", cluster_id="same", content_sha256="a" * 64
    )
    cases = [
        EvaluationCase(
            case_id="a", parent_deck_id="deck-a", split=Split.DEVELOPMENT, **base
        ),
        EvaluationCase(
            case_id="b", parent_deck_id="deck-b", split=Split.SEALED_TEST, **base
        ),
    ]
    with pytest.raises(ValueError, match="split leakage"):
        BenchmarkManifest(
            benchmark_id="x",
            revision="1",
            created_at=datetime.now(UTC),
            sources=[],
            cases=cases,
        )


def test_prepare_is_reproducible_and_checks_license(tmp_path: Path) -> None:
    item = source(tmp_path / "fixture.txt")
    spec = {
        "benchmark_id": "smoke",
        "revision": "v1",
        "allowed_licenses": ["CC-BY-4.0"],
        "sources": [item.model_dump(mode="json")],
        "cases": [
            {
                "parent_deck_id": "deck",
                "source_id": "source",
                "defect_class": "G1",
                "input_uri": "fixture.txt",
                "content_sha256": item.sha256,
            }
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    first = prepare_manifest(spec_path, tmp_path / "a.json")
    second = prepare_manifest(spec_path, tmp_path / "b.json")
    repeated = prepare_manifest(spec_path, tmp_path / "a.json")
    assert first.manifest_hash == second.manifest_hash == repeated.manifest_hash
    assert first.cases[0].case_id == second.cases[0].case_id
    spec["allowed_licenses"] = ["MIT"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="license"):
        prepare_manifest(spec_path, tmp_path / "c.json")


def test_pixel_diff_rejects_snapping(tmp_path: Path) -> None:
    clean, changed = tmp_path / "clean.png", tmp_path / "changed.png"
    Image.new("RGB", (4, 4), "white").save(clean)
    Image.new("RGB", (4, 4), "white").save(changed)
    with pytest.raises(ValueError, match="zero pixel"):
        require_nonzero_pixel_difference(clean, changed)
    Image.new("RGB", (4, 4), "black").save(changed)
    assert require_nonzero_pixel_difference(clean, changed) > 0


def test_metrics_keep_abstentions_and_calibration() -> None:
    metrics = binary_metrics([True, True, False, False], [True, False, True, False])
    assert metrics["balanced_accuracy"] == 0.5
    assert calibration([0.9, 0.1], [True, False])["brier_score"] == pytest.approx(0.01)
    assert holm_adjust([0.01, 0.04]) == [0.02, 0.04]
    assert weighted_kappa([1, 2, 3], [1, 2, 3]) == 1


@pytest.mark.asyncio
async def test_runner_resume_errors_and_immutable_summary(tmp_path: Path) -> None:
    item = source(tmp_path / "fixture.txt")
    case = EvaluationCase(
        case_id="case",
        parent_deck_id="deck",
        source_id="source",
        split=Split.DEVELOPMENT,
        input_uri="fixture.txt",
        cluster_id="deck",
        content_sha256=item.sha256,
    )
    manifest = BenchmarkManifest(
        benchmark_id="smoke",
        revision="v1",
        created_at=datetime.now(UTC),
        sources=[item],
        cases=[case],
        manifest_hash="m",
    )
    manifest_path = tmp_path / "manifest.json"
    write_immutable(manifest_path, manifest)
    run = EvaluationRun(
        run_id="run",
        suite=Suite.INTRINSIC,
        arm=Arm.HYBRID,
        seed=0,
        manifest_hash="m",
        config_hash="c",
        git_commit="commit",
    )
    calls = 0

    async def executor(case, run):
        nonlocal calls
        calls += 1
        return CaseResult(case_id=case.case_id, outcome=Outcome.DEFER)

    runner = EvaluationRunner(tmp_path / "runs", concurrency=2)
    result = await runner.run(manifest, run, executor)
    await runner.run(manifest, run, executor)
    assert calls == 1 and result.results[0].outcome is Outcome.DEFER
    report = summarize_run(
        manifest_path, tmp_path / "runs/run/result.json", tmp_path / "summary.json"
    )
    assert report["metrics"]["outcomes"] == {"defer": 1}
    with pytest.raises(FileExistsError):
        write_immutable(tmp_path / "summary.json", {"changed": True})


def test_three_arm_pairing_enforces_shared_artifact_and_budget() -> None:
    runs = [
        EvaluationRun(
            run_id=arm.value,
            suite=Suite.E2E,
            arm=arm,
            seed=1,
            manifest_hash="m",
            config_hash="c",
            git_commit="g",
            initial_artifact_id="same",
            metadata={"task_id": "task"},
        )
        for arm in (Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID)
    ]
    validate_paired_runs(runs)
    runs[-1].initial_artifact_id = "different"
    with pytest.raises(ValueError, match="initial artifact"):
        validate_paired_runs(runs)


@pytest.mark.asyncio
async def test_e2e_generates_once_before_three_arm_fork() -> None:
    from deeppresenter.eval.e2e import run_paired_task

    case = EvaluationCase(
        case_id="task",
        parent_deck_id="deck",
        source_id="source",
        split=Split.SEALED_TEST,
        input_uri="input",
        cluster_id="deck",
        content_sha256="a" * 64,
    )
    runs = [
        EvaluationRun(
            run_id=arm.value,
            suite=Suite.E2E,
            arm=arm,
            seed=7,
            manifest_hash="m",
            config_hash="c",
            git_commit="g",
        )
        for arm in (Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID)
    ]
    generated = 0

    async def generate(case, seed):
        nonlocal generated
        generated += 1
        return "artifact-first"

    async def repair(case, run):
        return CaseResult(
            case_id=case.case_id,
            outcome=Outcome.PASS,
            artifact_lineage=[run.initial_artifact_id],
        )

    results = await run_paired_task(case, 7, runs, generate, repair)
    assert generated == 1
    assert len(results) == 3
    assert {run.initial_artifact_id for run in runs} == {"artifact-first"}
