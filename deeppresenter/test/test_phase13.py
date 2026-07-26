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


def test_defect_specific_mutations_are_deterministic(tmp_path: Path) -> None:
    from deeppresenter.eval.mutations import mutate_pptx
    from deeppresenter.slidex.models import DefectClass

    fixture = Path("pptagent/test/test.pptx")
    supported = [
        DefectClass.G1,
        DefectClass.G2,
        DefectClass.G3,
        DefectClass.G4,
        DefectClass.G5,
        DefectClass.G6,
        DefectClass.G7,
        DefectClass.S1,
        DefectClass.S3,
        DefectClass.S4,
        DefectClass.S5,
        DefectClass.S6,
    ]
    for defect in supported:
        first = tmp_path / f"{defect.value}-a.pptx"
        second = tmp_path / f"{defect.value}-b.pptx"
        left = mutate_pptx(fixture, first, defect, "fixture", 0)
        right = mutate_pptx(fixture, second, defect, "fixture", 0)
        assert left == right
        assert first.read_bytes() == second.read_bytes()

    # S2 (narrative-order break) needs a multi-slide deck, unlike the other
    # twelve single-slide-scoped mutations above.
    multi_slide_fixture = Path("pptagent/templates/default/source.pptx")
    s2_first = tmp_path / "S2-a.pptx"
    s2_second = tmp_path / "S2-b.pptx"
    s2_left = mutate_pptx(multi_slide_fixture, s2_first, DefectClass.S2, "fixture", 0)
    s2_right = mutate_pptx(multi_slide_fixture, s2_second, DefectClass.S2, "fixture", 0)
    assert s2_left == s2_right
    assert s2_first.read_bytes() == s2_second.read_bytes()


def test_image_integrity_records_bbox_and_zero_signal(tmp_path: Path) -> None:
    from deeppresenter.eval.integrity import image_difference, validate_integrity_record
    from deeppresenter.eval.models import IntegrityStatus

    clean = tmp_path / "clean.png"
    same = tmp_path / "same.png"
    changed = tmp_path / "changed.png"
    Image.new("RGB", (20, 20), "white").save(clean)
    Image.new("RGB", (20, 20), "white").save(same)
    image = Image.new("RGB", (20, 20), "white")
    for x in range(4, 8):
        for y in range(5, 10):
            image.putpixel((x, y), (0, 0, 0))
    image.save(changed)
    assert (
        validate_integrity_record(image_difference(clean, same))
        is IntegrityStatus.PIXEL_DIFF_ZERO
    )
    record = image_difference(clean, changed)
    assert record.changed_bbox == (4, 5, 8, 10)
    assert validate_integrity_record(record) is IntegrityStatus.VALID


def test_crosswalk_preserves_unmapped_and_multi_label() -> None:
    from deeppresenter.eval.datasets import (
        apply_slideaudit_crosswalk,
        validate_slideaudit_crosswalk,
    )
    from deeppresenter.eval.models import CrosswalkEntry

    entries = [
        CrosswalkEntry(
            source_label="crowded",
            target_labels=["S4", "G1"],
            rationale="Both density and overflow may apply.",
            evidence_condition="image_only",
            version="v1",
            reviewed=True,
        )
    ]
    validate_slideaudit_crosswalk(entries)
    case = EvaluationCase(
        case_id="slideaudit",
        parent_deck_id="deck",
        source_id="source",
        split=Split.SEALED_TEST,
        input_uri="slide.png",
        cluster_id="deck",
        content_sha256="a" * 64,
    )
    mapped = apply_slideaudit_crosswalk(case, ["crowded", "unknown"], entries)
    assert {label.defect_class.value for label in mapped.labels} == {"S4", "G1"}
    assert mapped.metadata["unmapped_labels"] == ["unknown"]


def test_manifest_rejects_near_duplicate_cross_split() -> None:
    common = dict(
        source_id="source",
        content_sha256="a" * 64,
        metadata={"text_fingerprint": "near-duplicate"},
    )
    with pytest.raises(ValueError, match="split leakage"):
        BenchmarkManifest(
            benchmark_id="x",
            revision="1",
            created_at=datetime.now(UTC),
            sources=[],
            cases=[
                EvaluationCase(
                    case_id="a",
                    parent_deck_id="a",
                    split=Split.PILOT,
                    input_uri="a",
                    cluster_id="a",
                    **common,
                ),
                EvaluationCase(
                    case_id="b",
                    parent_deck_id="b",
                    split=Split.SEALED_TEST,
                    input_uri="b",
                    cluster_id="b",
                    **common,
                ),
            ],
        )


def test_pipeline_requires_stage_order_and_keeps_rejections(tmp_path: Path) -> None:
    from deeppresenter.eval.pipeline import DatasetPipeline

    pipeline = DatasetPipeline(tmp_path / "cache")
    with pytest.raises(ValueError, match="requires acquire"):
        pipeline.normalize("p")
    bad = tmp_path / "sources.json"
    bad.write_text(
        json.dumps(
            [
                {
                    "source_id": "bad",
                    "url": "https://example.invalid/a",
                    "license": "unknown",
                    "revision": "v1",
                    "sha256": "a" * 64,
                    "acquired_at": datetime.now(UTC).isoformat(),
                }
            ]
        )
    )
    result = pipeline.acquire("p", bad, {"CC-BY-4.0"})
    assert not result["acquired"] and result["rejected"][0]["source_id"] == "bad"


def test_freeze_gate_rejects_incomplete_confirmatory_manifest(tmp_path: Path) -> None:
    from deeppresenter.eval.integrity import freeze_gate

    manifest = BenchmarkManifest(
        benchmark_id="incomplete",
        revision="v1",
        created_at=datetime.now(UTC),
        sources=[],
        cases=[],
    )
    report = freeze_gate(manifest, tmp_path, "controlled_pairs")
    assert not report["passed"]
    assert any("G1:defect count" in item for item in report["failures"])


def test_local_acquisition_enforces_license_hash_and_resume(tmp_path: Path) -> None:
    from deeppresenter.eval.io import file_hash
    from deeppresenter.eval.pipeline import DatasetPipeline

    fixture = tmp_path / "fixture.pptx"
    fixture.write_bytes(Path("pptagent/test/test.pptx").read_bytes())
    spec = tmp_path / "sources.json"
    spec.write_text(
        json.dumps(
            [
                {
                    "source_id": "fixture",
                    "dataset_id": "fixture-set",
                    "url": "https://example.org/fixture.pptx",
                    "license": "CC-BY-4.0",
                    "license_text_sha256": "b" * 64,
                    "revision": "commit-1",
                    "sha256": file_hash(fixture),
                    "acquired_at": datetime.now(UTC).isoformat(),
                    "local_path": fixture.name,
                }
            ]
        )
    )
    pipeline = DatasetPipeline(tmp_path / "cache")
    first = pipeline.acquire("local", spec, {"CC-BY-4.0"})
    second = pipeline.acquire("local", spec, {"CC-BY-4.0"})
    assert first == second
    assert first["acquired"][0]["path"].startswith("sources/")


def test_preregistration_is_replayable_and_revision_safe(tmp_path: Path) -> None:
    from deeppresenter.eval.preregister import freeze_preregistration

    output = tmp_path / "registration.json"
    first = freeze_preregistration(output, {"router": "a" * 64})
    second = freeze_preregistration(output, {"router": "a" * 64})
    assert first.preregistration_hash == second.preregistration_hash
    with pytest.raises(FileExistsError, match="preregistration"):
        freeze_preregistration(output, {"router": "b" * 64})


def test_zenodo10k_sampling_is_deterministic_and_year_stratified() -> None:
    from deeppresenter.eval.zenodo10k import _select_stratified

    rows = [
        {
            "checksum": f"md5:{index:032x}",
            "doi": f"10.1/{index}",
            "filename": f"{index}.pptx",
            "created": f"{2018 + index % 3}-01-01",
        }
        for index in range(12)
    ]
    first = _select_stratified(rows, 6, 13)
    second = _select_stratified(list(reversed(rows)), 6, 13)
    assert first == second
    counts: dict[str, int] = {}
    for item in first:
        year = item["created"][:4]
        counts[year] = counts.get(year, 0) + 1
    assert counts == {"2018": 2, "2019": 2, "2020": 2}


def test_zenodo10k_pptx_validation_checks_md5_and_zip(tmp_path: Path) -> None:
    import hashlib

    from deeppresenter.eval.zenodo10k import _validate_pptx

    fixture = tmp_path / "fixture.pptx"
    fixture.write_bytes(Path("pptagent/test/test.pptx").read_bytes())
    md5 = hashlib.md5(fixture.read_bytes(), usedforsecurity=False).hexdigest()
    assert len(_validate_pptx(fixture, md5)) == 64
    with pytest.raises(ValueError, match="MD5 mismatch"):
        _validate_pptx(fixture, "0" * 32)


def test_repair_and_paired_effect_metrics() -> None:
    from deeppresenter.eval.metrics import paired_effect, repair_metrics

    results = [
        CaseResult(
            case_id="a",
            outcome=Outcome.PASS,
            repair_rounds=1,
            target_removed=True,
        ),
        CaseResult(
            case_id="b",
            outcome=Outcome.FAIL,
            repair_rounds=3,
            target_removed=False,
            collateral_defects=1,
        ),
    ]
    repair = repair_metrics(results)
    assert repair["target_defect_removal_rate"] == 0.5
    assert repair["first_round_success_rate"] == 0.5
    assert repair["three_round_cumulative_success_rate"] == 0.5
    effect = paired_effect([False, False, True], [True, False, True], samples=100)
    assert effect["absolute_difference"] == pytest.approx(1 / 3)
    assert len(effect["bootstrap_95_ci"]) == 2


def test_intrinsic_controls_and_balanced_reference_order() -> None:
    from deeppresenter.eval.executors import prompt_hash, run_intrinsic_case

    assert prompt_hash(Arm.FROZEN_HYBRID, "G1") != prompt_hash(Arm.C0, "G1")
    case = EvaluationCase(
        case_id="12345678case",
        parent_deck_id="deck",
        source_id="source",
        split=Split.SEALED_TEST,
        input_uri="input",
        cluster_id="deck",
        content_sha256="a" * 64,
    )
    run = EvaluationRun(
        run_id="intrinsic",
        suite=Suite.INTRINSIC,
        arm=Arm.C0_X10,
        seed=0,
        manifest_hash="m",
        config_hash="c",
        git_commit="g",
    )
    orders = []

    async def call(case, prompt, order):
        from deeppresenter.eval.executors import CriticObservation

        orders.append(order)
        return CriticObservation(outcome=Outcome.PASS, raw_output=prompt)

    import asyncio

    result = asyncio.run(run_intrinsic_case(case, run, call))
    assert result.outcome is Outcome.PASS
    assert orders.count("AB") == orders.count("BA") == 5


@pytest.mark.asyncio
async def test_fake_three_arm_budget_defer_error_and_resume(tmp_path: Path) -> None:
    item = source(tmp_path / "fixture.txt")
    cases = [
        EvaluationCase(
            case_id=case_id,
            parent_deck_id=case_id,
            source_id=item.source_id,
            split=Split.SEALED_TEST,
            input_uri="fixture.txt",
            cluster_id=case_id,
            content_sha256=item.sha256,
        )
        for case_id in ("defer", "error")
    ]
    manifest = BenchmarkManifest(
        benchmark_id="fake",
        revision="v1",
        created_at=datetime.now(UTC),
        sources=[item],
        cases=cases,
        manifest_hash="m",
    )
    calls = 0

    async def executor(case, run):
        nonlocal calls
        calls += 1
        if case.case_id == "error":
            raise RuntimeError("fake model failure")
        return CaseResult(case_id=case.case_id, outcome=Outcome.DEFER)

    run = EvaluationRun(
        run_id="fake-run",
        suite=Suite.INTRINSIC,
        arm=Arm.FROZEN_HYBRID,
        seed=0,
        manifest_hash="m",
        config_hash="c",
        git_commit="g",
        model_budget=3,
        prompt_hashes={"frozen_hybrid": "a" * 64},
    )
    runner = EvaluationRunner(tmp_path / "runs")
    first = await runner.run(manifest, run, executor)
    second = await runner.run(manifest, run, executor)
    assert calls == 2
    assert [item.outcome for item in first.results] == [Outcome.DEFER, Outcome.ERROR]
    assert second.immutable_hash == first.immutable_hash


def test_replay_case_verifies_full_lineage(tmp_path: Path) -> None:
    from deeppresenter.eval.models import (
        EvaluationResult,
        LineageRecord,
        PreparationRecord,
    )
    from deeppresenter.eval.runner import replay_case

    item = source(tmp_path / "fixture.txt")
    case = EvaluationCase(
        case_id="case",
        parent_deck_id="deck",
        source_id=item.source_id,
        split=Split.SEALED_TEST,
        input_uri="fixture.txt",
        cluster_id="deck",
        content_sha256=item.sha256,
        preparation_record=PreparationRecord(
            lineage=[
                LineageRecord(
                    artifact_id="source",
                    kind="source",
                    uri="fixture.txt",
                    sha256=item.sha256,
                )
            ]
        ),
    )
    manifest = BenchmarkManifest(
        benchmark_id="replay",
        revision="v1",
        created_at=datetime.now(UTC),
        sources=[item],
        cases=[case],
        manifest_hash="m",
    )
    run = EvaluationRun(
        run_id="run",
        suite=Suite.INTRINSIC,
        arm=Arm.C0,
        seed=0,
        manifest_hash="m",
        config_hash="c",
        git_commit="g",
    )
    result = EvaluationResult(
        run=run,
        results=[CaseResult(case_id="case", outcome=Outcome.PASS)],
        immutable_hash="r",
    )
    assert replay_case(manifest, result, "case", tmp_path)["replayable"]


def test_paired_summary_reports_effects(tmp_path: Path) -> None:
    from deeppresenter.eval.io import content_hash
    from deeppresenter.eval.models import EvaluationResult
    from deeppresenter.eval.summarize import summarize_paired_runs

    item = source(tmp_path / "fixture.txt")
    case = EvaluationCase(
        case_id="case",
        parent_deck_id="deck",
        source_id=item.source_id,
        split=Split.SEALED_TEST,
        input_uri="fixture.txt",
        cluster_id="deck",
        content_sha256=item.sha256,
    )
    manifest = BenchmarkManifest(
        benchmark_id="paired",
        revision="v1",
        created_at=datetime.now(UTC),
        sources=[item],
        cases=[case],
        manifest_hash="m",
    )
    manifest_path = tmp_path / "manifest.json"
    write_immutable(manifest_path, manifest)
    paths = []
    for arm in (Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID):
        run = EvaluationRun(
            run_id=arm.value,
            suite=Suite.E2E,
            arm=arm,
            seed=0,
            manifest_hash="m",
            config_hash="c",
            git_commit="g",
        )
        passed = arm is Arm.HYBRID
        result = EvaluationResult(
            run=run,
            results=[
                CaseResult(
                    case_id="case",
                    outcome=Outcome.PASS,
                    severe_defects=0 if passed else 1,
                    export_passed=True,
                    render_fidelity_passed=True,
                )
            ],
        )
        result.immutable_hash = content_hash(
            result.model_dump(exclude={"immutable_hash"}, mode="json")
        )
        path = tmp_path / f"{arm.value}.json"
        write_immutable(path, result)
        paths.append(path)
    report = summarize_paired_runs(
        manifest_path, paths, tmp_path / "paired-summary.json"
    )
    assert report["paired_effects"]["hybrid_vs_no_critic"]["absolute_difference"] == 1


def test_zenodo_freeze_rejects_different_cached_selection(tmp_path: Path) -> None:
    from deeppresenter.eval.zenodo10k import ZENODO10K_REVISION, freeze_zenodo10k_sample

    output = tmp_path / "zenodo10k" / ZENODO10K_REVISION
    output.mkdir(parents=True)
    (output / "frozen-sources.json").write_text(
        json.dumps(
            {
                "selection_policy": {
                    "revision": ZENODO10K_REVISION,
                    "sample_size": 60,
                    "seed": 13,
                    "allowed_licenses": ["cc-by-4.0"],
                    "min_bytes": 100_000,
                    "max_bytes": 8 * 1024 * 1024,
                },
                "sources": [],
            }
        )
    )
    with pytest.raises(FileExistsError, match="different selection inputs"):
        freeze_zenodo10k_sample(tmp_path, sample_size=30)


def _write_fake_slideaudit_repo(root: Path) -> Path:
    """Build a minimal local git repo shaped like zhuohaouw/SlideAudit."""
    import subprocess

    repo = root / "slideaudit_repo"
    (repo / "data" / "images").mkdir(parents=True)
    (repo / "data" / "annotations").mkdir(parents=True)
    (repo / "data" / "descriptions").mkdir(parents=True)

    slides = {
        "slide_0001": {
            "annotations": [
                {
                    "design_deficiency_category": "Composition & Layout",
                    "design_deficiency": "Content Overflow/Cut-off",
                    "response": True,
                    "has_strong_agreement": True,
                    "bounding_boxes": [
                        {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}
                    ],
                },
                {
                    "design_deficiency_category": "Typography",
                    "design_deficiency": "Poor Text Hierarchy",
                    "response": True,
                    "has_strong_agreement": False,
                },
            ],
            "source_type": "Gemini",
        },
        "slide_0002": {
            "annotations": [
                {
                    "design_deficiency_category": "Color",
                    "design_deficiency": "Insufficient Color Contrast for Readability",
                    "response": False,
                    "has_strong_agreement": True,
                },
            ],
            "source_type": "GDC",
        },
    }

    csv_lines = ["id,source_type,alteration,image_width,image_height"]
    for slide_id, payload in slides.items():
        numeric_id = slide_id.removeprefix("slide_")
        Image.new("RGB", (16, 9), color=(10, 20, 30)).save(
            repo / "data" / "images" / f"{slide_id}.png"
        )
        (repo / "data" / "annotations" / f"{slide_id}.json").write_text(
            json.dumps(
                {
                    "slide_id": int(numeric_id),
                    "annotations": payload["annotations"],
                    "image_dimensions": {"width": 16, "height": 9},
                }
            ),
            encoding="utf-8",
        )
        (repo / "data" / "descriptions" / f"{slide_id}.json").write_text(
            json.dumps({"slide_id": int(numeric_id), "elements": []}),
            encoding="utf-8",
        )
        csv_lines.append(f"{numeric_id},{payload['source_type']},nojitter,16,9")
    (repo / "data" / "metadata.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True, env=env
    )
    return repo


def test_slideaudit_crosswalk_covers_all_source_labels_exactly_once() -> None:
    from deeppresenter.eval.datasets import validate_slideaudit_crosswalk
    from deeppresenter.eval.slideaudit import SLIDEAUDIT_CROSSWALK

    entries = list(SLIDEAUDIT_CROSSWALK)
    validate_slideaudit_crosswalk(entries)
    labels = [entry.source_label for entry in entries]
    assert len(labels) == len(set(labels)) == 19
    assert all(entry.reviewed for entry in entries)
    assert all(entry.evidence_condition == "image_only" for entry in entries)


def test_slideaudit_freeze_and_manifest_are_idempotent(tmp_path: Path) -> None:
    from deeppresenter.eval.slideaudit import (
        build_slideaudit_manifest,
        freeze_slideaudit_sample,
        resolve_revision,
    )

    repo = _write_fake_slideaudit_repo(tmp_path)
    output_dir = tmp_path / "eval"
    frozen = freeze_slideaudit_sample(output_dir, repo)
    assert frozen["revision"] == resolve_revision(repo)
    assert frozen["slide_count"] == 2
    assert frozen["rejected_count"] == 0

    manifest_path = output_dir / "manifests" / f"slideaudit-{frozen['revision']}.json"
    manifest = build_slideaudit_manifest(frozen, output_dir, manifest_path)
    assert len(manifest.cases) == 2
    assert len(manifest.sources) == 2
    assert manifest.taxonomy_version == "slideaudit-crosswalk-v1"

    by_deck = {case.parent_deck_id: case for case in manifest.cases}
    overflow_case = by_deck["slide_0001"]
    mapped = {label.defect_class.value for label in overflow_case.labels}
    assert mapped == {"G1", "G7"}
    assert overflow_case.metadata["unmapped_labels"] == ["Poor Text Hierarchy"]
    assert overflow_case.metadata["evidence_condition"] == "image_only"
    for label in overflow_case.labels:
        assert label.evidence_condition == "image_only"

    clean_case = by_deck["slide_0002"]
    assert clean_case.labels == []
    assert overflow_case.preparation_record is not None
    assert overflow_case.preparation_record.lineage[0].kind == "source_image"

    from deeppresenter.eval.integrity import freeze_gate

    report = freeze_gate(manifest, output_dir, benchmark_kind="slideaudit")
    assert report["passed"], report["failures"]

    # Rerunning both stages must reproduce identical frozen artifacts.
    frozen_again = freeze_slideaudit_sample(output_dir, repo)
    assert frozen_again["freeze_hash"] == frozen["freeze_hash"]
    manifest_again = build_slideaudit_manifest(frozen, output_dir, manifest_path)
    assert manifest_again.manifest_hash == manifest.manifest_hash


def test_slideaudit_freeze_rejects_unapproved_license(tmp_path: Path) -> None:
    from deeppresenter.eval.slideaudit import freeze_slideaudit_sample

    repo = _write_fake_slideaudit_repo(tmp_path)
    with pytest.raises(ValueError, match="unapproved or missing license"):
        freeze_slideaudit_sample(
            tmp_path / "eval", repo, license="proprietary"
        )


def test_deck_order_difference_validates_s2_without_pixels(tmp_path: Path) -> None:
    from deeppresenter.eval.integrity import deck_order_difference, validate_integrity_record
    from deeppresenter.eval.models import IntegrityStatus
    from deeppresenter.eval.mutations import mutate_pptx
    from deeppresenter.slidex.models import DefectClass

    source = Path("pptagent/templates/default/source.pptx")
    target = tmp_path / "s2-mutated.pptx"
    mutate_pptx(source, target, DefectClass.S2, "fixture", 0)

    record = deck_order_difference(source, target)
    assert record.pixel_difference is None
    assert record.target_rule_passed is True
    assert validate_integrity_record(record) is IntegrityStatus.VALID

    # Comparing a file against itself must never look like a valid reorder.
    unchanged = deck_order_difference(source, source)
    assert unchanged.target_rule_passed is False
    assert (
        validate_integrity_record(unchanged) is IntegrityStatus.TARGET_RULE_FAILED
    )


def test_validate_integrity_record_pending_without_any_evidence() -> None:
    from deeppresenter.eval.integrity import validate_integrity_record
    from deeppresenter.eval.models import IntegrityRecord, IntegrityStatus

    assert validate_integrity_record(IntegrityRecord()) is IntegrityStatus.PENDING


def test_render_stage_no_longer_hardcodes_nine_class_image_arm() -> None:
    import inspect

    from deeppresenter.eval.pipeline import DatasetPipeline

    source = inspect.getsource(DatasetPipeline.render)
    assert "image_defects" not in source
    assert '"S2"' in source
