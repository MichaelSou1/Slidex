"""Phase 13 reproducible evaluation harness tests."""

from __future__ import annotations

import hashlib
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
    mixed_effects_logistic,
    summarize,
    weighted_kappa,
)
from deeppresenter.eval.models import (
    Arm,
    BenchmarkManifest,
    CaseResult,
    DefectLabel,
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


def test_summarize_counts_unlabeled_negative_cases_as_true_negative_everywhere() -> None:
    """13.15 regression: open-world negative cases (empty ``labels``, e.g.
    SlideAudit true negatives) must contribute a true negative to *every*
    candidate defect class this corpus/arm actually predicted somewhere, not
    just to classes that happen to intersect with this one case's own
    (empty) label set. Before this fix, an unlabeled case only entered
    per-class stats when the model predicted a defect for it -- meaning it
    could only ever contribute a false positive, never a true negative,
    which silently floors specificity/balanced_accuracy for every class on
    any corpus with real negatives."""
    positive_case = EvaluationCase(
        case_id="pos-1",
        parent_deck_id="pos-1",
        source_id="slideaudit",
        split=Split.SEALED_TEST,
        input_uri="pos-1.png",
        labels=[DefectLabel(defect_class="G1", defective=True, evidence_condition="image_only")],
        cluster_id="pos-1",
        content_sha256="a" * 64,
    )
    negative_case_correct = EvaluationCase(
        case_id="neg-correct",
        parent_deck_id="neg-correct",
        source_id="slideaudit",
        split=Split.SEALED_TEST,
        input_uri="neg-correct.png",
        labels=[],
        cluster_id="neg-correct",
        content_sha256="b" * 64,
    )
    negative_case_false_positive = EvaluationCase(
        case_id="neg-fp",
        parent_deck_id="neg-fp",
        source_id="slideaudit",
        split=Split.SEALED_TEST,
        input_uri="neg-fp.png",
        labels=[],
        cluster_id="neg-fp",
        content_sha256="c" * 64,
    )
    results = [
        CaseResult(case_id="pos-1", outcome=Outcome.FAIL, predicted_defects=["G1"]),
        CaseResult(case_id="neg-correct", outcome=Outcome.PASS, predicted_defects=[]),
        CaseResult(case_id="neg-fp", outcome=Outcome.FAIL, predicted_defects=["G1"]),
    ]
    report = summarize(
        [positive_case, negative_case_correct, negative_case_false_positive], results
    )
    g1 = report["per_class"]["G1"]
    # truth=[True, False, False], prediction=[True, False, True]
    # -> tp=1, tn=1, fp=1, fn=0 -> recall=1, specificity=0.5
    assert g1["recall"] == 1.0
    assert g1["specificity"] == 0.5
    assert g1["balanced_accuracy"] == 0.75


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


@pytest.mark.asyncio
async def test_e2e_first_round_failure_does_not_block_other_arms() -> None:
    """If the shared reference generate() call raises (e.g. no_critic's own
    export gate legitimately blocks on real unassisted-generation defects --
    a meaningful 13.7 measurement, not a harness bug), repair() must still
    run independently for every arm: generic/hybrid re-run their own
    complete AgentLoop under the same task/seed and must not be skipped just
    because the discarded first-round attempt failed (regression test for
    the bug where process_case's outer except treated one generate()
    exception as three cascaded per-arm failures, silently discarding
    generic/hybrid data for every case where unassisted no_critic
    generation happened to fail)."""
    from deeppresenter.eval.e2e import FIRST_ROUND_GENERATION_FAILED, run_paired_task

    case = EvaluationCase(
        case_id="task::seed7",
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

    async def generate(case, seed):
        raise RuntimeError("export blocked by hard findings")

    repaired_arms = []

    async def repair(case, run):
        repaired_arms.append(run.arm)
        outcome = Outcome.FAIL if run.arm is Arm.NO_CRITIC else Outcome.PASS
        return CaseResult(case_id=case.case_id, outcome=outcome)

    results = await run_paired_task(case, 7, runs, generate, repair)
    assert set(repaired_arms) == {Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID}
    assert {run.initial_artifact_id for run in runs} == {FIRST_ROUND_GENERATION_FAILED}
    by_arm = dict(zip((r.arm for r in runs), results, strict=True))
    assert by_arm[Arm.NO_CRITIC].outcome is Outcome.FAIL
    assert by_arm[Arm.GENERIC].outcome is Outcome.PASS
    assert by_arm[Arm.HYBRID].outcome is Outcome.PASS


@pytest.mark.asyncio
async def test_e2e_arm_specific_repair_failure_does_not_affect_other_arms() -> None:
    """If generate() succeeds but repair() fails for only one arm, the other
    two arms must still return their own (non-error) results -- an
    arm-specific repair failure is not a cascade and must not blank out
    sibling arms."""
    from deeppresenter.eval.e2e import run_paired_task

    case = EvaluationCase(
        case_id="task::seed7",
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

    async def generate(case, seed):
        return "artifact-first"

    async def repair(case, run):
        if run.arm is Arm.GENERIC:
            raise RuntimeError("critic endpoint timed out")
        return CaseResult(
            case_id=case.case_id,
            outcome=Outcome.PASS,
            artifact_lineage=[run.initial_artifact_id],
        )

    results = await run_paired_task(case, 7, runs, generate, repair)
    by_arm = dict(zip((r.arm for r in runs), results, strict=True))
    assert by_arm[Arm.NO_CRITIC].outcome is Outcome.PASS
    assert by_arm[Arm.HYBRID].outcome is Outcome.PASS
    assert by_arm[Arm.GENERIC].outcome is Outcome.ERROR
    assert by_arm[Arm.GENERIC].error is not None
    assert "critic endpoint timed out" in by_arm[Arm.GENERIC].error


@pytest.mark.asyncio
async def test_run_e2e_manifest_expands_seeds_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """13.7 integration: build_e2e_manifest(seeds=...) + run_e2e_manifest must
    (a) give every (task, seed) pair a unique case_id so per-seed CaseResults
    never collide under summarize()'s case_id-keyed lookup, (b) generate the
    first-round artifact exactly once per (task, seed) shared across all three
    arms, and (c) skip already-persisted non-failed cases on a second call."""
    from deeppresenter.eval import e2e as e2e_module
    from deeppresenter.eval.e2e import _run_e2e_manifest_async, build_e2e_manifest
    from deeppresenter.eval.io import content_hash

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    source_path = corpus_dir / "source.md"
    source_path.write_text("# Source\n\nSome fact.\n", encoding="utf-8")
    task = {
        "task_id": "task-1",
        "task_type": "academic",
        "split": "sealed_test",
        "source": {
            "source_id": "src-1",
            "url": "https://example.org/src-1",
            "license": "CC-BY-4.0",
            "sha256": "a" * 64,
        },
        "normalized_path": "source.md",
        "brief": {
            "audience": "general",
            "purpose": "test",
            "page_count": [3, 5],
            "language": "en",
            "required_facts": [
                {
                    "fact_id": "fact-1",
                    "text": "Some fact.",
                    "source_locator": "src-1#sentence-1",
                }
            ],
            "required_sections": ["Intro"],
            "style_constraints": [],
            "forbidden_claims": [],
        },
    }
    corpus = {"tasks": [task]}
    corpus["corpus_hash"] = content_hash(corpus)
    corpus_path = corpus_dir / "e2e_task_corpus.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

    manifest = build_e2e_manifest(corpus_path, seeds=[0, 1])
    assert {case.case_id for case in manifest.cases} == {
        "task-1::seed0",
        "task-1::seed1",
    }
    assert {case.parent_deck_id for case in manifest.cases} == {"task-1"}

    generate_calls: list[tuple[str, int]] = []

    class FakeExecutor:
        def __init__(self, config_path: str, corpus_dir: Path, workspace_root: Path) -> None:
            pass

        async def generate_first_round(self, case, seed: int) -> str:
            generate_calls.append((case.case_id, seed))
            return f"artifact-{case.case_id}"

        async def repair(self, case, run) -> CaseResult:
            return CaseResult(
                case_id=case.case_id,
                outcome=Outcome.PASS,
                artifact_lineage=[run.initial_artifact_id],
            )

    monkeypatch.setattr(e2e_module, "E2ECaseExecutor", FakeExecutor)

    output_dir = tmp_path / "runs"
    results = await _run_e2e_manifest_async(
        manifest,
        config_path="unused.yaml",
        corpus_dir=corpus_dir,
        workspace_root=tmp_path / "workspace",
        output_dir=output_dir,
        max_repairs=3,
        model_budget=3,
        concurrency=1,
        rerun_failed=False,
        git_commit="test",
        environment={},
    )
    assert set(results) == {Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID}
    # Exactly one generate() call per (task, seed) pair, shared across arms.
    assert sorted(generate_calls) == [("task-1::seed0", 0), ("task-1::seed1", 1)]
    for arm, result in results.items():
        assert {r.case_id for r in result.results} == {"task-1::seed0", "task-1::seed1"}
        assert all(r.outcome == Outcome.PASS for r in result.results)

    # Second call must be idempotent: no new generate() calls for persisted cases.
    generate_calls.clear()
    results_again = await _run_e2e_manifest_async(
        manifest,
        config_path="unused.yaml",
        corpus_dir=corpus_dir,
        workspace_root=tmp_path / "workspace",
        output_dir=output_dir,
        max_repairs=3,
        model_budget=3,
        concurrency=1,
        rerun_failed=False,
        git_commit="test",
        environment={},
    )
    assert generate_calls == []
    for arm in (Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID):
        assert results_again[arm].immutable_hash == results[arm].immutable_hash

    # Output must be directly consumable by summarize()/summarize_paired_runs().
    summary = summarize_run_metrics(manifest.cases, results[Arm.HYBRID].results)
    assert summary["outcomes"]["pass"] == 2


def summarize_run_metrics(cases, results):
    from deeppresenter.eval.metrics import summarize as _summarize

    return _summarize(cases, results)


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


def test_mutation_is_isolated_to_one_slide_part(tmp_path: Path) -> None:
    """Non-target-defect regression guard: every mutation operator must touch
    at most one slide XML part (or, for S2, only the presentation's slide
    order list) and must leave every other archive entry byte-for-byte
    identical to the source. This is a structural, fully offline proxy for
    "non-target inspectors must not fire a new high-severity defect": if a
    mutation only ever changes bytes on its own target slide, it cannot by
    construction introduce a new defect on any other slide, and any change
    on the target slide beyond the declared operator would show up as an
    unexpectedly large diff that a reviewer can inspect."""
    import zipfile

    from deeppresenter.eval.mutations import mutate_pptx
    from deeppresenter.slidex.models import DefectClass

    fixture = Path("pptagent/test/test.pptx")
    multi_slide_fixture = Path("pptagent/templates/default/source.pptx")
    with zipfile.ZipFile(fixture) as archive:
        source_bytes = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(multi_slide_fixture) as archive:
        multi_source_bytes = {name: archive.read(name) for name in archive.namelist()}

    single_slide_defects = [
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
    for defect in single_slide_defects:
        target = tmp_path / f"{defect.value}-isolation.pptx"
        record = mutate_pptx(fixture, target, defect, "fixture", 0)
        with zipfile.ZipFile(target) as archive:
            mutated_bytes = {name: archive.read(name) for name in archive.namelist()}
        assert set(mutated_bytes) == set(source_bytes), f"{defect.value} changed archive membership"
        changed = [
            name
            for name in source_bytes
            if source_bytes[name] != mutated_bytes[name]
        ]
        assert changed == [record.parameters["slide_part"]], (
            f"{defect.value} mutation touched unexpected parts: {changed}"
        )

    # S2 reorders slide IDs in the presentation part only; no slide XML byte
    # may change (the narrative-order defect is purely an ordering defect).
    s2_target = tmp_path / "S2-isolation.pptx"
    mutate_pptx(multi_slide_fixture, s2_target, DefectClass.S2, "fixture", 0)
    with zipfile.ZipFile(s2_target) as archive:
        s2_bytes = {name: archive.read(name) for name in archive.namelist()}
    assert set(s2_bytes) == set(multi_source_bytes)
    changed = [
        name
        for name in multi_source_bytes
        if multi_source_bytes[name] != s2_bytes[name]
    ]
    assert changed == ["ppt/presentation.xml"]
    for name in multi_source_bytes:
        if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
            assert multi_source_bytes[name] == s2_bytes[name], (
                "S2 must not rewrite any individual slide's own XML content"
            )


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


def test_summarize_reports_capability_downgrades_and_defer_reasons_as_is() -> None:
    """13.15 acceptance gate: capability downgrades (e.g. image-only inputs
    with no native-IR guarantees) and defer reasons must show up verbatim in
    the summary rather than being silently absorbed into pass/fail counts."""
    from deeppresenter.eval.metrics import summarize

    case_a = EvaluationCase(
        case_id="downgraded",
        parent_deck_id="deck",
        source_id="source",
        split=Split.SEALED_TEST,
        input_uri="fixture.txt",
        cluster_id="deck",
        content_sha256="a" * 64,
    )
    case_b = EvaluationCase(
        case_id="deferred",
        parent_deck_id="deck",
        source_id="source",
        split=Split.SEALED_TEST,
        input_uri="fixture.txt",
        cluster_id="deck",
        content_sha256="a" * 64,
    )
    downgraded = CaseResult(
        case_id="downgraded",
        outcome=Outcome.FAIL,
        capability_downgrade="Image-only input has no native-IR guarantees; only neural predicates are available.",
    )
    deferred = CaseResult(
        case_id="deferred",
        outcome=Outcome.DEFER,
        raw_output=[
            {
                "results": [
                    {
                        "status": "defer",
                        "evidence": [{"detail": "missing=['trusted_native_ir']"}],
                    }
                ]
            }
        ],
    )
    report = summarize([case_a, case_b], [downgraded, deferred])
    assert report["capability_downgrades"] == {
        "Image-only input has no native-IR guarantees; only neural predicates are available.": 1
    }
    assert report["defer_reasons"] == {"missing=['trusted_native_ir']": 1}
    # An error outcome's reason must also surface as-is, not be swallowed.
    errored = CaseResult(case_id="deferred", outcome=Outcome.DEFER, error="upstream timeout")
    report_with_error = summarize([case_b], [errored])
    assert report_with_error["defer_reasons"] == {"upstream timeout": 1}


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


def test_mixed_effects_logistic_recovers_treatment_effect_and_respects_clustering() -> None:
    """A strong, evenly-distributed treatment effect must be recovered with the
    correct sign and a large positive odds ratio, and the fit must not confuse
    a pure cluster-level confound (no per-observation treatment effect) with a
    real treatment effect once cluster identity is passed in."""
    import random

    rng = random.Random(13)
    outcomes: list[bool] = []
    treatment: list[bool] = []
    clusters: list[str] = []
    # 20 source decks, 6 paired (treated, control) observations each; treated
    # observations succeed with probability 0.9, control with probability 0.2,
    # so the fixed treatment effect should dominate any single deck's baseline.
    for deck_index in range(20):
        cluster_name = f"deck-{deck_index}"
        for _ in range(6):
            for is_treated in (True, False):
                p = 0.9 if is_treated else 0.2
                outcomes.append(rng.random() < p)
                treatment.append(is_treated)
                clusters.append(cluster_name)

    fit = mixed_effects_logistic(outcomes, treatment, clusters)
    assert fit["treatment_log_odds"] > 1.0
    assert fit["treatment_odds_ratio"] > pow(2.718281828459045, 1.0)
    assert fit["n_clusters"] == 20.0
    assert fit["n_observations"] == float(len(outcomes))
    assert fit["random_intercept_variance"] >= 0.0

    # A null design (treatment has no effect within any cluster) must not
    # manufacture a large spurious effect.
    null_outcomes: list[bool] = []
    null_treatment: list[bool] = []
    null_clusters: list[str] = []
    for deck_index in range(20):
        cluster_name = f"null-deck-{deck_index}"
        base_p = rng.uniform(0.2, 0.8)
        for _ in range(6):
            for is_treated in (True, False):
                null_outcomes.append(rng.random() < base_p)
                null_treatment.append(is_treated)
                null_clusters.append(cluster_name)
    null_fit = mixed_effects_logistic(null_outcomes, null_treatment, null_clusters)
    assert abs(null_fit["treatment_log_odds"]) < 1.0


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


@pytest.mark.asyncio
async def test_offline_prepare_run_summarize_smoke(tmp_path: Path) -> None:
    """Offline end-to-end smoke test over the prepare -> run -> summarize path
    using only a tiny local fixture (no network, no real model calls). This
    exercises the same immutable-artifact pipeline the CLI commands wrap:
    ``pptagent eval prepare`` -> ``pptagent eval run`` -> ``pptagent eval
    summarize``, so a regression in any stage's on-disk contract fails here
    without requiring the full frozen benchmark or model credentials."""
    item = source(tmp_path / "fixture.txt")
    spec = {
        "benchmark_id": "offline-smoke",
        "revision": "v1",
        "allowed_licenses": ["CC-BY-4.0"],
        "sources": [item.model_dump(mode="json")],
        "cases": [
            {
                "parent_deck_id": "deck-a",
                "source_id": "source",
                "defect_class": "G1",
                "input_uri": "fixture.txt",
                "content_sha256": item.sha256,
            },
            {
                "parent_deck_id": "deck-a",
                "source_id": "source",
                "target_defect_class": "G1",
                "input_uri": "fixture.txt",
                "content_sha256": item.sha256,
            },
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    # Stage 1: prepare (mirrors `pptagent eval prepare`).
    manifest = prepare_manifest(spec_path, tmp_path / "manifest.json")
    assert len(manifest.cases) == 2

    # Stage 2: run with a deterministic fake executor (mirrors `pptagent eval
    # run` without --real-models, except we return PASS/FAIL instead of
    # MISSING so summarize has non-trivial metrics to compute).
    async def fake_executor(case: EvaluationCase, run: EvaluationRun) -> CaseResult:
        is_defective = any(label.defective for label in case.labels)
        return CaseResult(
            case_id=case.case_id,
            outcome=Outcome.FAIL if is_defective else Outcome.PASS,
            predicted_defects=["G1"] if is_defective else [],
        )

    run_record = EvaluationRun(
        run_id="offline-smoke-run",
        suite=Suite.INTRINSIC,
        arm=Arm.FROZEN_HYBRID,
        seed=0,
        manifest_hash=manifest.manifest_hash or "unfrozen",
        config_hash="unconfigured",
        git_commit="offline-smoke",
    )
    runner = EvaluationRunner(tmp_path / "runs", concurrency=2)
    result = await runner.run(manifest, run_record, fake_executor)
    assert len(result.results) == 2
    result_path = tmp_path / "runs" / "offline-smoke-run" / "result.json"
    assert result_path.exists()

    # Re-running must reuse the immutable per-case records rather than calling
    # the executor again (idempotent resume contract).
    calls = 0

    async def counting_executor(case: EvaluationCase, run: EvaluationRun) -> CaseResult:
        nonlocal calls
        calls += 1
        return CaseResult(case_id=case.case_id, outcome=Outcome.PASS)

    await runner.run(manifest, run_record, counting_executor)
    assert calls == 0

    # Stage 3: summarize (mirrors `pptagent eval summarize`); must not call
    # any executor and must recompute metrics purely from immutable records.
    manifest_path = tmp_path / "manifest.json"
    report = summarize_run(manifest_path, result_path, tmp_path / "summary.json")
    assert report["manifest_hash"] == manifest.manifest_hash
    assert (tmp_path / "summary.json").exists()
    assert report["metrics"]["outcomes"]["fail"] == 1
    assert report["metrics"]["outcomes"]["pass"] == 1
    assert report["metrics"]["macro_balanced_accuracy"] == pytest.approx(1.0)
    assert report["metrics"]["model_errors"] == 0


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


@pytest.mark.export
@pytest.mark.asyncio
async def test_clean_defective_render_fidelity_via_libreoffice(tmp_path: Path) -> None:
    """End-to-end dataset-integrity check: render a clean fixture and a
    deterministically mutated defective sibling through the same LibreOffice
    renderer used to freeze the Zenodo10K benchmark, then confirm both PPTX
    files actually re-render to non-empty pixel content and that the mutation
    is visible in the final rasterized PPTX (not just in the source XML)."""
    from deeppresenter.eval.prepare import require_nonzero_pixel_difference
    from deeppresenter.eval.mutations import mutate_pptx
    from deeppresenter.slidex.export import LibreOfficeRenderer
    from deeppresenter.slidex.models import DefectClass

    renderer = LibreOfficeRenderer()
    if not renderer.executable:
        pytest.skip("LibreOffice is unavailable")

    clean_fixture = Path("pptagent/test/test.pptx")
    defective_fixture = tmp_path / "G6-defective.pptx"
    mutate_pptx(clean_fixture, defective_fixture, DefectClass.G6, "fixture", 0)

    clean_dir = tmp_path / "clean_render"
    defective_dir = tmp_path / "defective_render"
    _, clean_pages, clean_command = await renderer.render(clean_fixture, clean_dir)
    _, defective_pages, defective_command = await renderer.render(
        defective_fixture, defective_dir
    )

    assert clean_command.return_code == 0
    assert defective_command.return_code == 0
    assert clean_pages, "clean fixture must produce at least one rasterized page"
    assert defective_pages, "defective fixture must produce at least one rasterized page"
    assert clean_pages[0].exists() and clean_pages[0].stat().st_size > 0
    assert defective_pages[0].exists() and defective_pages[0].stat().st_size > 0

    # The margin-violation mutation must survive the full XML -> LibreOffice ->
    # PDF -> PNG pipeline as an observable pixel difference; a zero diff here
    # would mean the final PPTX fidelity silently swallowed the injected defect
    # (the exact "render-gap" failure mode Phase 13 requires catching).
    difference = require_nonzero_pixel_difference(clean_pages[0], defective_pages[0])
    assert difference > 0


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



def test_build_context_handles_open_world_image_only_case_without_pptx(
    tmp_path: Path,
) -> None:
    """13.4 regression: SlideAudit-style cases carry a real external image as
    input_uri, never a PPTX/zip archive. build_context() must route these
    through CaseArtifactBuilder.image_only_artifact() and never attempt
    pptx_to_slide_artifacts()/deck_outline() (which previously raised
    'File is not a zip file' / 'list index out of range')."""
    from deeppresenter.eval.models import DefectLabel, EvaluationCase, Split
    from deeppresenter.eval.real_executors import CaseArtifactBuilder, build_context
    from deeppresenter.slidex.models import ArtifactTrust

    image_path = tmp_path / "slide_0001.png"
    Image.new("RGB", (160, 90), "white").save(image_path)
    content_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    case = EvaluationCase(
        case_id="slideaudit-case-1",
        parent_deck_id="slideaudit-case-1",
        source_id="slideaudit",
        split=Split.SEALED_TEST,
        input_uri=str(image_path),
        labels=[
            DefectLabel(defect_class="G2", defective=True, evidence_condition="image_only")
        ],
        cluster_id="slideaudit-case-1",
        content_sha256=content_sha256,
        metadata={"evidence_condition": "image_only"},
    )
    builder = CaseArtifactBuilder(tmp_path)
    context = build_context(case, builder)
    assert context.artifact.trust is ArtifactTrust.IMAGE_ONLY
    assert context.render_path == str(image_path.resolve())
    assert context.reference_artifact is None
    assert context.deck_outline == []
    assert "trusted_native_ir" in context.artifact.missing_bookkeeping


def test_build_context_handles_negative_case_with_no_labels(tmp_path: Path) -> None:
    """13.15 regression: open-world negative (no-defect) cases carry an empty
    ``labels`` list -- e.g. ~38%% of SlideAudit -- so build_context() must not
    unconditionally dereference ``case.labels[0]``. Doing so previously raised
    'list index out of range' for every negative image_only case before any
    model call was made (latency_seconds == 0.0), silently turning all
    negative cases into ``error`` outcomes."""
    from deeppresenter.eval.models import EvaluationCase, Split
    from deeppresenter.eval.real_executors import CaseArtifactBuilder, build_context
    from deeppresenter.slidex.models import ArtifactTrust

    image_path = tmp_path / "slide_clean_0001.png"
    Image.new("RGB", (160, 90), "white").save(image_path)
    content_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    case = EvaluationCase(
        case_id="slideaudit-negative-1",
        parent_deck_id="slideaudit-negative-1",
        source_id="slideaudit",
        split=Split.SEALED_TEST,
        input_uri=str(image_path),
        labels=[],
        cluster_id="slideaudit-negative-1",
        content_sha256=content_sha256,
        metadata={"evidence_condition": "image_only"},
    )
    builder = CaseArtifactBuilder(tmp_path)
    context = build_context(case, builder)
    assert context.artifact.trust is ArtifactTrust.IMAGE_ONLY
    assert context.deck_outline == []


@pytest.mark.asyncio
async def test_real_hybrid_executor_negative_case_uses_whole_taxonomy_verdict(
    tmp_path: Path,
) -> None:
    """13.15 regression: RealCriticExecutors.hybrid() must not dereference
    case.labels[0] for negative cases. The frozen_hybrid arm-level verdict
    for a no-label case must come from scanning resolved_status across every
    DefectClass the router inspected: any FAIL anywhere is a false positive,
    all-DEFER is a DEFER, and otherwise it is a true-negative PASS."""
    from deeppresenter.eval.models import EvaluationCase, Split
    from deeppresenter.eval.models import Outcome as EvalOutcome
    from deeppresenter.eval.real_executors import CaseArtifactBuilder, RealCriticExecutors
    from deeppresenter.slidex.critic import _report
    from deeppresenter.slidex.models import DefectClass, InspectionResult, InspectionStatus

    image_path = tmp_path / "slide_negative.png"
    Image.new("RGB", (160, 90), "white").save(image_path)
    content_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    def make_case(case_id: str) -> EvaluationCase:
        return EvaluationCase(
            case_id=case_id,
            parent_deck_id=case_id,
            source_id="slideaudit",
            split=Split.SEALED_TEST,
            input_uri=str(image_path),
            labels=[],
            cluster_id=case_id,
            content_sha256=content_sha256,
            metadata={"evidence_condition": "image_only"},
        )

    def result(defect_class: DefectClass, status: InspectionStatus) -> InspectionResult:
        return InspectionResult(
            defect_class=defect_class,
            status=status,
            severity=0.0,
            confidence=0.4,
            inspector_version="test-1",
        )

    executors = RealCriticExecutors.__new__(RealCriticExecutors)
    executors.builder = CaseArtifactBuilder(tmp_path)

    class _FakeHybrid:
        def __init__(self, report):
            self._report = report

        async def inspect(self, context):
            return self._report

    def make_report(results: list[InspectionResult]):
        resolved_status = {item.defect_class: item.status for item in results}
        return _report(
            executors.builder.image_only_artifact(str(image_path)),
            results,
            router_version="test",
            taxonomy_version="test",
            resolved_status=resolved_status,
        )

    # Case 1: router found a defect on one class -> false positive (FAIL).
    executors._hybrid = _FakeHybrid(
        make_report(
            [
                result(DefectClass.G1, InspectionStatus.PASS),
                result(DefectClass.G2, InspectionStatus.FAIL),
            ]
        )
    )
    observation = await executors.hybrid(make_case("neg-fail"), "", "AB")
    assert observation.outcome is EvalOutcome.FAIL
    assert observation.defects == [DefectClass.G2.value]

    # Case 2: router passed on every inspected class -> true negative (PASS).
    executors._hybrid = _FakeHybrid(
        make_report(
            [
                result(DefectClass.G1, InspectionStatus.PASS),
                result(DefectClass.G2, InspectionStatus.PASS),
            ]
        )
    )
    observation = await executors.hybrid(make_case("neg-pass"), "", "AB")
    assert observation.outcome is EvalOutcome.PASS
    assert observation.defects == []

    # Case 3: router could only defer everywhere -> DEFER, not silently PASS.
    executors._hybrid = _FakeHybrid(
        make_report(
            [
                result(DefectClass.G1, InspectionStatus.DEFER),
                result(DefectClass.G2, InspectionStatus.DEFER),
            ]
        )
    )
    observation = await executors.hybrid(make_case("neg-defer"), "", "AB")
    assert observation.outcome is EvalOutcome.DEFER


def _tool_call(name: str, arguments: dict) -> "ToolCall":
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as ToolCall,
    )

    return ToolCall(
        id=f"call-{name}-{hash(json.dumps(arguments, sort_keys=True)) & 0xFFFF}",
        type="function",
        function={"name": name, "arguments": json.dumps(arguments)},
    )


def test_slide_quota_streak_accumulates_on_the_same_slide_and_resets_on_switch() -> None:
    """13.7 hardening: consecutive write_file/edit_file/inspect_slide turns on
    the same slide file accumulate a streak; touching a different slide file,
    zero slide files, or more than one slide file in the same turn resets it,
    since only genuine single-slide fixation should burn the per-slide quota."""
    from deeppresenter.agents.design import _update_streak

    streak_file, streak_count = None, 0
    streak_file, streak_count = _update_streak(
        streak_file, streak_count, [_tool_call("write_file", {"path": "slides/slide_01.html", "content": "x"})]
    )
    assert (streak_file, streak_count) == ("slides/slide_01.html", 1)

    streak_file, streak_count = _update_streak(
        streak_file,
        streak_count,
        [_tool_call("inspect_slide", {"html_file": "slides/slide_01.html"})],
    )
    assert (streak_file, streak_count) == ("slides/slide_01.html", 2)

    streak_file, streak_count = _update_streak(
        streak_file,
        streak_count,
        [_tool_call("edit_file", {"path": "slides/slide_01.html", "old": "a", "new": "b"})],
    )
    assert (streak_file, streak_count) == ("slides/slide_01.html", 3)

    # Switching to a different slide resets the streak to 1, not 0: the new
    # slide starts its own fresh fixation window.
    streak_file, streak_count = _update_streak(
        streak_file,
        streak_count,
        [_tool_call("write_file", {"path": "slides/slide_02.html", "content": "y"})],
    )
    assert (streak_file, streak_count) == ("slides/slide_02.html", 1)

    # A non-slide tool call (e.g. reading the manuscript) resets to (None, 0).
    streak_file, streak_count = _update_streak(
        streak_file, streak_count, [_tool_call("read_file", {"path": "manuscript.md"})]
    )
    assert (streak_file, streak_count) == (None, 0)

    # A turn touching two different slide files at once is not fixation on
    # one slide either, so it also resets.
    streak_file, streak_count = _update_streak(
        None,
        0,
        [
            _tool_call("write_file", {"path": "slides/slide_03.html", "content": "z"}),
            _tool_call("write_file", {"path": "slides/slide_04.html", "content": "w"}),
        ],
    )
    assert (streak_file, streak_count) == (None, 0)


@pytest.mark.asyncio
async def test_design_agent_nudges_after_exceeding_per_slide_quota(tmp_path: Path) -> None:
    """13.7 hardening: if the Design agent spends more than
    config.slidex.max_turns_per_slide consecutive turns stuck on one slide
    file, loop() must inject SLIDE_QUOTA_EXCEEDED_MSG_TEMPLATE into the tool
    observations so the agent is nudged to move on, instead of silently
    letting one hard slide exhaust the whole max_episode_steps budget (the
    real failure observed in a 13.7 E2E pilot: Design exceeded max turns
    20/20 after 8 straight edit/inspect turns on slide_01.html)."""
    from deeppresenter.agents.design import Design
    from deeppresenter.utils.constants import SLIDE_QUOTA_EXCEEDED_MSG_TEMPLATE
    from deeppresenter.utils.typings import ChatMessage, InputRequest, Role

    design = Design.__new__(Design)
    design.workspace = tmp_path
    design.config = type(
        "Cfg", (), {"slidex": type("Slidex", (), {"max_turns_per_slide": 2})()}
    )()
    design.chat_history = []

    # Three straight edit/inspect turns on slide_01 (exceeding quota=2), then
    # one turn moving on to slide_02.
    turns = [
        [_tool_call("write_file", {"path": "slides/slide_01.html", "content": "v1"})],
        [_tool_call("inspect_slide", {"html_file": "slides/slide_01.html"})],
        [_tool_call("edit_file", {"path": "slides/slide_01.html", "old": "a", "new": "b"})],
        [_tool_call("write_file", {"path": "slides/slide_02.html", "content": "v1"})],
    ]
    turn_iter = iter(turns)

    async def fake_action(**_kwargs):
        message = ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=next(turn_iter))
        design.chat_history.append(message)
        return message

    async def fake_execute(_tool_calls):
        return [ChatMessage(role=Role.TOOL, content="ok", tool_call_id="x")]

    design.action = fake_action
    design.execute = fake_execute

    request = InputRequest(instruction="test")
    seen_messages: list[ChatMessage] = []
    async for msg in design.loop(request, "manuscript.md"):
        seen_messages.append(msg)
        if len(seen_messages) >= len(turns) * 2:
            break

    tool_observations = [m for m in seen_messages if m.role is Role.TOOL]
    nudged = [
        m
        for m in tool_observations
        if any("URGENT" in block.get("text", "") for block in m.content)
    ]
    assert len(nudged) == 1, "expected exactly one quota-exceeded nudge, injected after the 3rd slide_01 turn"
    expected_text = SLIDE_QUOTA_EXCEEDED_MSG_TEMPLATE.format(turns=3, slide="slides/slide_01.html")
    assert any(block.get("text") == expected_text for block in nudged[0].content)


def test_setup_toolset_deduplicates_tools_listed_in_both_server_and_include_tools() -> None:
    """13.7 hardening: `deeppresenter/roles/Design.yaml` lists `local` under
    `include_tool_servers` and also re-lists `thinking`/`inspect_slide`/
    `finalize` under `include_tools` (to document intent). Before this fix,
    `_setup_toolset` appended each name twice, producing a `tools` list with
    duplicate `function.name` entries. gpt-4o-mini/Gemini tolerated this
    silently, but a real Kimi-k3 smoke run hit a hard `400 Invalid request:
    function name thinking is duplicated` on every single turn. This test
    pins the fix: overlapping names must appear exactly once in `self.tools`,
    regardless of whether they came from the server sweep or the explicit
    `include_tools` list."""
    from deeppresenter.agents.agent import Agent
    from deeppresenter.utils.typings import RoleConfig, ToolSet

    def _tool_schema(name: str) -> dict:
        return {"type": "function", "function": {"name": name, "parameters": {}}}

    local_tools = ["thinking", "inspect_slide", "finalize", "write_file", "edit_file"]
    tools_dict = {name: _tool_schema(name) for name in local_tools}

    agent = Agent.__new__(Agent)
    agent.name = "Design"
    agent.role_config = RoleConfig(
        system={"en": "test"},
        instruction="test",
        use_model="design_agent",
        toolset=ToolSet(
            include_tool_servers=["local"],
            # Deliberately re-lists tools already covered by the `local`
            # server sweep, mirroring deeppresenter/roles/Design.yaml.
            include_tools=["thinking", "inspect_slide", "finalize"],
        ),
    )
    agent.agent_env = type(
        "FakeEnv",
        (),
        {
            "_server_tools": {"local": local_tools},
            "_tools_dict": tools_dict,
        },
    )()

    agent._setup_toolset()

    tool_names = [tool["function"]["name"] for tool in agent.tools]
    assert len(tool_names) == len(set(tool_names)), (
        f"duplicate tool schemas in self.tools: {tool_names}"
    )
    assert set(tool_names) == set(local_tools)



def test_authoring_skill_preflight_reports_fixed_canvas_contract(tmp_path: Path) -> None:
    """The authoring skill catches the smoke-run box-model failure before render."""
    from deeppresenter.slidex.authoring import (
        AUTHORING_SKILL_VERSION,
        authoring_skill_hash,
        preflight_html,
    )

    invalid = tmp_path / "invalid.html"
    invalid.write_text(
        """<!doctype html><html><body style="width:1280px;height:720px;padding:24px">
        <p data-slidex-id="duplicate">One</p><p data-slidex-id="duplicate">Two</p>
        <h1>Missing ID</h1></body></html>""",
        encoding="utf-8",
    )
    report = preflight_html(invalid)
    assert report["skill_version"] == AUTHORING_SKILL_VERSION
    assert report["skill_hash"] == authoring_skill_hash()
    assert report["ok"] is False
    assert {finding["code"] for finding in report["findings"]} >= {
        "fixed_canvas_padding",
        "duplicate_stable_id",
        "missing_stable_id",
    }
    margin = tmp_path / "margin.html"
    margin.write_text(
        '<html><body style="width:1280px;height:720px;margin:8px"><p data-slidex-id="body">Body</p></body></html>',
        encoding="utf-8",
    )
    assert "fixed_canvas_margin" in {item["code"] for item in preflight_html(margin)["findings"]}

    valid = tmp_path / "valid.html"
    valid.write_text(
        """<!doctype html><html><body style="width:1280px;height:720px;box-sizing:border-box">
        <main class="slide-content"><h1 data-slidex-id="title">Title</h1>
        <p data-slidex-id="body">Body</p></main></body></html>""",
        encoding="utf-8",
    )
    valid_report = preflight_html(valid)
    assert valid_report["ok"] is True
    assert {finding["code"] for finding in valid_report["findings"]} <= {
        "missing_border_box"
    }


@pytest.mark.asyncio
async def test_e2e_repair_clones_initial_workspace_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generic/hybrid arms must resume the byte-identical initial workspace."""
    from deeppresenter.eval.e2e import E2ECaseExecutor

    case = EvaluationCase(
        case_id="task::seed0",
        parent_deck_id="task",
        source_id="source",
        split=Split.SEALED_TEST,
        input_uri="input.md",
        cluster_id="task",
        content_sha256="a" * 64,
    )
    initial = tmp_path / "initial"
    (initial / "slides").mkdir(parents=True)
    (initial / "slides" / "slide_01.html").write_text("initial", encoding="utf-8")
    pptx = initial / "initial.pptx"
    pptx.write_bytes(b"pptx")
    executor = E2ECaseExecutor("config.yaml", tmp_path, tmp_path / "workspaces")
    executor._initial_workspaces[(case.case_id, 0)] = initial
    executor._initial_pptx[(case.case_id, 0)] = pptx
    observed: dict[str, object] = {}

    async def fake_run_arm(self, *args, **kwargs):
        workspace = kwargs["workspace"]
        observed["repair_only"] = kwargs["repair_only"]
        observed["max_repairs"] = kwargs["max_repairs"]
        observed["model_budget"] = kwargs["model_budget"]
        observed["initial_html"] = (workspace / "slides" / "slide_01.html").read_text()
        history = workspace / ".history" / "slidex"
        history.mkdir(parents=True)
        (history / "task_outcome.json").write_text('{"outline_checks":{},"required_content":{},"user_constraints":{}}')
        (history / "grounding_report.json").write_text('{"supported_rate":1}')
        (history / "deck_report.json").write_text('{"hard_failures":0}')
        (history / "export_manifest.json").write_text('{"status":"pptx_render_validated"}')
        result = workspace / "result.pptx"
        result.write_bytes(b"result")
        return result, workspace, 1.0

    monkeypatch.setattr(E2ECaseExecutor, "_run_arm", fake_run_arm)
    run = EvaluationRun(
        run_id="generic",
        suite=Suite.E2E,
        arm=Arm.GENERIC,
        seed=0,
        manifest_hash="m",
        config_hash="c",
        git_commit="g",
        initial_artifact_id="shared",
        max_repairs=2,
        model_budget=9,
    )
    result = await executor.repair(case, run)
    assert result.outcome is Outcome.PASS
    assert result.artifact_lineage == ["shared"]
    assert observed == {
        "repair_only": True,
        "max_repairs": 2,
        "model_budget": 9,
        "initial_html": "initial",
    }


@pytest.mark.asyncio
async def test_agent_splits_excess_tool_calls_instead_of_rejecting_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model batch over the hard limit must make forward progress."""
    from deeppresenter.agents.agent import Agent
    from deeppresenter.utils.typings import ChatMessage, Role
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as ToolCall,
    )

    agent = object.__new__(Agent)
    agent.name = "Test"
    agent.workspace = tmp_path
    agent.context_warning = 0
    agent.context_length = 0
    agent.context_window = 1000
    agent.chat_history = []
    agent.error_history = []
    agent.model = "test"
    agent.log_message = lambda _: None

    class Env:
        async def tool_execute(self, call):
            return ChatMessage(role=Role.TOOL, content=call.function.name, tool_call_id=call.id)

    agent.agent_env = Env()
    calls = [
        ToolCall(id=str(index), type="function", function={"name": "write_file", "arguments": "{}"})
        for index in range(9)
    ]
    observations = await agent.execute(calls)
    assert isinstance(observations, list)
    assert len(observations) == 9  # all calls run in bounded batches
    assert [item.tool_call_id for item in observations] == [str(index) for index in range(9)]


def test_geometry_error_parser_retains_structured_payload() -> None:
    from deeppresenter.utils.webview import Html2PptxError

    error = Html2PptxError(
        'overflow\nSLIDEX_GEOMETRY={"code":"collapsed_flow_margin","body":{"flowTail":[]}}',
        None,  # command metadata is irrelevant to parser behavior
    )
    assert error.geometry == {"code": "collapsed_flow_margin", "body": {"flowTail": []}}


@pytest.mark.asyncio
async def test_repeated_edit_miss_becomes_patch_instruction(tmp_path: Path) -> None:
    from deeppresenter.agents.agent import Agent
    from deeppresenter.utils.typings import ChatMessage, Role
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as ToolCall,
    )

    agent = object.__new__(Agent)
    agent._edit_failures = {}
    agent._patch_html_failures = {}
    call = ToolCall(id="edit", type="function", function={"name": "edit_file", "arguments": '{"path":"slide.html"}'})
    first = agent._edit_failure_observation(
        call, ChatMessage(role=Role.TOOL, content="Expected exactly one match in slide.html, found 0", tool_call_id="edit", is_error=True)
    )
    second = agent._edit_failure_observation(
        call, ChatMessage(role=Role.TOOL, content="Expected exactly one match in slide.html, found 0", tool_call_id="edit", is_error=True)
    )
    assert first.is_error is True
    assert second.is_error is False and "EDIT_FILE_TERMINAL" in second.text


def test_repeated_patch_html_failure_opens_circuit() -> None:
    """A repeated selector miss must be blocked before it burns more repair turns."""
    from deeppresenter.agents.agent import Agent
    from deeppresenter.utils.typings import ChatMessage, Role

    agent = object.__new__(Agent)
    agent._edit_failures = {}
    agent._patch_html_failures = {}
    call = _tool_call(
        "patch_html",
        {"path": "slides/slide_01.html", "selector": "li", "operation": "set_style", "name": "margin-bottom", "value": "8px"},
    )
    first = agent._edit_failure_observation(
        call, ChatMessage(role=Role.TOOL, content="selector must match exactly one element, found 4", tool_call_id=call.id, is_error=True)
    )
    second = agent._edit_failure_observation(
        call, ChatMessage(role=Role.TOOL, content="selector must match exactly one element, found 4", tool_call_id=call.id, is_error=True)
    )
    assert first.is_error is True
    assert second.is_error is False and "PATCH_HTML_CIRCUIT_OPEN" in second.text
    assert agent._circuit_open_observation(call) is not None


def test_repair_gate_requires_fresh_inspection_after_source_patch() -> None:
    """Repair-only mutation chains require a separate rendered inspection turn."""
    from deeppresenter.agents.design import _repair_inspection_gate

    patch = [_tool_call("patch_slide_element", {"path": "slides/slide_01.html", "element_id": "card-1", "styles": {"font-size": "20px"}})]
    assert _repair_inspection_gate(patch, requires_inspection=False, enabled=True) is None
    blocked = _repair_inspection_gate(patch, requires_inspection=True, enabled=True)
    assert blocked is not None and "REPAIR_INSPECTION_REQUIRED" in blocked.text
    mixed = patch + [_tool_call("inspect_slide", {"html_file": "slides/slide_01.html"})]
    assert _repair_inspection_gate(mixed, requires_inspection=False, enabled=True) is not None
