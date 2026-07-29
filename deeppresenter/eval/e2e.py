"""Phase 13 E2E three-arm paired execution (13.6/13.7).

Turns the frozen 120-task corpus (:mod:`deeppresenter.eval.tasks`) into a
``BenchmarkManifest`` and executes the ``no_critic`` / ``generic`` / ``hybrid``
arms of :class:`deeppresenter.main.AgentLoop` for one (task, seed) pair,
mapping the loop's own ``task_outcome.json`` / ``grounding_report.json`` /
``deck_report.json`` / ``export_manifest.json`` artifacts into a
:class:`CaseResult`. All three arms share the same generation model, task
brief, and seed so any endpoint difference reflects the critic mechanism
(13.7), not generation randomness.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from deeppresenter.slidex.export import extract_pptx_structure
from deeppresenter.slidex.models import FinalArtifactStatus
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.typings import ConvertType, InputRequest

from collections.abc import Awaitable, Callable

from .io import content_hash, file_hash, write_immutable
from .models import (
    Arm,
    BenchmarkManifest,
    CaseResult,
    DefectLabel,
    EvaluationCase,
    EvaluationResult,
    EvaluationRun,
    Outcome,
    PreparationRecord,
    Split,
)
from .runner import validate_paired_runs

#: Recorded when initial generation failed before a shared artifact existed.
#: Arms are still recorded so a failed initial generation cannot silently drop
#: paired rows; repair-capable executors must fail them as unavailable rather
#: than regenerate independent decks.
FIRST_ROUND_GENERATION_FAILED = "first-round-generation-failed"

async def run_paired_task(
    case: EvaluationCase,
    seed: int,
    runs: list[EvaluationRun],
    generate: Callable[[EvaluationCase, int], Awaitable[str]],
    repair: Callable[[EvaluationCase, EvaluationRun], Awaitable[CaseResult]],
) -> list[CaseResult]:
    """Generate a shared initial artifact once, then execute each paired arm.

    The executor owns a materialized immutable initial workspace. This helper
    records its ID on every arm and never performs a second no-critic
    generation itself. Each repair callable must consume that snapshot rather
    than silently creating an unrelated initial deck.
    """
    if not runs:
        raise ValueError("run_paired_task requires at least one run")
    if len({run.seed for run in runs}) != 1 or runs[0].seed != seed:
        raise ValueError("all paired runs must use the requested seed")
    try:
        initial_artifact_id = await generate(case, seed)
    except Exception:
        initial_artifact_id = FIRST_ROUND_GENERATION_FAILED
    for run in runs:
        run.initial_artifact_id = initial_artifact_id
    validate_paired_runs(runs)
    results: list[CaseResult] = []
    for run in runs:
        try:
            results.append(await repair(case, run))
        except Exception as exc:
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    outcome=Outcome.ERROR,
                    artifact_lineage=[initial_artifact_id],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


_ARM_CRITIC_MODE = {
    Arm.NO_CRITIC: "no_critic",
    Arm.GENERIC: "generic",
    Arm.HYBRID: "hybrid",
}


def build_e2e_manifest(
    corpus_path: Path, *, split: str | None = None, seeds: list[int] | None = None
) -> BenchmarkManifest:
    """Convert the frozen 120-task corpus into a :class:`BenchmarkManifest`.

    Each ``(task, seed)`` pair becomes one deterministic ``EvaluationCase``
    with ``case_id = f"{task_id}::seed{seed}"`` so that per-seed ``CaseResult``
    records never collide under the same manifest key (13.12: seeds are
    repeated measures, not independent cases). ``parent_deck_id``/
    ``cluster_id`` are kept as the bare ``task_id`` so ``summarize()``'s
    per-deck clustered bootstrap correctly treats all seeds of one task as
    one cluster. E2E cases carry no defect labels (there is no injected
    ground-truth defect), so ``labels`` is left empty and downstream E2E
    metrics rely on ``CaseResult``'s task/grounding/export fields instead.
    """
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tasks = corpus["tasks"]
    if split is not None:
        tasks = [task for task in tasks if task["split"] == split]
    seed_list = seeds if seeds else [0]
    cases: list[EvaluationCase] = []
    for task in tasks:
        source = task["source"]
        for seed in seed_list:
            cases.append(
                EvaluationCase(
                    case_id=f"{task['task_id']}::seed{seed}",
                    parent_deck_id=task["task_id"],
                    source_id=source["source_id"],
                    split=Split.PILOT if task["split"] == "pilot" else Split.SEALED_TEST,
                    input_uri=task["normalized_path"],
                    labels=[],
                    task_brief=task["brief"],
                    cluster_id=task["task_id"],
                    content_sha256=source["sha256"],
                    preparation_record=PreparationRecord(
                        original_sha256=source["sha256"],
                        normalized_sha256=source["sha256"],
                        source_url=source["url"],
                    ),
                    metadata={
                        "task_type": task["task_type"],
                        "corpus_hash": corpus["corpus_hash"],
                        "task_id": task["task_id"],
                        "seed": seed,
                    },
                )
            )
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    manifest = BenchmarkManifest(
        benchmark_id="slidex-phase13-e2e",
        revision=corpus["corpus_hash"][:16],
        created_at=now,
        frozen_at=now,
        sources=[],
        cases=sorted(cases, key=lambda item: item.case_id),
        preparation={
            "corpus_hash": corpus["corpus_hash"],
            "corpus_path": str(corpus_path),
            "human_scoring": "skipped_by_user",
            "seeds": seed_list,
        },
    )
    manifest.manifest_hash = content_hash(
        manifest.model_dump(exclude={"manifest_hash", "created_at", "frozen_at"}, mode="json")
    )
    return manifest


def _instruction_markdown(brief: dict[str, Any], source_text: str, title: str) -> str:
    facts = "\n".join(f"- {fact['text']}" for fact in brief["required_facts"])
    sections = ", ".join(brief["required_sections"])
    style = "; ".join(brief.get("style_constraints", []))
    forbidden = "; ".join(brief.get("forbidden_claims", []))
    return (
        f"# {title}\n\n"
        f"Audience: {brief['audience']}\n"
        f"Purpose: {brief['purpose']}\n"
        f"Required sections (must appear as slide titles or headings): {sections}\n"
        f"Style constraints: {style}\n"
        f"Do not fabricate: {forbidden}\n\n"
        "## Source material\n\n"
        f"{source_text}\n\n"
        "## Facts that must be retained in the deck\n\n"
        f"{facts}\n"
    )


def build_input_request(case: EvaluationCase, corpus_dir: Path) -> InputRequest:
    """Turn one E2E task's frozen brief + source into a generation request."""
    brief = case.task_brief.model_dump(mode="json")
    normalized_path = corpus_dir / case.input_uri
    source_text = normalized_path.read_text(encoding="utf-8") if normalized_path.exists() else ""
    title = normalized_path.stem
    markdown = _instruction_markdown(brief, source_text, title)
    brief_dir = corpus_dir / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    brief_path = brief_dir / f"{case.case_id}.md"
    brief_path.write_text(markdown, encoding="utf-8")
    lower, upper = brief["page_count"]
    required_terms = [fact["text"][:60] for fact in brief["required_facts"]]
    return InputRequest(
        instruction=(
            f"Create a {lower}-{upper} page presentation for: {brief['purpose']}. "
            f"Use the attached source material and required sections: "
            f"{', '.join(brief['required_sections'])}."
        ),
        attachments=[str(brief_path)],
        num_pages=str(lower) if lower == upper else f"{lower}-{upper}",
        convert_type=ConvertType.SLIDEX,
        extra_info={"required_terms": required_terms},
    )


def case_result_from_workspace(
    case: EvaluationCase, workspace: Path, *, final_pptx: Path, latency_seconds: float
) -> CaseResult:
    """Read the AgentLoop's own task/grounding/export/deck artifacts into a CaseResult."""
    history = workspace / ".history" / "slidex"
    task_outcome = json.loads((history / "task_outcome.json").read_text(encoding="utf-8"))
    grounding_report = json.loads((history / "grounding_report.json").read_text(encoding="utf-8"))
    deck_report = json.loads((history / "deck_report.json").read_text(encoding="utf-8"))
    export_manifest = json.loads(
        (history / "export_manifest.json").read_text(encoding="utf-8")
    )
    outline_checks = task_outcome.get("outline_checks", {})
    required_content = task_outcome.get("required_content", {})
    user_constraints = task_outcome.get("user_constraints", {})
    section_coverage = (
        sum(outline_checks.values()) / len(outline_checks) if outline_checks else 1.0
    )
    facts_retained = all(required_content.values()) if required_content else True
    constraints_passed = all(user_constraints.values()) if user_constraints else True
    export_passed = export_manifest.get("status") == FinalArtifactStatus.PPTX_RENDER_VALIDATED.value
    render_fidelity_passed = export_passed
    severe_defects = int(deck_report.get("hard_failures", 0))
    return CaseResult(
        case_id=case.case_id,
        outcome=Outcome.PASS if export_passed else Outcome.FAIL,
        artifact_lineage=[str(final_pptx)],
        export_passed=export_passed,
        render_fidelity_passed=render_fidelity_passed,
        severe_defects=severe_defects,
        required_facts_retained=facts_retained,
        task_constraints_passed=constraints_passed,
        section_coverage=section_coverage,
        page_count_passed=user_constraints.get("page_count_within_range", True),
        grounding_score=grounding_report.get("supported_rate"),
        latency_seconds=latency_seconds,
        raw_output={
            "task_outcome": task_outcome,
            "export_status": export_manifest.get("status"),
            "deck_hard_failures": severe_defects,
        },
    )


class E2ECaseExecutor:
    """Materialize one initial workspace and fork it for the paired arms.

    The initial generation is run once. Its HTML/PPTX workspace is copied
    byte-for-byte to each arm workspace before any arm-specific work starts,
    and its content hash is the paired ``initial_artifact_id``. The repair
    continuation skips Research and first-draft generation, so no arm can be
    reported as paired after independently generating a different deck.
    """

    def __init__(self, config_path: str, corpus_dir: Path, workspace_root: Path) -> None:
        self.config_path = config_path
        self.corpus_dir = corpus_dir
        self.workspace_root = workspace_root
        self._initial_workspaces: dict[tuple[str, int], Path] = {}
        self._initial_pptx: dict[tuple[str, int], Path | None] = {}
        self._initial_errors: dict[tuple[str, int], str | None] = {}

    async def _run_arm(
        self,
        case: EvaluationCase,
        seed: int,
        arm: Arm,
        critic_arm: str,
        *,
        stage: str,
        max_repairs: int | None = None,
        model_budget: int | None = None,
        repair_only: bool = False,
        workspace: Path | None = None,
    ) -> tuple[Path | None, Path, float]:
        from deeppresenter.main import AgentLoop

        config = DeepPresenterConfig.load_from_file(self.config_path)
        # Initial generation and repair continuation use isolated workspace
        # names. Repair passes receive an explicit copied initial workspace.
        session_id = f"e2e-{stage}-{arm.value}-{seed}-{case.case_id}"[:63]
        workspace = workspace or self.workspace_root / session_id
        loop = AgentLoop(config, session_id=session_id, workspace=workspace)
        request = build_input_request(case, self.corpus_dir)
        started = time.perf_counter()
        final_pptx: Path | None = None
        async for message in loop.run(
            request,
            critic_arm=critic_arm,
            max_repairs=max_repairs,
            model_budget=model_budget,
            repair_only=repair_only,
        ):
            if isinstance(message, (str, Path)):
                final_pptx = Path(message)
        return final_pptx, workspace, time.perf_counter() - started

    async def generate_first_round(self, case: EvaluationCase, seed: int) -> str:
        """Generate once and retain the initial workspace even if its export gate fails.

        A failed no-critic export is an E2E measurement, not a reason to throw
        away the shared HTML/IR snapshot needed by the two repair arms.
        """
        workspace = self.workspace_root / f"e2e-initial-{Arm.NO_CRITIC.value}-{seed}-{case.case_id}"[:100]
        final_pptx: Path | None = None
        initial_error: str | None = None
        try:
            final_pptx, workspace, _ = await self._run_arm(
                case, seed, Arm.NO_CRITIC, "no_critic", stage="initial", workspace=workspace
            )
        except Exception as exc:
            initial_error = f"{type(exc).__name__}: {exc}"
        snapshot_files = [
            path for path in sorted(workspace.rglob("*"))
            if path.is_file() and (
                path.relative_to(workspace).parts[0] in {"slides", "attachments"}
                or path.name in {"intermediate_output.json", "pico_technology_presentation.md"}
            )
        ]
        if not snapshot_files:
            raise RuntimeError(initial_error or "initial generation produced no shareable workspace")
        initial_id = content_hash(
            {str(path.relative_to(workspace)): file_hash(path) for path in snapshot_files}
        )
        self._initial_workspaces[(case.case_id, seed)] = workspace
        self._initial_pptx[(case.case_id, seed)] = final_pptx
        self._initial_errors[(case.case_id, seed)] = initial_error
        return initial_id

    async def repair(self, case: EvaluationCase, run: EvaluationRun) -> CaseResult:
        key = (case.case_id, run.seed)
        initial_workspace = self._initial_workspaces.get(key)
        initial_pptx = self._initial_pptx.get(key)
        if initial_workspace is None:
            raise RuntimeError("paired repair requires generate_first_round first")
        if run.arm is Arm.NO_CRITIC:
            initial_error = self._initial_errors.get(key)
            if initial_pptx is None:
                return CaseResult(
                    case_id=case.case_id,
                    outcome=Outcome.FAIL,
                    artifact_lineage=[run.initial_artifact_id or ""],
                    error=initial_error or "initial no_critic export did not produce PPTX",
                )
            return case_result_from_workspace(
                case, initial_workspace, final_pptx=initial_pptx, latency_seconds=0
            ).model_copy(update={"artifact_lineage": [run.initial_artifact_id or ""]})

        # Clone the byte-identical initial workspace, then enter the
        # repair-only continuation. Research and initial drafting are skipped.
        arm_workspace = self.workspace_root / f"e2e-repair-{run.arm.value}-{run.seed}-{case.case_id}"[:100]
        if arm_workspace.exists():
            shutil.rmtree(arm_workspace)
        shutil.copytree(initial_workspace, arm_workspace)
        started = time.perf_counter()
        try:
            final_pptx, workspace, elapsed = await self._run_arm(
                case,
                run.seed,
                run.arm,
                _ARM_CRITIC_MODE[run.arm],
                stage="repair",
                max_repairs=run.max_repairs,
                model_budget=run.model_budget,
                repair_only=True,
                workspace=arm_workspace,
            )
        except Exception as exc:
            return CaseResult(
                case_id=case.case_id,
                outcome=Outcome.ERROR,
                artifact_lineage=[run.initial_artifact_id or ""],
                error=f"{type(exc).__name__}: {exc}",
                latency_seconds=time.perf_counter() - started,
            )
        if final_pptx is None:
            return CaseResult(
                case_id=case.case_id,
                outcome=Outcome.ERROR,
                artifact_lineage=[run.initial_artifact_id or ""],
                error="repair continuation completed without a final PPTX path",
                latency_seconds=elapsed,
            )
        return case_result_from_workspace(
            case, workspace, final_pptx=final_pptx, latency_seconds=elapsed
        ).model_copy(update={"artifact_lineage": [run.initial_artifact_id or ""]})

    async def __call__(self, case: EvaluationCase, run: EvaluationRun) -> CaseResult:
        """Single-arm convenience path (bypasses the run_paired_task fork)."""
        return await self.repair(case, run)


def _e2e_run_id(manifest_hash: str, arm: Arm) -> str:
    return content_hash({"manifest": manifest_hash, "suite": "e2e", "arm": arm.value})[:24]


def run_e2e_manifest(
    manifest: BenchmarkManifest,
    *,
    config_path: str,
    corpus_dir: Path,
    workspace_root: Path,
    output_dir: Path,
    max_repairs: int = 3,
    model_budget: int = 3,
    concurrency: int = 1,
    rerun_failed: bool = False,
    git_commit: str = "unknown",
    environment: dict[str, str] | None = None,
) -> dict[Arm, EvaluationResult]:
    """Drive the full 13.7 three-arm paired E2E benchmark for one manifest.

    ``manifest.cases`` must already be the seed-expanded cases produced by
    :func:`build_e2e_manifest` (one ``EvaluationCase`` per ``(task, seed)``
    pair, ``case_id = f"{task_id}::seed{seed}"``, sharing ``parent_deck_id``
    across seeds of the same task). For every case this calls
    :func:`run_paired_task` exactly once, which in turn calls
    ``E2ECaseExecutor.generate_first_round`` exactly once and pins all three
    arms (``no_critic``/``generic``/``hybrid``) to that same first-round
    artifact id before running ``repair`` once per arm (13.7: "only generate
    once per task/seed; fork three arms from the same first-round artifact").
    Results are grouped and persisted per arm so the output is directly
    consumable by ``summarize`` / ``summarize-paired``.

    Idempotent: a case is skipped once all three arms already have a
    non-failed persisted ``CaseResult``; failed/error/missing outcomes are
    only retried when ``rerun_failed`` is set, mirroring
    :class:`EvaluationRunner`.
    """
    return asyncio.run(
        _run_e2e_manifest_async(
            manifest,
            config_path=config_path,
            corpus_dir=corpus_dir,
            workspace_root=workspace_root,
            output_dir=output_dir,
            max_repairs=max_repairs,
            model_budget=model_budget,
            concurrency=concurrency,
            rerun_failed=rerun_failed,
            git_commit=git_commit,
            environment=environment or {},
        )
    )


_FAILED_OUTCOMES = {Outcome.ERROR, Outcome.TIMEOUT, Outcome.MISSING}


async def _run_e2e_manifest_async(
    manifest: BenchmarkManifest,
    *,
    config_path: str,
    corpus_dir: Path,
    workspace_root: Path,
    output_dir: Path,
    max_repairs: int,
    model_budget: int,
    concurrency: int,
    rerun_failed: bool,
    git_commit: str,
    environment: dict[str, str],
) -> dict[Arm, EvaluationResult]:
    arms = (Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID)
    run_ids = {arm: _e2e_run_id(manifest.manifest_hash or "unfrozen", arm) for arm in arms}
    run_dirs = {arm: output_dir / run_ids[arm] for arm in arms}
    for arm in arms:
        (run_dirs[arm] / "cases").mkdir(parents=True, exist_ok=True)

    def _case_result_path(arm: Arm, case_id: str) -> Path:
        return run_dirs[arm] / "cases" / f"{case_id}.json"

    def _load_existing(arm: Arm, case_id: str) -> CaseResult | None:
        path = _case_result_path(arm, case_id)
        if not path.exists():
            return None
        result = CaseResult.model_validate_json(path.read_text(encoding="utf-8"))
        if rerun_failed and result.outcome in _FAILED_OUTCOMES:
            return None
        return result

    def _load_or_create_run(arm: Arm) -> EvaluationRun:
        run_path = run_dirs[arm] / "run.json"
        if run_path.exists():
            # Resuming (idempotent re-invocation) must reuse the frozen run
            # record verbatim; a fresh started_at would make the immutable
            # run.json write non-idempotent, mirroring eval/cli.py run_command.
            return EvaluationRun.model_validate_json(run_path.read_text(encoding="utf-8"))
        return EvaluationRun(
            run_id=run_ids[arm],
            suite="e2e",
            arm=arm,
            seed=-1,
            manifest_hash=manifest.manifest_hash or "unfrozen",
            config_hash="unconfigured",
            git_commit=git_commit,
            environment=environment,
            max_repairs=max_repairs,
            model_budget=model_budget,
            metadata={"seeds": sorted({int(c.metadata.get("seed", 0)) for c in manifest.cases})},
        )

    run_records = {arm: _load_or_create_run(arm) for arm in arms}
    executor = E2ECaseExecutor(config_path, corpus_dir, workspace_root)
    semaphore = asyncio.Semaphore(concurrency)
    all_results: dict[Arm, list[CaseResult]] = {arm: [] for arm in arms}
    lock = asyncio.Lock()

    async def process_case(case: EvaluationCase) -> None:
        seed = int(case.metadata.get("seed", 0))
        existing = {arm: _load_existing(arm, case.case_id) for arm in arms}
        if all(existing[arm] is not None for arm in arms):
            async with lock:
                for arm in arms:
                    all_results[arm].append(existing[arm])
            return

        async def generate(case: EvaluationCase, seed: int) -> str:
            return await executor.generate_first_round(case, seed)

        runs = [
            EvaluationRun(
                run_id=f"{run_ids[arm]}-{case.case_id}",
                suite="e2e",
                arm=arm,
                seed=seed,
                manifest_hash=manifest.manifest_hash or "unfrozen",
                config_hash="unconfigured",
                git_commit=git_commit,
                environment=environment,
                max_repairs=max_repairs,
                model_budget=model_budget,
                metadata={"task_id": case.parent_deck_id},
            )
            for arm in arms
        ]
        async with semaphore:
            # run_paired_task never raises for generation/repair failures (it
            # converts them into per-run CaseResult.error, tagging cascade
            # failures with CASCADE_GENERATION_FAILURE_PREFIX); an exception
            # here means a genuine bug in the harness itself, so let it
            # surface via asyncio.gather rather than mislabeling it as three
            # independent arm failures.
            results = await run_paired_task(case, seed, runs, generate, executor.repair)
        async with lock:
            for arm, result in zip(arms, results, strict=True):
                write_immutable(_case_result_path(arm, case.case_id), result)
                all_results[arm].append(result)

    await asyncio.gather(*(process_case(case) for case in manifest.cases))

    final: dict[Arm, EvaluationResult] = {}
    for arm in arms:
        run_record = run_records[arm]
        write_immutable(run_dirs[arm] / "run.json", run_record)
        sorted_results = sorted(all_results[arm], key=lambda item: item.case_id)
        result = EvaluationResult(run=run_record, results=sorted_results)
        result.immutable_hash = content_hash(
            result.model_dump(exclude={"immutable_hash"}, mode="json")
        )
        result_path = run_dirs[arm] / "result.json"
        if result_path.exists():
            # Re-running after new cases were added/retried legitimately
            # changes result.json; only cases/*.json need to stay append-only.
            result_path.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        else:
            write_immutable(result_path, result)
        final[arm] = result
    return final
