"""Pure offline summarization; this module never imports model clients."""

from __future__ import annotations

from pathlib import Path

from .io import write_immutable
from .metrics import paired_effect, summarize
from .models import Arm, BenchmarkManifest, EvaluationResult


def summarize_run(
    manifest_path: Path, result_path: Path, output_path: Path
) -> dict[str, object]:
    manifest = BenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    result = EvaluationResult.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "1.0",
        "manifest_hash": manifest.manifest_hash,
        "run_hash": result.immutable_hash,
        "suite": result.run.suite.value,
        "arm": result.run.arm.value,
        "metrics": summarize(manifest.cases, result.results),
        "filters": {"included": "all", "excluded_case_ids": []},
        "limitations": [
            "Single-expert ratings do not establish inter-rater reliability."
        ],
    }
    write_immutable(output_path, report)
    return report


def summarize_paired_runs(
    manifest_path: Path,
    result_paths: list[Path],
    output_path: Path,
) -> dict[str, object]:
    """Compare immutable E2E arms on matched task/seed records."""
    manifest = BenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    results = [
        EvaluationResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in result_paths
    ]
    by_arm = {result.run.arm: result for result in results}
    required = {Arm.NO_CRITIC, Arm.GENERIC, Arm.HYBRID}
    if set(by_arm) != required:
        raise ValueError("paired summary requires no_critic, generic, and hybrid runs")
    case_ids = [case.case_id for case in manifest.cases]

    def endpoint(result: EvaluationResult) -> list[bool]:
        records = {item.case_id: item for item in result.results}
        if set(records) != set(case_ids):
            raise ValueError("paired runs do not contain identical cases")
        return [
            bool(
                records[case_id].severe_defects == 0
                and records[case_id].export_passed
                and records[case_id].render_fidelity_passed
            )
            for case_id in case_ids
        ]

    hybrid = endpoint(by_arm[Arm.HYBRID])
    report: dict[str, object] = {
        "schema_version": "1.0",
        "manifest_hash": manifest.manifest_hash,
        "run_hashes": {
            arm.value: result.immutable_hash for arm, result in by_arm.items()
        },
        "arms": {
            arm.value: summarize(manifest.cases, result.results)
            for arm, result in by_arm.items()
        },
        "paired_effects": {
            "hybrid_vs_no_critic": paired_effect(
                endpoint(by_arm[Arm.NO_CRITIC]), hybrid
            ),
            "hybrid_vs_generic": paired_effect(endpoint(by_arm[Arm.GENERIC]), hybrid),
        },
    }
    write_immutable(output_path, report)
    return report
