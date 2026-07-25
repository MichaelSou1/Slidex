"""Pure offline summarization; this module never imports model clients."""

from __future__ import annotations

from pathlib import Path

from .io import write_immutable
from .metrics import summarize
from .models import BenchmarkManifest, EvaluationResult


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
