"""Frozen confirmatory evaluation registration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .io import capture_environment, content_hash, git_commit, write_immutable
from .models import Preregistration


def freeze_preregistration(
    path: Path, config_hashes: dict[str, str]
) -> Preregistration:
    registration = Preregistration(
        research_questions=[
            "Does the critic detect presentation defects accurately?",
            "Does frozen hybrid routing outperform whole-rubric VLM inspection?",
            "Does critic-in-the-loop repair improve final exported deck quality?",
        ],
        primary_endpoints=[
            "intrinsic_macro_balanced_accuracy",
            "e2e_severe_defect_free_and_export_fidelity_pass_rate",
        ],
        confirmatory_metrics=["macro_balanced_accuracy", "e2e_primary_rate"],
        secondary_metrics=["repair_rate", "collateral_defect_rate", "cost", "latency"],
        exploratory_metrics=[
            "distribution_transfer",
            "natural_failure_rate",
            "model_family_difference",
        ],
        minimum_effects={
            "hybrid_vs_generic_macro_ba": 0.05,
            "hybrid_vs_no_critic_e2e": 0.05,
        },
        noninferiority_margins={"grounding": -0.02, "export_success": -0.02},
        frozen_config_hashes=config_hashes,
        git_commit=git_commit(),
        environment=capture_environment(),
        frozen_at=datetime.now(UTC),
    )
    registration.preregistration_hash = content_hash(
        registration.model_dump(exclude={"preregistration_hash"}, mode="json")
    )
    write_immutable(path, registration)
    return registration
