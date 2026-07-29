"""Offline metrics and clustered uncertainty for immutable results."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Iterable

from .models import CaseResult, EvaluationCase, Outcome


def _safe(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(truth and pred for truth, pred in zip(y_true, y_pred, strict=True))
    tn = sum(not truth and not pred for truth, pred in zip(y_true, y_pred, strict=True))
    fp = sum(not truth and pred for truth, pred in zip(y_true, y_pred, strict=True))
    fn = sum(truth and not pred for truth, pred in zip(y_true, y_pred, strict=True))
    recall, specificity = _safe(tp, tp + fn), _safe(tn, tn + fp)
    precision = _safe(tp, tp + fp)
    return {
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": _safe(2 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def bbox_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    area = max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - area
    return _safe(area, union)


def calibration(
    confidences: list[float], correctness: list[bool], bins: int = 10
) -> dict[str, float]:
    brier = _safe(
        sum(
            (confidence - int(correct)) ** 2
            for confidence, correct in zip(confidences, correctness, strict=True)
        ),
        len(confidences),
    )
    ece = 0.0
    for index in range(bins):
        members = [
            i
            for i, value in enumerate(confidences)
            if index / bins <= value < (index + 1) / bins
            or (index == bins - 1 and value == 1)
        ]
        if members:
            ece += (
                len(members)
                / len(confidences)
                * abs(
                    sum(confidences[i] for i in members) / len(members)
                    - sum(correctness[i] for i in members) / len(members)
                )
            )
    return {"brier_score": brier, "ece": ece}


def clustered_bootstrap(
    values: dict[str, list[float]], seed: int = 0, samples: int = 2000
) -> tuple[float, float]:
    rng = random.Random(seed)
    clusters = sorted(values)
    if not clusters:
        return (0.0, 0.0)
    estimates = []
    for _ in range(samples):
        chosen = [rng.choice(clusters) for _ in clusters]
        observations = [value for cluster in chosen for value in values[cluster]]
        estimates.append(sum(observations) / len(observations))
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[
        min(samples - 1, int(samples * 0.975))
    ]


def holm_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return adjusted


def weighted_kappa(first: list[int], second: list[int]) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    levels = sorted(set(first) | set(second))
    maximum = max(1, len(levels) - 1)
    index = {value: i for i, value in enumerate(levels)}
    observed = sum(
        ((index[a] - index[b]) / maximum) ** 2
        for a, b in zip(first, second, strict=True)
    ) / len(first)
    ca, cb = Counter(first), Counter(second)
    expected = sum(
        ca[a] * cb[b] / len(first) ** 2 * ((index[a] - index[b]) / maximum) ** 2
        for a in levels
        for b in levels
    )
    return 1 - _safe(observed, expected) if expected else 1.0


def mixed_effects_logistic(
    outcomes: list[bool],
    treatment: list[bool],
    clusters: list[str],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Random-intercept logistic mixed model fit by penalized quasi-likelihood.

    This estimates a fixed treatment effect on a binary endpoint while
    accounting for source-deck-level clustering via a per-cluster random
    intercept ``u_j ~ N(0, sigma^2)``, avoiding the pseudo-replication that a
    plain logistic regression would introduce when multiple cases share one
    source deck. Implemented without a statistics dependency: Laplace-style
    penalized IRLS alternates fixed-effect and random-intercept updates, then
    refits the random-intercept variance from the shrunken residuals.
    """
    if not (len(outcomes) == len(treatment) == len(clusters)):
        raise ValueError("outcomes, treatment, and clusters must be aligned")
    if not outcomes:
        raise ValueError("mixed-effects fit requires at least one observation")
    unique_clusters = sorted(set(clusters))
    cluster_index = {name: index for index, name in enumerate(unique_clusters)}
    n_clusters = len(unique_clusters)
    y = [float(value) for value in outcomes]
    x = [float(value) for value in treatment]
    z = [cluster_index[name] for name in clusters]

    beta0, beta1 = 0.0, 0.0
    random_intercepts = [0.0] * n_clusters
    sigma2 = 1.0

    def linear_predictor(index: int) -> float:
        return beta0 + beta1 * x[index] + random_intercepts[z[index]]

    for _ in range(max_iter):
        previous = (beta0, beta1, sigma2)
        # Fixed-effects + random-intercept Newton step (penalized IRLS).
        for _inner in range(25):
            grad_b0 = grad_b1 = 0.0
            hess_b0b0 = hess_b0b1 = hess_b1b1 = 0.0
            grad_u = [0.0] * n_clusters
            hess_u = [1.0 / sigma2 if sigma2 > 0 else 1e6] * n_clusters
            for index in range(len(y)):
                eta = linear_predictor(index)
                mu = 1.0 / (1.0 + pow(2.718281828459045, -eta))
                weight = max(mu * (1 - mu), 1e-6)
                residual = y[index] - mu
                grad_b0 += residual
                grad_b1 += residual * x[index]
                hess_b0b0 += weight
                hess_b0b1 += weight * x[index]
                hess_b1b1 += weight * x[index] * x[index]
                cluster = z[index]
                grad_u[cluster] += residual
                hess_u[cluster] += weight
            grad_u = [
                grad_u[j] - (random_intercepts[j] / sigma2 if sigma2 > 0 else 0)
                for j in range(n_clusters)
            ]
            determinant = hess_b0b0 * hess_b1b1 - hess_b0b1 * hess_b0b1
            if abs(determinant) < 1e-12:
                break
            delta_b0 = (grad_b0 * hess_b1b1 - grad_b1 * hess_b0b1) / determinant
            delta_b1 = (grad_b1 * hess_b0b0 - grad_b0 * hess_b0b1) / determinant
            beta0 += delta_b0
            beta1 += delta_b1
            for j in range(n_clusters):
                random_intercepts[j] += grad_u[j] / hess_u[j]
            if abs(delta_b0) < tol and abs(delta_b1) < tol:
                break
        # Update the random-intercept variance including each cluster's posterior
        # curvature (1/hess_u), matching the standard PQL variance-component update;
        # using only sum(u_j^2) would systematically shrink sigma2 toward zero.
        sigma2 = max(
            1e-6,
            sum(
                random_intercepts[j] * random_intercepts[j] + 1.0 / hess_u[j]
                for j in range(n_clusters)
            )
            / max(1, n_clusters),
        )
        if (
            abs(beta0 - previous[0]) < tol
            and abs(beta1 - previous[1]) < tol
            and abs(sigma2 - previous[2]) < tol
        ):
            break

    odds_ratio = pow(2.718281828459045, beta1)
    return {
        "fixed_intercept": beta0,
        "treatment_log_odds": beta1,
        "treatment_odds_ratio": odds_ratio,
        "random_intercept_variance": sigma2,
        "n_clusters": float(n_clusters),
        "n_observations": float(len(y)),
    }


def paired_effect(
    control: list[bool], treatment: list[bool], *, seed: int = 0, samples: int = 2000
) -> dict[str, float | list[float]]:
    """Report paired absolute/relative effects with a deterministic bootstrap CI."""
    if len(control) != len(treatment) or not control:
        raise ValueError("paired effects require equally sized non-empty samples")
    control_rate = sum(control) / len(control)
    treatment_rate = sum(treatment) / len(treatment)
    absolute = treatment_rate - control_rate
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(control)) for _ in control]
        estimates.append(
            sum(int(treatment[index]) - int(control[index]) for index in indices)
            / len(indices)
        )
    estimates.sort()
    low = estimates[int(samples * 0.025)]
    high = estimates[min(samples - 1, int(samples * 0.975))]
    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_difference": absolute,
        "relative_difference": _safe(absolute, control_rate),
        "risk_ratio": _safe(treatment_rate, control_rate),
        "bootstrap_95_ci": [low, high],
    }


def repair_metrics(results: Iterable[CaseResult]) -> dict[str, float]:
    records = list(results)
    attempted = [record for record in records if record.target_removed is not None]
    first_round = [record for record in attempted if record.repair_rounds >= 1]
    three_round = [record for record in attempted if record.repair_rounds <= 3]
    return {
        "target_defect_removal_rate": _safe(
            sum(record.target_removed is True for record in attempted), len(attempted)
        ),
        "first_round_success_rate": _safe(
            sum(
                record.target_removed is True and record.repair_rounds == 1
                for record in first_round
            ),
            len(first_round),
        ),
        "three_round_cumulative_success_rate": _safe(
            sum(record.target_removed is True for record in three_round),
            len(three_round),
        ),
        "collateral_defect_rate": _safe(
            sum(record.collateral_defects > 0 for record in attempted), len(attempted)
        ),
    }


def _defer_reason(result: CaseResult) -> str:
    """Best-effort human-readable defer/error reason for as-is reporting.

    ``raw_output`` shapes differ across arms (frozen-router reports, atomic
    verdict dicts, plain error strings), so this walks the common places a
    reason lives rather than assuming one fixed schema; unknown shapes fall
    back to "unspecified" instead of raising, since this is a reporting path
    that must never mask the original defer as a crash.
    """
    if result.error:
        return result.error
    raw = result.raw_output
    reports = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    for report in reports:
        if not isinstance(report, dict):
            continue
        for inspection in report.get("results", []) or []:
            if not isinstance(inspection, dict) or inspection.get("status") != "defer":
                continue
            for evidence in inspection.get("evidence", []) or []:
                detail = evidence.get("detail") if isinstance(evidence, dict) else None
                if detail:
                    return detail
        if report.get("reason"):
            return str(report["reason"])
    return "unspecified"


def summarize(
    cases: Iterable[EvaluationCase], results: Iterable[CaseResult]
) -> dict[str, object]:
    case_map = {case.case_id: case for case in cases}
    records = list(results)
    by_class: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"truth": [], "prediction": [], "cluster": []}
    )
    # Open-world negative cases (e.g. SlideAudit true negatives) carry an
    # empty ``labels`` list: they are a true negative against *every*
    # candidate defect class the corpus inspects, not against none. Restrict
    # that candidate pool to classes this corpus actually predicted anywhere,
    # not the full taxonomy, so an unlabeled case never falsely contributes a
    # true negative to a defect class this arm/corpus never inspects.
    candidate_classes = {
        label.defect_class for result in records for label in case_map[result.case_id].labels
    } | {defect for result in records for defect in result.predicted_defects}
    for result in records:
        case = case_map[result.case_id]
        truth_classes = {label.defect_class for label in case.labels if label.defective}
        labeled_classes = {label.defect_class for label in case.labels}
        classes = labeled_classes | set(result.predicted_defects) if labeled_classes else candidate_classes
        for defect in classes:
            by_class[defect]["truth"].append(defect in truth_classes)
            by_class[defect]["prediction"].append(defect in result.predicted_defects)
            by_class[defect]["cluster"].append(case.parent_deck_id)
    per_class = {
        key: binary_metrics(value["truth"], value["prediction"])
        for key, value in sorted(by_class.items())
    }
    for key, value in by_class.items():
        if len(set(value["cluster"])) < 2 or len(set(value["truth"])) < 2:
            continue
        fit = mixed_effects_logistic(value["prediction"], value["truth"], value["cluster"])
        per_class[key]["mixed_effects_truth_log_odds"] = fit["treatment_log_odds"]
        per_class[key]["mixed_effects_random_intercept_variance"] = fit[
            "random_intercept_variance"
        ]
    macro = _safe(
        sum(item["balanced_accuracy"] for item in per_class.values()), len(per_class)
    )
    primary = [
        r
        for r in records
        if r.severe_defects is not None
        and r.export_passed is not None
        and r.render_fidelity_passed is not None
    ]
    return {
        "macro_balanced_accuracy": macro,
        "per_class": per_class,
        "e2e_primary_rate": _safe(
            sum(
                r.severe_defects == 0 and r.export_passed and r.render_fidelity_passed
                for r in primary
            ),
            len(primary),
        ),
        "repair": repair_metrics(records),
        "automatic_e2e": {
            "task_constraints_pass_rate": _safe(
                sum(r.task_constraints_passed is True for r in records),
                sum(r.task_constraints_passed is not None for r in records),
            ),
            "mean_section_coverage": _safe(
                sum(
                    r.section_coverage or 0
                    for r in records
                    if r.section_coverage is not None
                ),
                sum(r.section_coverage is not None for r in records),
            ),
            "mean_grounding_score": _safe(
                sum(
                    r.grounding_score or 0
                    for r in records
                    if r.grounding_score is not None
                ),
                sum(r.grounding_score is not None for r in records),
            ),
            "page_count_pass_rate": _safe(
                sum(r.page_count_passed is True for r in records),
                sum(r.page_count_passed is not None for r in records),
            ),
            "render_fidelity_pass_rate": _safe(
                sum(r.render_fidelity_passed is True for r in records),
                sum(r.render_fidelity_passed is not None for r in records),
            ),
        },
        "outcomes": dict(Counter(r.outcome.value for r in records)),
        # Capability downgrades (e.g. image-only SlideAudit cases with no
        # native-IR guarantees) must be reported as-is rather than silently
        # folded into pass/fail, per the 13.15 acceptance gate.
        "capability_downgrades": dict(
            Counter(r.capability_downgrade for r in records if r.capability_downgrade)
        ),
        "defer_reasons": dict(Counter(_defer_reason(r) for r in records if r.outcome == Outcome.DEFER)),
        "integrity_failures": sum(
            case.integrity_status != "valid" for case in case_map.values()
        ),
        "model_errors": sum(
            r.outcome in {Outcome.ERROR, Outcome.TIMEOUT, Outcome.MISSING}
            for r in records
        ),
        "export_failures": sum(r.export_passed is False for r in records),
        "mean_tokens": _safe(sum(r.tokens for r in records), len(records)),
        "mean_cost": _safe(sum(r.cost for r in records), len(records)),
        "mean_latency_seconds": _safe(
            sum(r.latency_seconds for r in records), len(records)
        ),
    }
