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


def summarize(
    cases: Iterable[EvaluationCase], results: Iterable[CaseResult]
) -> dict[str, object]:
    case_map = {case.case_id: case for case in cases}
    records = list(results)
    by_class: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"truth": [], "prediction": []}
    )
    for result in records:
        case = case_map[result.case_id]
        truth_classes = {label.defect_class for label in case.labels if label.defective}
        labeled_classes = {label.defect_class for label in case.labels}
        classes = labeled_classes | set(result.predicted_defects)
        for defect in classes:
            by_class[defect]["truth"].append(defect in truth_classes)
            by_class[defect]["prediction"].append(defect in result.predicted_defects)
    per_class = {
        key: binary_metrics(value["truth"], value["prediction"])
        for key, value in sorted(by_class.items())
    }
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
        "outcomes": dict(Counter(r.outcome.value for r in records)),
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
