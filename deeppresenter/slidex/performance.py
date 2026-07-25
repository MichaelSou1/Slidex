"""Small dependency-free latency and episode-cost aggregation."""

from __future__ import annotations

import math
from collections import defaultdict


class PerformanceTracker:
    def __init__(self) -> None:
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._episode_costs: dict[str, float] = defaultdict(float)

    def record(
        self,
        operation: str,
        latency_ms: float,
        *,
        episode_id: str | None = None,
        cost: float = 0,
    ) -> None:
        if latency_ms < 0 or cost < 0:
            raise ValueError("latency and cost must be non-negative")
        self._latencies[operation].append(latency_ms)
        if episode_id:
            self._episode_costs[episode_id] += cost

    def summary(self) -> dict[str, object]:
        operations = {}
        for operation, values in sorted(self._latencies.items()):
            ordered = sorted(values)
            operations[operation] = {
                "count": len(ordered),
                "p50_latency_ms": self._percentile(ordered, 0.50),
                "p95_latency_ms": self._percentile(ordered, 0.95),
            }
        return {
            "operations": operations,
            "episode_costs": dict(sorted(self._episode_costs.items())),
        }

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0
        return values[max(0, math.ceil(len(values) * quantile) - 1)]
