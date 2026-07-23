"""Browser-side stable ID normalization used by the native IR extractor."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def deterministic_fallback_id(dom_path: str) -> str:
    """Generate a stable fallback ID from an element's structural DOM path."""
    digest = hashlib.sha256(dom_path.encode()).hexdigest()[:16]
    return f"auto-{digest}"


def validate_element_ids(ids: Iterable[str]) -> list[str]:
    """Validate explicit IDs and return warnings for missing IDs."""
    seen: set[str] = set()
    warnings: list[str] = []
    for index, element_id in enumerate(ids):
        if not element_id or element_id.isspace():
            warnings.append(f"element {index} is missing data-slidex-id")
            continue
        if element_id in seen:
            raise ValueError(f"duplicate data-slidex-id: {element_id}")
        seen.add(element_id)
    return warnings
