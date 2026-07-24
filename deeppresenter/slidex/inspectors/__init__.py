"""Deterministic Slidex inspector implementations."""

from deeppresenter.slidex.inspectors.base import Inspector, inspect_safely
from deeppresenter.slidex.inspectors.geometry import (
    AlignmentInspector,
    DeclaredOverflowInspector,
    MarginInspector,
    OverlapInspector,
    RenderOverflowInspector,
)
from deeppresenter.slidex.inspectors.style import (
    BrandColorInspector,
    TypographyInspector,
)
from deeppresenter.slidex.inspectors.terminology import TerminologyInspector

__all__ = [
    "AlignmentInspector",
    "BrandColorInspector",
    "DeclaredOverflowInspector",
    "Inspector",
    "MarginInspector",
    "OverlapInspector",
    "RenderOverflowInspector",
    "TerminologyInspector",
    "TypographyInspector",
    "inspect_safely",
]
