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
from deeppresenter.slidex.inspectors.neural import (
    AtomicNeuralClient,
    DeckSemanticInspector,
    DensityInspector,
    ImageTextContradictionInspector,
    NeuralCapabilityError,
    RenderAnomalyInspector,
    RenderOnlyGeometryInspector,
    TitleBodyMismatchInspector,
)
from deeppresenter.slidex.inspectors.reference import ReferenceInspector

__all__ = [
    "AlignmentInspector",
    "AtomicNeuralClient",
    "DeckSemanticInspector",
    "DensityInspector",
    "ImageTextContradictionInspector",
    "NeuralCapabilityError",
    "ReferenceInspector",
    "RenderAnomalyInspector",
    "RenderOnlyGeometryInspector",
    "TitleBodyMismatchInspector",
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
