"""Slidex domain models, browser observation, and artifact persistence."""

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.browser import BrowserObserver, extract_declared_ir
from deeppresenter.slidex.critic import SymbolicCritic
from deeppresenter.slidex.models import SlideArtifact

__all__ = [
    "ArtifactStore",
    "BrowserObserver",
    "SlideArtifact",
    "SymbolicCritic",
    "extract_declared_ir",
]
