"""Slidex domain models, browser observation, and artifact persistence."""

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.attribution import FailureAttributor
from deeppresenter.slidex.browser import BrowserObserver, extract_declared_ir
from deeppresenter.slidex.critic import HybridCritic, SymbolicCritic
from deeppresenter.slidex.deck import DeckInspector, enforce_export_gate
from deeppresenter.slidex.repair import DeterministicRepairer, actions_from_report
from deeppresenter.slidex.router import FrozenCriticRouter, FrozenRouterConfig
from deeppresenter.slidex.models import SlideArtifact

__all__ = [
    "ArtifactStore",
    "BrowserObserver",
    "DeckInspector",
    "DeterministicRepairer",
    "FailureAttributor",
    "FrozenCriticRouter",
    "FrozenRouterConfig",
    "HybridCritic",
    "SlideArtifact",
    "SymbolicCritic",
    "actions_from_report",
    "enforce_export_gate",
    "extract_declared_ir",
]
