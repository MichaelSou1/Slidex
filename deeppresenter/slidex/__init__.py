"""Slidex domain models, browser observation, and artifact persistence."""

from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.attribution import FailureAttributor
from deeppresenter.slidex.browser import BrowserObserver, extract_declared_ir
from deeppresenter.slidex.critic import HybridCritic, SymbolicCritic
from deeppresenter.slidex.deck import DeckInspector, enforce_export_gate
from deeppresenter.slidex.export import (
    FinalExportService,
    LibreOfficeRenderer,
    RenderFidelityValidator,
)
from deeppresenter.slidex.repair import DeterministicRepairer, actions_from_report
from deeppresenter.slidex.reward import (
    RewardCalibrator,
    RewardConfig,
    RewardEngine,
    ValiditySignals,
    explain_reward,
    persist_reward,
)
from deeppresenter.slidex.router import FrozenCriticRouter, FrozenRouterConfig
from deeppresenter.slidex.models import SlideArtifact

__all__ = [
    "ArtifactStore",
    "BrowserObserver",
    "DeckInspector",
    "DeterministicRepairer",
    "FailureAttributor",
    "FinalExportService",
    "FrozenCriticRouter",
    "FrozenRouterConfig",
    "HybridCritic",
    "LibreOfficeRenderer",
    "RenderFidelityValidator",
    "RewardCalibrator",
    "RewardConfig",
    "RewardEngine",
    "ValiditySignals",
    "SlideArtifact",
    "SymbolicCritic",
    "actions_from_report",
    "enforce_export_gate",
    "explain_reward",
    "persist_reward",
    "extract_declared_ir",
]
