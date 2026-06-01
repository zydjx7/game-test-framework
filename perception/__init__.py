"""Unified perception layer for game-state extraction.

This package bridges the AssaultCube baseline (classical CV in
``Code/GameStateChecker``) and the upcoming ViZDoom main line (ViZDoom
ground-truth + VLM perceptors). All backends share the
:class:`GameStatePerceptor` interface and emit a :class:`GameState`.

Phase 0.1 only ships the base interface and the CV wrapper; Phase 1+
will add ``ground_truth.py`` and ``vlm_perceptor.py`` next to them.
"""

from .base import GameState, GameStatePerceptor
from .cv_perceptor import CVPerceptor
from .ground_truth import GroundTruthPerceptor, ammo_level

__all__ = [
    "GameState",
    "GameStatePerceptor",
    "CVPerceptor",
    "GroundTruthPerceptor",
    "ammo_level",
]
