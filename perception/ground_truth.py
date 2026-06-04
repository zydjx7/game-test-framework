"""Ground-truth perceptor reading ViZDoom game variables directly.

This is the oracle for Phase 1 perception accuracy: ViZDoom exposes the
true ammo / health through ``state.game_variables``, so we never need
human labels to score the VLM. The ``VizDoomEnv`` wrapper already converts
the raw numpy ``game_variables`` array into a named dict (e.g.
``{"ammo": 7, "health": 24}``) and stores it on every recorded
``TrajectoryFrame``; this perceptor consumes that dict.

``ammo_level`` is the SINGLE source of truth for the high/medium/low
bucketing. Both this ground-truth perceptor and the VLM-side scoring must
import it, so the abstract-level accuracy compares like with like
(Phase 1 design doc §4 / §5).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import GameState, GameStatePerceptor

# Ammo level boundaries for defend_the_center (ammo range [0, 26]).
# Even thirds: high 18-26, medium 9-17, low 0-8. See design doc §4.
AMMO_LEVEL_HIGH_MIN = 18
AMMO_LEVEL_MEDIUM_MIN = 9


def ammo_level(ammo: Optional[int]) -> str:
    """Bucket an ammo count into 'high' / 'medium' / 'low' / 'unknown'.

    Shared by the ground-truth side and the VLM-scoring side so the
    abstract-level accuracy is computed with identical boundaries.
    """

    if ammo is None:
        return "unknown"
    if ammo >= AMMO_LEVEL_HIGH_MIN:
        return "high"
    if ammo >= AMMO_LEVEL_MEDIUM_MIN:
        return "medium"
    return "low"


class GroundTruthPerceptor(GameStatePerceptor):
    """Reads ground-truth state from ViZDoom game variables.

    Unlike the CV / VLM backends, this perceptor ignores the screenshot
    pixels entirely -- it reads the engine's own variables, which is why it
    serves as the accuracy oracle.

    Recognised kwargs:

    - ``game_variables`` (dict): named variables from a recorded
      ``TrajectoryFrame`` (preferred path for offline evaluation).
    - ``vizdoom_state`` (object with ``.game_variables`` and an optional
      ``var_names`` list): a live state; used only if ``game_variables``
      is not supplied.
    """

    def perceive(
        self,
        screenshot: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> GameState:
        game_variables = kwargs.get("game_variables")

        if game_variables is None:
            game_variables = self._extract_from_live_state(kwargs.get("vizdoom_state"))

        gv: Dict[str, int] = game_variables or {}
        ammo = gv.get("ammo")
        health = gv.get("health")
        score = gv.get("score")

        return GameState(
            ammo=int(ammo) if ammo is not None else None,
            health=int(health) if health is not None else None,
            score=int(score) if score is not None else None,
            raw_response={"game_variables": dict(gv)},
        )

    @staticmethod
    def _extract_from_live_state(vizdoom_state: Any) -> Dict[str, int]:
        """Best-effort extraction from a live ViZDoom state object.

        Offline evaluation uses recorded ``game_variables`` dicts, so this
        path is a fallback. A live ``vizdoom_state.game_variables`` is a
        positional numpy array; without the scenario's var-name ordering we
        cannot name them, so we only use it when the caller attached a
        ``var_names`` list (as ``VizDoomEnv`` could in the future).
        """

        if vizdoom_state is None:
            return {}

        values = getattr(vizdoom_state, "game_variables", None)
        names = getattr(vizdoom_state, "var_names", None)
        if values is None or names is None:
            return {}

        return {name: int(values[i]) for i, name in enumerate(names) if i < len(values)}
