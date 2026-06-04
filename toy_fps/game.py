"""ToyFPS state machine — a minimal FPS with three metrics (no engine).

State: ammo / health / score. It is deliberately tiny; the point is not game
realism but to drive the framework's perception/action/agent layers with
MULTIPLE metrics (a decreasing one: ammo; increasing ones: score, health).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ToyState:
    """Mirrors the shape the framework expects: a `game_variables` dict and an
    optional `screen` (ToyFPS has none, which exercises the screen-OPTIONAL
    Observation contract)."""

    step: int
    game_variables: Dict[str, int]
    done: bool = False
    screen: None = None  # ToyFPS is screen-less on purpose


class ToyFPS:
    START_AMMO = 10
    START_HEALTH = 50  # below max so healing visibly increases it
    START_SCORE = 0
    MAX_HEALTH = 100

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> ToyState:
        self.ammo = self.START_AMMO
        self.health = self.START_HEALTH
        self.score = self.START_SCORE
        self._step = 0
        return self._state()

    def _state(self, done: bool = False) -> ToyState:
        return ToyState(
            step=self._step,
            game_variables={"ammo": self.ammo, "health": self.health, "score": self.score},
            done=done,
        )

    # -- mechanics (called by ToyPrimitives) -----------------------------

    def fire(self) -> ToyState:
        """A shot: consumes 1 ammo and scores 1 (if ammo remains)."""
        self._step += 1
        if self.ammo > 0:
            self.ammo -= 1
            self.score += 1
        return self._state(done=self.ammo == 0)

    def heal(self, amount: int = 10) -> ToyState:
        self._step += 1
        self.health = min(self.MAX_HEALTH, self.health + amount)
        return self._state()

    def idle(self) -> ToyState:
        self._step += 1
        return self._state()
