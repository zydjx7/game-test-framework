"""ToyFPS adapter: primitives + a multi-metric action library.

Implements the SAME contracts the agent layer consumes, so `run_agent_loop` /
`run_reflective_agent` run on ToyFPS unchanged:
- ToyPrimitives: reset / observe / fire_once / heal / wait  (the `prim` an
  action library drives; `observe()` returns the latest state).
- ToyActions: composites returning canonical <metric>_before/<metric>_after via
  snapshot_result, plus DESCRIPTIONS / EXPECTATIONS / list_templates / run /
  check_expectation — exactly the action-library contract TestActions exposes.

Perception is REUSED: GroundTruthPerceptor reads ammo/health/score from the
state's game_variables, so even the perceptor is shared across games.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from actions.result import decreased, increased, snapshot_result
from perception.base import GameStatePerceptor

from .game import ToyFPS


class ToyPrimitives:
    """Layer-1 primitives over a ToyFPS state machine (mirrors ActionPrimitives)."""

    def __init__(self, game: Optional[ToyFPS] = None) -> None:
        self.game = game or ToyFPS()
        self.last_state = None

    def reset(self):
        self.last_state = self.game.reset()
        return self.last_state

    def fire_once(self):
        self.last_state = self.game.fire()
        return self.last_state

    def heal(self):
        self.last_state = self.game.heal()
        return self.last_state

    def wait(self, tics: int = 1):
        self.last_state = self.game.idle()
        return self.last_state

    def observe(self):
        return self.last_state


class ToyActions:
    """Layer-3 test templates for ToyFPS — multi-metric (ammo / score / health)."""

    __test__ = False

    DESCRIPTIONS: Dict[str, str] = {
        "fire_and_check_ammo": "Fire a shot; report ammo before/after. Use when the goal is about consuming ammo.",
        "fire_and_check_score": "Fire a shot; report score before/after. Use when the goal is about scoring/hitting.",
        "heal_and_check_health": "Heal; report health before/after. Use when the goal is about restoring health.",
    }

    EXPECTATIONS: Dict[str, Dict[str, Any]] = {
        "fire_and_check_ammo": {"describe": "firing should decrease ammo by at least 1", "check": decreased("ammo", 1)},
        "fire_and_check_score": {"describe": "firing should increase score by at least 1", "check": increased("score", 1)},
        "heal_and_check_health": {"describe": "healing should increase health by at least 1", "check": increased("health", 1)},
    }

    def __init__(self, primitives: ToyPrimitives) -> None:
        self.prim = primitives

    def fire_and_check_ammo(self, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.fire_once()
        after = self._read(perceptor)
        return snapshot_result({"ammo": before.ammo}, {"ammo": after.ammo})

    def fire_and_check_score(self, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.fire_once()
        after = self._read(perceptor)
        return snapshot_result({"score": before.score}, {"score": after.score})

    def heal_and_check_health(self, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.heal()
        after = self._read(perceptor)
        return snapshot_result({"health": before.health}, {"health": after.health})

    # -- contract methods (same shape as TestActions) --------------------

    def run(self, template_name: str, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        if template_name not in self.list_templates():
            raise ValueError(f"unknown test template: {template_name}")
        return getattr(self, template_name)(perceptor)

    def observe(self, perceptor: GameStatePerceptor):
        """Re-perceive current state without acting (graph re_observe recovery)."""

        return self._read(perceptor)

    @classmethod
    def list_templates(cls) -> List[str]:
        return list(cls.DESCRIPTIONS.keys())

    @classmethod
    def check_expectation(
        cls, template_name: str, result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        spec = cls.EXPECTATIONS.get(template_name)
        if spec is None:
            return None
        check: Callable[[Dict[str, Any]], bool] = spec["check"]
        if check(result):
            return None
        return {"action": template_name, "expected": spec["describe"], "result": result}

    def _read(self, perceptor: GameStatePerceptor):
        state = self.prim.observe()
        screen = getattr(state, "screen", None)
        game_variables = dict(getattr(state, "game_variables", {}) or {})
        return perceptor.perceive(screen, game_variables=game_variables)


# Perception is reused, not reimplemented: ToyFPS uses GroundTruthPerceptor.
from perception import GroundTruthPerceptor as ToyPerceptor  # noqa: E402,F401
