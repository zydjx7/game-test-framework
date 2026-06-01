"""Layer 2/3 of the action library: composite test templates.

A composite does one observable test action: read state, act, read state,
return a structured result dict the agent's goal-success criteria can judge.
These template names are exactly what the agent chooses among (its tools);
the LLM never sees raw ViZDoom buttons.

Perception is INJECTED: pass a GroundTruthPerceptor for deterministic v1 demos
(so any failure is attributable to action/decision, not perception noise), or
a VLMPerceptor to exercise the full perceive->decide loop. Both satisfy the
GameStatePerceptor interface, so composites are perceptor-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List

from perception.base import GameState, GameStatePerceptor


class TestActions:
    """Layer-3 test templates the agent operates on."""

    # The class name starts with "Test" (it holds *test* action templates, per
    # research-plan.md §2.2), which collides with pytest's collection rule.
    # This tells pytest the class is NOT a test case.
    __test__ = False

    def __init__(self, primitives: Any) -> None:
        self.prim = primitives

    # -- the tools the agent can choose ----------------------------------

    def fire_and_check_ammo(self, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        """Fire one shot; report ammo before/after and the delta."""

        before = self._read(perceptor)
        self.prim.fire_once()
        after = self._read(perceptor)
        return self._delta_result(before, after)

    def idle_and_check_ammo(self, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        """Let the game advance without firing; ammo should be unchanged."""

        before = self._read(perceptor)
        self.prim.wait()
        after = self._read(perceptor)
        return self._delta_result(before, after)

    # -- helpers ---------------------------------------------------------

    def run(self, template_name: str, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        """Dispatch by template name (used by the agent loop)."""

        if template_name not in self.list_templates():
            raise ValueError(f"unknown test template: {template_name}")
        return getattr(self, template_name)(perceptor)

    @classmethod
    def list_templates(cls) -> List[str]:
        return ["fire_and_check_ammo", "idle_and_check_ammo"]

    def _read(self, perceptor: GameStatePerceptor) -> GameState:
        state = self.prim.observe()
        return perceptor.perceive(
            state.screen, game_variables=dict(state.game_variables)
        )

    @staticmethod
    def _delta_result(before: GameState, after: GameState) -> Dict[str, Any]:
        b = before.ammo if before.ammo is not None else 0
        a = after.ammo if after.ammo is not None else 0
        return {
            "ammo_before": before.ammo,
            "ammo_after": after.ammo,
            "delta": b - a,
        }
