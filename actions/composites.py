"""Layer 2/3 of the action library: composite test templates.

A composite does one observable test action: read state, act, read state,
return a structured result in the canonical <metric>_before/<metric>_after shape
(via `snapshot_result`) that the agent's goal-success criteria can judge. These
template names are exactly what the agent chooses among (its tools); the LLM
never sees raw engine buttons.

Perception is INJECTED: pass a GroundTruthPerceptor for deterministic v1 demos
(so any failure is attributable to action/decision, not perception noise), or
a VLMPerceptor to exercise the full perceive->decide loop. Both satisfy the
GameStatePerceptor interface, so composites are perceptor-agnostic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from perception.base import GameState, GameStatePerceptor

from .result import snapshot_result


def _decreased(metric: str, by: int = 1) -> Callable[[Dict[str, Any]], bool]:
    """Expectation: <metric> dropped by at least `by`. None reads -> violated."""

    def check(result: Dict[str, Any]) -> bool:
        before = result.get(f"{metric}_before")
        after = result.get(f"{metric}_after")
        return before is not None and after is not None and (before - after) >= by

    return check


def _unchanged(metric: str) -> Callable[[Dict[str, Any]], bool]:
    """Expectation: <metric> is unchanged. None reads -> violated."""

    def check(result: Dict[str, Any]) -> bool:
        before = result.get(f"{metric}_before")
        after = result.get(f"{metric}_after")
        return before is not None and after is not None and before == after

    return check


class TestActions:
    """Layer-3 test templates the agent operates on."""

    # The class name starts with "Test" (it holds *test* action templates, per
    # research-plan.md §2.2), which collides with pytest's collection rule.
    # This tells pytest the class is NOT a test case.
    __test__ = False

    # Natural-language descriptions the agent's function-calling decider shows
    # the LLM so it can pick the right template for a goal. The distinction
    # between "fire" and "idle" is deliberately explicit so the agent can tell
    # a fire-goal from a do-not-fire goal.
    DESCRIPTIONS: Dict[str, str] = {
        "fire_and_check_ammo": (
            "Fire exactly one shot, then report ammo before and after. Choose "
            "this when the goal is to fire, shoot, or consume ammo."
        ),
        "idle_and_check_ammo": (
            "Let the game advance WITHOUT firing, then report ammo before and "
            "after. Choose this when the goal is about staying idle or NOT firing."
        ),
    }

    # Per-template EXPECTED EFFECT (Phase 3 Stage A). Each maps a template to a
    # human-readable description (shown to the reflection LLM) plus a predicate
    # over the result dict. When a template runs and its predicate is FALSE, the
    # step is an anomaly -> reflection (Stage B/C) decides whether it was a
    # perception / execution / logic failure. Predicates read the canonical
    # <metric>_before/<metric>_after keys, so they generalize to new metrics.
    EXPECTATIONS: Dict[str, Dict[str, Any]] = {
        "fire_and_check_ammo": {
            "describe": "firing should decrease ammo by at least 1",
            "check": _decreased("ammo", 1),
        },
        "idle_and_check_ammo": {
            "describe": "idling should leave ammo unchanged",
            "check": _unchanged("ammo"),
        },
    }

    def __init__(self, primitives: Any) -> None:
        self.prim = primitives

    # -- the tools the agent can choose ----------------------------------

    def fire_and_check_ammo(self, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        """Fire one shot; report ammo before/after."""

        before = self._read(perceptor)
        self.prim.fire_once()
        after = self._read(perceptor)
        return snapshot_result({"ammo": before.ammo}, {"ammo": after.ammo})

    def idle_and_check_ammo(self, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        """Let the game advance without firing; ammo should be unchanged."""

        before = self._read(perceptor)
        self.prim.wait()
        after = self._read(perceptor)
        return snapshot_result({"ammo": before.ammo}, {"ammo": after.ammo})

    # -- helpers ---------------------------------------------------------

    def run(self, template_name: str, perceptor: GameStatePerceptor) -> Dict[str, Any]:
        """Dispatch by template name (used by the agent loop)."""

        if template_name not in self.list_templates():
            raise ValueError(f"unknown test template: {template_name}")
        return getattr(self, template_name)(perceptor)

    @classmethod
    def list_templates(cls) -> List[str]:
        return ["fire_and_check_ammo", "idle_and_check_ammo"]

    @classmethod
    def check_expectation(
        cls, template_name: str, result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return None if the step met its expectation (or has none), else an
        anomaly dict the reflection layer can consume.
        """

        spec = cls.EXPECTATIONS.get(template_name)
        if spec is None:
            return None
        check: Callable[[Dict[str, Any]], bool] = spec["check"]
        if check(result):
            return None
        return {
            "action": template_name,
            "expected": spec["describe"],
            "result": result,
        }

    def _read(self, perceptor: GameStatePerceptor) -> GameState:
        # Observation contract: a state must expose game_variables (a dict);
        # screen is OPTIONAL (a screen-less ToyGame state has none). Tolerant
        # access keeps composites usable across very different games.
        state = self.prim.observe()
        screen = getattr(state, "screen", None)
        game_variables = dict(getattr(state, "game_variables", {}) or {})
        return perceptor.perceive(screen, game_variables=game_variables)
