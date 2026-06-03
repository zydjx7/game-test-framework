"""Controlled failure injection for Phase 3 evaluation (Stage D).

These wrappers fabricate PERCEPTION and EXECUTION failures so reflection can be
exercised deterministically on the real stack, with the injected type known as
ground truth (Doc/phase3-design.md §5/§6). They live here, in the experiment
layer -- production perceptors/primitives/composites are never modified.

- PERCEPTION failure: the game is fine and ammo really dropped, but the
  perceptor masks the change (reports the previous ammo) so the observed delta
  looks like 0. A truthful re-read (re_observe) then sees it -> recoverable.
- EXECUTION failure: the action does not go through, so ammo really does NOT
  drop. A retry actually fires -> recoverable.
- PERSISTENT failure: set the count very large so the action never goes through,
  approximating a LOGIC bug ("retries keep failing -> should be reported"),
  since real logic bugs (mutation) are Phase 4.

Determinism: a fixed count (mask the first N ammo changes / fail the first N
fires) makes tests and eval reproducible without RNG.
"""

from __future__ import annotations

from typing import Any, Optional

from perception.base import GameState, GameStatePerceptor


class PerceptionFailureInjector(GameStatePerceptor):
    """Wrap a perceptor and MASK the first `mask_changes` ammo changes: when the
    true ammo differs from the last reported value, report the stale value
    instead (delta looks like 0) though the game really changed. Once the budget
    is spent, pass through truthfully -- so a re-observe recovers.
    """

    def __init__(self, wrapped: GameStatePerceptor, mask_changes: int = 1) -> None:
        self.wrapped = wrapped
        self.mask_changes = mask_changes
        self._remaining = mask_changes
        self._prev_ammo: Optional[int] = None

    def perceive(self, screenshot: Any, **kwargs: Any) -> GameState:
        state = self.wrapped.perceive(screenshot, **kwargs)
        true_ammo = state.ammo
        if (
            self._remaining > 0
            and self._prev_ammo is not None
            and true_ammo is not None
            and true_ammo != self._prev_ammo
        ):
            self._remaining -= 1
            # report the stale ammo; keep _prev unchanged so the masked round
            # reads as "no change". The next (truthful) read advances _prev.
            return GameState(
                ammo=self._prev_ammo,
                health=state.health,
                weapon=state.weapon,
                crosshair_red=state.crosshair_red,
                enemies_visible=state.enemies_visible,
                raw_response={"injected": "perception", "true_ammo": true_ammo},
            )
        self._prev_ammo = true_ammo
        return state


class ExecutionFailureInjector:
    """Wrap ActionPrimitives so the first `fail_fires` fire_once() calls do NOT
    advance ATTACK (the shot never goes through, ammo unchanged); later calls
    fire normally. A large `fail_fires` approximates a persistent LOGIC-like bug.
    """

    def __init__(self, wrapped: Any, fail_fires: int = 1) -> None:
        self.wrapped = wrapped
        self.fail_fires = fail_fires
        self._fire_calls = 0

    def fire_once(self) -> Any:
        self._fire_calls += 1
        if self._fire_calls <= self.fail_fires:
            # "fired" but nothing happens: advance the same number of tics as a
            # real shot, just without ATTACK, so timing matches but ammo is unchanged
            from actions.primitives import FIRE_TICS, SETTLE_TICS

            return self.wrapped.wait(FIRE_TICS + SETTLE_TICS)
        return self.wrapped.fire_once()

    def __getattr__(self, name: str) -> Any:
        # delegate observe / wait / reset / etc. to the real primitives
        return getattr(self.wrapped, name)
