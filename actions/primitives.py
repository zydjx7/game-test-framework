"""Layer 1 of the action library: thin wrappers over VizDoomEnv.step().

Primitives translate a semantic intent ("fire one shot", "wait") into the
right number of ViZDoom tics + button vector, and keep a handle on the most
recent state. They do NOT read perception or judge anything -- that is the
composites' job (Layer 2).

Phase 2 Stage 0 finding baked in here:
- A single pistol shot needs ~14 tics to fire: episode_start_time (10, the
  gun-raise) + PISTOL1 pre-fire (4). Firing for only 4-8 tics yields ZERO
  ammo change, which a naive test would mis-read as a logic bug. FIRE_TICS=16
  guarantees exactly one decrement with margin.
- SETTLE_TICS lets the rendered HUD catch up to game_variables before a
  pixel-based perceptor reads it (the Phase 1 ~1-tic HUD/state desync).
"""

from __future__ import annotations

from typing import Any, List, Optional

FIRE_TICS = 16
SETTLE_TICS = 2


class ActionPrimitives:
    """Stateful wrapper: holds the env and the last observed state."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self._button_names: List[str] = list(env.get_button_names())
        self._n = len(self._button_names)
        self.last_state: Any = None

    def _vec(self, button: Optional[str]) -> List[int]:
        vec = [0] * self._n
        if button is not None and button in self._button_names:
            vec[self._button_names.index(button)] = 1
        return vec

    def _hold(self, button: Optional[str], tics: int) -> Any:
        """Repeat one button (or noop) for `tics`, stopping early on done."""

        vec = self._vec(button)
        for _ in range(tics):
            self.last_state = self.env.step(vec)
            if getattr(self.last_state, "done", False):
                break
        return self.last_state

    def reset(self) -> Any:
        self.last_state = self.env.reset()
        return self.last_state

    def fire_once(self) -> Any:
        """Fire exactly one shot, then settle so the HUD reflects the new ammo."""

        self._hold("ATTACK", FIRE_TICS)
        self._hold(None, SETTLE_TICS)
        return self.last_state

    def wait(self, tics: int = FIRE_TICS) -> Any:
        """Let the game advance without firing (noop), e.g. the idle control."""

        return self._hold(None, tics)

    def observe(self) -> Any:
        """Return the latest state without advancing the game."""

        return self.last_state
