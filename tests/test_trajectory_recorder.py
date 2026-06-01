"""Unit tests for ``env.trajectory_recorder``.

A ``FakeEnv`` stands in for ``VizDoomEnv`` so these tests do NOT launch
ViZDoom. They verify three things:

1. ``record_episode`` records one frame per pre-step state and stops at
   ``done`` without storing the terminal degenerate frame.
2. ``max_tics`` caps recording.
3. ``save_trajectory`` / ``load_trajectory`` round-trip frames losslessly
   despite the on-disk PNG compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from env.trajectory_recorder import (
    load_trajectory,
    record_episode,
    save_trajectory,
)


@dataclass
class FakeState:
    step: int
    screen: np.ndarray
    game_variables: Dict[str, int]
    done: bool


@dataclass
class FakeEnv:
    """Emits ``n_steps`` valid frames, then a terminal done-state.

    Mimics ``VizDoomEnv``: ammo starts at 26 and drops by 1 each step;
    the terminal state carries the 1x1 zero screen ViZDoom returns once
    the episode is finished.
    """

    n_steps: int
    height: int = 8
    width: int = 8
    _i: int = field(default=0, init=False)

    def _frame(self, step: int) -> FakeState:
        ammo = max(26 - step, 0)
        # Fill the screen with the ammo value so we can assert which frame
        # survived the save/load round-trip.
        screen = np.full((self.height, self.width, 3), ammo, dtype=np.uint8)
        return FakeState(step=step, screen=screen, game_variables={"ammo": ammo}, done=False)

    def reset(self) -> FakeState:
        self._i = 0
        return self._frame(0)

    def step(self, action: List[int]) -> FakeState:
        self._i += 1
        if self._i >= self.n_steps:
            return FakeState(
                step=self._i,
                screen=np.zeros((1, 1, 3), dtype=np.uint8),
                game_variables={},
                done=True,
            )
        return self._frame(self._i)


def _always_attack(_state) -> List[int]:
    return [0, 0, 1]


def test_records_one_frame_per_state_until_done() -> None:
    env = FakeEnv(n_steps=5)
    traj = record_episode(env, _always_attack, scenario="basic")

    # 5 valid states (steps 0..4); the terminal done-state is not stored.
    assert len(traj) == 5
    assert [f.tick for f in traj.frames] == [0, 1, 2, 3, 4]
    assert traj.frames[0].game_variables["ammo"] == 26
    assert traj.frames[-1].game_variables["ammo"] == 22
    assert traj.scenario == "basic"
    assert traj.metadata["recorded_frames"] == 5


def test_max_tics_caps_recording() -> None:
    env = FakeEnv(n_steps=100)
    traj = record_episode(env, _always_attack, scenario="basic", max_tics=10)
    assert len(traj) == 10


def test_save_load_roundtrip_is_lossless(tmp_path) -> None:
    env = FakeEnv(n_steps=6)
    traj = record_episode(
        env, _always_attack, scenario="basic", metadata={"policy": "always_attack"}
    )

    path = save_trajectory(traj, tmp_path / "basic_ep0.pkl")
    assert path.exists()

    loaded = load_trajectory(path)
    assert loaded.scenario == "basic"
    assert loaded.metadata["policy"] == "always_attack"
    assert len(loaded) == len(traj)

    for original, restored in zip(traj.frames, loaded.frames):
        assert restored.tick == original.tick
        assert restored.game_variables == original.game_variables
        # PNG is lossless, so the decoded array must equal the original.
        np.testing.assert_array_equal(restored.screen, original.screen)
