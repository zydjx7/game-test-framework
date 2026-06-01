"""Unit tests for event-driven ammo-change keyframe sampling."""

from __future__ import annotations

import numpy as np

from env.trajectory_recorder import Trajectory, TrajectoryFrame
from experiments.sampling import sample_ammo_change_keyframes


def _frame(tick: int, ammo) -> TrajectoryFrame:
    return TrajectoryFrame(
        tick=tick,
        screen=np.zeros((2, 2, 3), dtype=np.uint8),
        game_variables={} if ammo is None else {"ammo": ammo},
    )


def _traj(frames) -> Trajectory:
    return Trajectory(scenario="defend_the_center", frames=frames)


def test_one_keyframe_per_distinct_ammo_value():
    # ammo holds 26 for several tics, then 25, then 25, then 24
    frames = [
        _frame(0, 26), _frame(1, 26), _frame(2, 26),
        _frame(3, 25), _frame(4, 25),
        _frame(5, 24),
    ]
    keys = sample_ammo_change_keyframes(_traj(frames))
    assert [f.tick for f in keys] == [0, 3, 5]
    assert [f.game_variables["ammo"] for f in keys] == [26, 25, 24]


def test_includes_first_frame():
    frames = [_frame(0, 10), _frame(1, 10)]
    keys = sample_ammo_change_keyframes(_traj(frames))
    assert len(keys) == 1
    assert keys[0].tick == 0


def test_skips_frames_without_ammo():
    frames = [_frame(0, None), _frame(1, 12), _frame(2, None), _frame(3, 11)]
    keys = sample_ammo_change_keyframes(_traj(frames))
    assert [f.game_variables["ammo"] for f in keys] == [12, 11]


def test_empty_trajectory_yields_no_keyframes():
    assert sample_ammo_change_keyframes(_traj([])) == []
