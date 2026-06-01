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


def test_one_settled_keyframe_per_distinct_ammo_value():
    # ammo holds 26 for 3 tics, then 25 for 2 tics, then 24 for 1 tic.
    # Middle-of-run: run[26]=ticks0-2 -> idx1 (tick1); run[25]=ticks3-4 -> idx1 (tick4);
    # run[24]=tick5 -> idx0 (tick5).
    frames = [
        _frame(0, 26), _frame(1, 26), _frame(2, 26),
        _frame(3, 25), _frame(4, 25),
        _frame(5, 24),
    ]
    keys = sample_ammo_change_keyframes(_traj(frames))
    assert [f.tick for f in keys] == [1, 4, 5]
    assert [f.game_variables["ammo"] for f in keys] == [26, 25, 24]


def test_single_run_returns_its_middle():
    frames = [_frame(0, 10), _frame(1, 10), _frame(2, 10)]
    keys = sample_ammo_change_keyframes(_traj(frames))
    assert len(keys) == 1
    assert keys[0].tick == 1  # middle of the single run


def test_skips_frames_without_ammo():
    frames = [_frame(0, None), _frame(1, 12), _frame(2, None), _frame(3, 11)]
    keys = sample_ammo_change_keyframes(_traj(frames))
    assert [f.game_variables["ammo"] for f in keys] == [12, 11]


def test_empty_trajectory_yields_no_keyframes():
    assert sample_ammo_change_keyframes(_traj([])) == []
