"""Event-driven keyframe sampling on ammo change (settled-frame variant).

Why event-driven (Phase 1 design doc §0 / §6): the pistol fires once per
~14 tics, so ammo is constant for long stretches. Equidistant sampling would
waste VLM calls on duplicate ammo values. We want one keyframe per distinct
ammo count.

WHY THE MIDDLE FRAME, not the first (2026-06-01 spike finding): at the exact
tic the ammo game-variable decrements, the rendered HUD still shows the
PREVIOUS digit for one tic -- the screen_buffer lags game_variables by ~1 tic.
Sampling that transition frame makes the VLM read GT+1 every time (we measured
concrete accuracy collapse to ~5% from this artifact alone). The state vector
and the rendered frame must correspond, so we sample a SETTLED frame from the
middle of each constant-ammo run, where the HUD has caught up.
"""

from __future__ import annotations

from typing import List

from env.trajectory_recorder import Trajectory, TrajectoryFrame


def _ammo_runs(trajectory: Trajectory) -> List[List[TrajectoryFrame]]:
    """Group consecutive frames sharing the same ammo value into runs."""

    runs: List[List[TrajectoryFrame]] = []
    for frame in trajectory.frames:
        ammo = frame.game_variables.get("ammo")
        if ammo is None:
            continue
        if runs and runs[-1][0].game_variables.get("ammo") == ammo:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def sample_ammo_change_keyframes(trajectory: Trajectory) -> List[TrajectoryFrame]:
    """Return one settled (middle) frame per distinct ammo run.

    Middle-of-run avoids the 1-tic HUD/state desync at run boundaries: the
    entry frame renders the old digit, so the middle frame is the safe choice
    where screen pixels and game_variables agree.
    """

    return [run[len(run) // 2] for run in _ammo_runs(trajectory)]
