"""Event-driven keyframe sampling on ammo change.

Why event-driven (Phase 1 design doc §0 / §6): the pistol fires once per
~14 tics, so ammo is constant for long stretches. Equidistant sampling would
waste VLM calls on duplicate ammo values. Sampling one frame per *change* in
the ammo value gives roughly one keyframe per distinct ammo count, which is
exactly what we want to probe the VLM's digit reading across the range.
"""

from __future__ import annotations

from typing import List

from env.trajectory_recorder import Trajectory, TrajectoryFrame

_SENTINEL = object()


def sample_ammo_change_keyframes(trajectory: Trajectory) -> List[TrajectoryFrame]:
    """Return the first frame at each new ammo value (in order).

    Includes the very first frame, then every frame whose ammo differs from
    the previously kept ammo value. Frames with no ammo variable are skipped.
    """

    keyframes: List[TrajectoryFrame] = []
    previous = _SENTINEL

    for frame in trajectory.frames:
        ammo = frame.game_variables.get("ammo")
        if ammo is None:
            continue
        if ammo != previous:
            keyframes.append(frame)
            previous = ammo

    return keyframes
