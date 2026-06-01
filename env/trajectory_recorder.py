"""Record full ViZDoom episodes to disk for offline perception evaluation.

Why this exists (Phase 1 design doc §1.4 / §6):
- The perception evaluation must be *reproducible* and *cheap to re-run*.
  If we called the VLM live during a game episode, every prompt tweak would
  require replaying the whole game, and any randomness in the rollout would
  make two runs incomparable.
- So we split the pipeline: (1) record the full episode once, here; then
  (2) sample keyframes offline; then (3) feed those keyframes to the VLM.
  Stages 2-3 can re-run hundreds of times against the same recorded data.

Storage note:
- A raw 640x480x3 uint8 frame is ~0.9 MB. A 300-tick episode would be
  ~270 MB raw, and 5 episodes >1 GB. Doom frames compress extremely well,
  so :func:`save_trajectory` PNG-encodes each frame on disk (~30 KB/frame)
  and :func:`load_trajectory` decodes it back to a numpy array. In memory,
  ``TrajectoryFrame.screen`` is always a plain ``(H, W, 3)`` uint8 array;
  the compression is purely a serialization concern.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import numpy as np

# A policy maps the current environment state to a ViZDoom action vector.
# For the Phase 1 spike this is simply "always ATTACK" (see scripts/).
Policy = Callable[[Any], List[int]]


@dataclass
class TrajectoryFrame:
    """One recorded game tick.

    Attributes
    ----------
    tick:
        Step index within the episode (0-based, matches ``DoomState.step``).
    screen:
        ``(H, W, 3)`` uint8 image as returned by ``VizDoomEnv`` (already
        transposed out of ViZDoom's channels-first layout).
    game_variables:
        Ground-truth variables exposed by the scenario, e.g. ``{"ammo": 26}``.
        This is what :class:`GroundTruthPerceptor` reads to score the VLM.
    """

    tick: int
    screen: np.ndarray
    game_variables: Dict[str, int]


@dataclass
class Trajectory:
    """A full recorded episode plus provenance metadata."""

    scenario: str
    frames: List[TrajectoryFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)


def record_episode(
    env: Any,
    policy: Policy,
    *,
    scenario: str,
    max_tics: int = 2100,
    metadata: Dict[str, Any] | None = None,
) -> Trajectory:
    """Run one episode and capture every tick into a :class:`Trajectory`.

    Parameters
    ----------
    env:
        A ``VizDoomEnv``-like object exposing ``reset() -> state`` and
        ``step(action) -> state`` where ``state`` has ``.step``, ``.screen``,
        ``.game_variables`` and ``.done``.
    policy:
        Maps the current state to an action vector for the next step.
    scenario:
        Scenario name, stored for provenance (e.g. ``"basic"``).
    max_tics:
        Hard cap on recorded frames, a safety net against runaway episodes.
    metadata:
        Extra provenance to stash (policy name, timestamp, ...).

    Returns
    -------
    Trajectory
        Frames are recorded *before* each step, so the terminal degenerate
        state (ViZDoom returns a 1x1 zero screen once the episode is over)
        is never stored.
    """

    trajectory = Trajectory(scenario=scenario, metadata=dict(metadata or {}))

    state = env.reset()
    while not state.done and len(trajectory.frames) < max_tics:
        trajectory.frames.append(
            TrajectoryFrame(
                tick=int(state.step),
                screen=np.ascontiguousarray(state.screen).copy(),
                game_variables=dict(state.game_variables),
            )
        )
        action = policy(state)
        state = env.step(action)

    trajectory.metadata.setdefault("recorded_frames", len(trajectory.frames))
    return trajectory


def save_trajectory(trajectory: Trajectory, path: str | Path) -> Path:
    """Pickle a trajectory with PNG-compressed frames.

    Each frame's screen is encoded to PNG bytes so a multi-hundred-frame
    episode stays in the tens of MB instead of hundreds.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scenario": trajectory.scenario,
        "metadata": trajectory.metadata,
        "frames": [
            {
                "tick": frame.tick,
                "png": _encode_png(frame.screen),
                "vars": frame.game_variables,
            }
            for frame in trajectory.frames
        ],
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_trajectory(path: str | Path) -> Trajectory:
    """Inverse of :func:`save_trajectory`; decodes PNG frames back to arrays."""

    path = Path(path)
    with path.open("rb") as fh:
        payload = pickle.load(fh)

    frames = [
        TrajectoryFrame(
            tick=int(item["tick"]),
            screen=_decode_png(item["png"]),
            game_variables=dict(item["vars"]),
        )
        for item in payload["frames"]
    ]
    return Trajectory(
        scenario=payload["scenario"],
        frames=frames,
        metadata=dict(payload.get("metadata", {})),
    )


def _encode_png(screen: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", screen)
    if not ok:
        raise ValueError("cv2.imencode failed to encode a trajectory frame")
    return buffer.tobytes()


def _decode_png(buffer: bytes) -> np.ndarray:
    array = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError("cv2.imdecode failed to decode a trajectory frame")
    return array
