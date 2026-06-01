"""Record Phase 1 spike trajectories on basic.wad (full-ATTACK policy).

Usage (from project root, with .venv active):

    python scripts/record_basic_trajectories.py
    python scripts/record_basic_trajectories.py --episodes 5 --max-tics 120

Output: data/trajectories/basic_<timestamp>_ep<N>.pkl (gitignored).

Design doc reference: Doc/phase1-design.md §6. The policy is constant ATTACK
so ammo decreases monotonically from 26 toward 0, naturally covering the
high / medium / low ammo levels we want the VLM to read.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import VizDoomEnv  # noqa: E402
from env.trajectory_recorder import record_episode, save_trajectory  # noqa: E402

# basic.cfg button order: MOVE_LEFT, MOVE_RIGHT, ATTACK
ALWAYS_ATTACK = [0, 0, 1]


def always_attack(_state) -> list[int]:
    return ALWAYS_ATTACK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--max-tics",
        type=int,
        default=120,
        help="Cap recorded frames per episode (basic.wad runs to ~300 otherwise).",
    )
    parser.add_argument("--out-dir", type=str, default="data/trajectories")
    parser.add_argument(
        "--window",
        action="store_true",
        help="Show the ViZDoom window while recording (slower).",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with VizDoomEnv(scenario="basic", window_visible=args.window) as env:
        print(f"Buttons: {env.get_button_names()}")
        for episode in range(args.episodes):
            trajectory = record_episode(
                env,
                always_attack,
                scenario="basic",
                max_tics=args.max_tics,
                metadata={
                    "policy": "always_attack",
                    "episode": episode,
                    "timestamp": stamp,
                },
            )
            path = out_dir / f"basic_{stamp}_ep{episode}.pkl"
            save_trajectory(trajectory, path)

            ammo_first = trajectory.frames[0].game_variables.get("ammo") if trajectory.frames else None
            ammo_last = trajectory.frames[-1].game_variables.get("ammo") if trajectory.frames else None
            print(
                f"ep{episode}: {len(trajectory)} frames, "
                f"ammo {ammo_first} -> {ammo_last}, saved {path.name}"
            )

    print(f"\nDone. Trajectories in {out_dir}")


if __name__ == "__main__":
    main()
