"""Record Phase 1 spike trajectories (default scenario: defend_the_center).

Usage (from project root, with .venv active):

    python scripts/record_spike_trajectories.py
    python scripts/record_spike_trajectories.py --episodes 5 --max-tics 300
    python scripts/record_spike_trajectories.py --scenario basic

Output: data/trajectories/<scenario>_<timestamp>_ep<N>.pkl (gitignored).

Design doc reference: Doc/phase1-design.md §0 + §6.

Why defend_the_center, not basic: real-ViZDoom smoke (2026-06-01) showed
basic.wad ammo only varies in [46,50] (pistol fires every ~14 tics and the
episode ends on the single kill), so it cannot exercise the medium/low ammo
levels. defend_the_center under a constant-ATTACK policy drives ammo 26 -> 7
(high/medium/low all covered) and exposes health too.

The policy is constant ATTACK. ATTACK is button index 2 in both basic
('MOVE_LEFT','MOVE_RIGHT','ATTACK') and defend_the_center
('TURN_LEFT','TURN_RIGHT','ATTACK'), so [0, 0, 1] works for both.
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

# Constant ATTACK. ATTACK is index 2 in both basic and defend_the_center.
ALWAYS_ATTACK = [0, 0, 1]


def always_attack(_state) -> list[int]:
    return ALWAYS_ATTACK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=str, default="defend_the_center")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--max-tics",
        type=int,
        default=300,
        help="Cap recorded frames per episode (defend_the_center ends on death ~270).",
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

    # render_hud=True so the ammo/health digits are drawn for the VLM to read.
    with VizDoomEnv(
        scenario=args.scenario, window_visible=args.window, render_hud=True
    ) as env:
        button_names = env.get_button_names()
        print(f"Scenario: {args.scenario} | Buttons: {button_names}")
        if len(button_names) < 3 or button_names[2] != "ATTACK":
            print("  WARNING: button index 2 is not ATTACK; check the policy vector.")

        for episode in range(args.episodes):
            trajectory = record_episode(
                env,
                always_attack,
                scenario=args.scenario,
                max_tics=args.max_tics,
                metadata={
                    "policy": "always_attack",
                    "episode": episode,
                    "timestamp": stamp,
                },
            )
            path = out_dir / f"{args.scenario}_{stamp}_ep{episode}.pkl"
            save_trajectory(trajectory, path)

            first = trajectory.frames[0].game_variables if trajectory.frames else {}
            last = trajectory.frames[-1].game_variables if trajectory.frames else {}
            print(
                f"ep{episode}: {len(trajectory)} frames, "
                f"ammo {first.get('ammo')}->{last.get('ammo')}, "
                f"health {first.get('health')}->{last.get('health')}, "
                f"saved {path.name}"
            )

    print(f"\nDone. Trajectories in {out_dir}")


if __name__ == "__main__":
    main()
