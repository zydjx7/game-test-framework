"""Generate a short ViZDoom trajectory dataset.

This script opens a ViZDoom window and writes local experiment artifacts:

- ``experiments/vizdoom/trajectory_001/trajectory.csv``
- ``experiments/vizdoom/trajectory_001/screenshots/step_000.png`` ...
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import VizDoomEnv  # noqa: E402


TRAJ_DIR = Path(__file__).resolve().parent / "trajectory_001"
SCREENSHOT_DIR = TRAJ_DIR / "screenshots"


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    action_script = [
        [0, 0, 1],  # attack
        [0, 0, 1],  # attack
        [1, 0, 0],  # move left
        [0, 0, 1],  # attack
        [0, 1, 0],  # move right
    ]

    with VizDoomEnv(scenario="basic", window_visible=True) as env:
        state = env.reset()

        for step in range(50):
            if state.done:
                print(f"Episode ended early at step {step}")
                break

            action = action_script[step % len(action_script)]
            screenshot_path = SCREENSHOT_DIR / f"step_{step:03d}.png"
            Image.fromarray(state.screen).save(screenshot_path)

            rows.append(
                {
                    "step": step,
                    "ammo_gt": state.game_variables.get("ammo", -1),
                    "action": str(action),
                    "reward": state.reward,
                    "screenshot": str(screenshot_path.relative_to(TRAJ_DIR.parent)),
                }
            )

            state = env.step(action)

    csv_path = TRAJ_DIR / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "ammo_gt", "action", "reward", "screenshot"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Trajectory saved to {TRAJ_DIR} ===")
    print(f"  {len(rows)} steps")
    print(f"  CSV: {csv_path}")
    print(f"  Screenshots: {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
