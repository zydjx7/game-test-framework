"""Read ViZDoom ground-truth state through the project wrapper.

This script opens a visible ViZDoom window and prints ammo plus screen shape.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import VizDoomEnv  # noqa: E402


def main() -> None:
    with VizDoomEnv(scenario="basic", window_visible=True) as env:
        state = env.reset()
        print(f"{'step':>4} | {'ammo':>5} | {'screen_shape':>16} | {'action':>10}")
        print("-" * 50)

        for _ in range(20):
            if state.done:
                print(f"Episode ended at step {state.step}")
                break

            action = [0, 0, 1]
            print(
                f"{state.step:>4} | "
                f"{state.game_variables.get('ammo', -1):>5} | "
                f"{str(state.screen.shape):>16} | "
                f"{str(action):>10}"
            )
            state = env.step(action)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
