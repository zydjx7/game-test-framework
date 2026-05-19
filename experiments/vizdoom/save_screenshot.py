"""Save a few ViZDoom screenshots through the project wrapper.

This script opens a visible ViZDoom window and writes local files under
``experiments/vizdoom/screenshots/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import VizDoomEnv  # noqa: E402


SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with VizDoomEnv(scenario="basic", window_visible=True) as env:
        state = env.reset()

        for step in range(5):
            if state.done:
                break

            ammo = state.game_variables.get("ammo", -1)
            filename = SCREENSHOT_DIR / f"step_{step:03d}_ammo_{ammo:03d}.png"
            Image.fromarray(state.screen).save(filename)
            print(f"Saved {filename}")

            state = env.step([0, 0, 1])


if __name__ == "__main__":
    main()
