"""Start ViZDoom through the project wrapper.

This script opens a visible ViZDoom window and doubles as the real wrapper
smoke command for Phase 0.2.
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
        print("Game initialized. Press Ctrl+C to stop.")
        print(f"Buttons: {env.get_button_names()}")

        try:
            for episode in range(5):
                state = env.reset()
                print(f"\n=== Episode {episode + 1} ===")

                while not state.done:
                    # basic.cfg action order: MOVE_LEFT, MOVE_RIGHT, ATTACK
                    state = env.step([0, 0, 1])
                    time.sleep(0.05)

                print(f"Episode {episode + 1} finished.")
        except KeyboardInterrupt:
            print("\nStopped by user.")

    print("\nDone.")


if __name__ == "__main__":
    main()
