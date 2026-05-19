"""Small ViZDoom environment wrapper for the Phase 0.2 main line.

This module gives later perception, action, and agent-loop code a stable
interface instead of exposing raw ViZDoom calls everywhere.

Scenarios not listed in ``VARIABLE_NAMES`` still run, but their
``game_variables`` dict will be empty. Add new scenarios to that mapping when
their ground-truth variables become part of the research pipeline.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _load_vizdoom() -> Any:
    try:
        return importlib.import_module("vizdoom")
    except ImportError as exc:
        raise ImportError(
            "vizdoom 未安装在当前 Python 环境。请先激活项目虚拟环境 "
            "`.venv\\Scripts\\Activate.ps1`，然后运行 "
            "`pip install -r requirements.txt`。"
        ) from exc


@dataclass
class DoomState:
    """Structured state returned by ``VizDoomEnv``."""

    step: int
    screen: np.ndarray
    game_variables: dict[str, int]
    done: bool
    reward: float = 0.0
    raw_state: Any = field(default=None, repr=False)


class VizDoomEnv:
    """Thin wrapper around ViZDoom's ``DoomGame`` API."""

    VARIABLE_NAMES = {
        "basic": ["ammo"],
        "defend_the_center": ["ammo", "health"],
        "health_gathering": ["health"],
        "deadly_corridor": ["health"],
    }

    def __init__(
        self,
        scenario: str = "basic",
        window_visible: bool = True,
        resolution: str = "RES_640X480",
    ) -> None:
        self.scenario = scenario
        self.vzd = _load_vizdoom()
        self.game = self.vzd.DoomGame()

        cfg_path = os.path.join(self.vzd.scenarios_path, f"{scenario}.cfg")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Scenario {scenario}.cfg not found at {cfg_path}")

        self.game.load_config(cfg_path)
        self.game.set_window_visible(window_visible)
        self.game.set_screen_resolution(getattr(self.vzd.ScreenResolution, resolution))
        self.game.init()

        self.step_count = 0
        self._var_names = self.VARIABLE_NAMES.get(scenario, [])

    def reset(self) -> DoomState:
        """Start a new episode and return its initial state."""

        self.game.new_episode()
        self.step_count = 0
        return self._build_state(reward=0.0)

    def step(self, action: list[int]) -> DoomState:
        """Execute one action and return the resulting state."""

        reward = float(self.game.make_action(action))
        self.step_count += 1
        return self._build_state(reward=reward)

    def _build_state(self, reward: float) -> DoomState:
        raw = self.game.get_state()
        done = self.game.is_episode_finished()

        if raw is None:
            return DoomState(
                step=self.step_count,
                screen=np.zeros((1, 1, 3), dtype=np.uint8),
                game_variables={},
                done=True,
                reward=reward,
                raw_state=None,
            )

        screen = np.transpose(raw.screen_buffer, (1, 2, 0))

        vars_dict: dict[str, int] = {}
        for index, name in enumerate(self._var_names):
            if index < len(raw.game_variables):
                vars_dict[name] = int(raw.game_variables[index])

        return DoomState(
            step=self.step_count,
            screen=screen,
            game_variables=vars_dict,
            done=done,
            reward=reward,
            raw_state=raw,
        )

    def get_button_names(self) -> list[str]:
        """Return the buttons available in the current scenario."""

        return [button.name for button in self.game.get_available_buttons()]

    def close(self) -> None:
        """Close the underlying ViZDoom game instance."""

        self.game.close()

    def __enter__(self) -> "VizDoomEnv":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


if __name__ == "__main__":
    with VizDoomEnv(scenario="basic") as env:
        print(f"Buttons: {env.get_button_names()}")
        state = env.reset()

        for _ in range(10):
            if state.done:
                break
            print(f"step={state.step} vars={state.game_variables} reward={state.reward}")
            state = env.step([0, 0, 1])
