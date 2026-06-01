"""Unit tests for the Phase 0.2 ViZDoom environment wrapper.

These tests use a fake ``vizdoom`` module, so they do not require a real
ViZDoom install or a graphical window.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from env import DoomState, VizDoomEnv


class FakeButton:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeRawState:
    def __init__(self, ammo: int = 7, health: int = 92) -> None:
        self.screen_buffer = np.arange(3 * 2 * 4, dtype=np.uint8).reshape(3, 2, 4)
        self.game_variables = [ammo, health]


class FakeScreenResolution:
    RES_640X480 = "RES_640X480"


class FakeDoomGame:
    instances: list["FakeDoomGame"] = []

    def __init__(self) -> None:
        self.loaded_config: str | None = None
        self.window_visible: bool | None = None
        self.screen_resolution: str | None = None
        self.render_hud: bool | None = None
        self.initialized = False
        self.new_episode_calls = 0
        self.actions: list[list[int]] = []
        self.closed = False
        FakeDoomGame.instances.append(self)

    def load_config(self, path: str) -> None:
        self.loaded_config = path

    def set_window_visible(self, visible: bool) -> None:
        self.window_visible = visible

    def set_screen_resolution(self, resolution: str) -> None:
        self.screen_resolution = resolution

    def set_render_hud(self, render_hud: bool) -> None:
        self.render_hud = render_hud

    def init(self) -> None:
        self.initialized = True

    def new_episode(self) -> None:
        self.new_episode_calls += 1

    def make_action(self, action: list[int]) -> float:
        self.actions.append(action)
        return 1.5

    def get_state(self) -> FakeRawState:
        return FakeRawState()

    def is_episode_finished(self) -> bool:
        return False

    def get_available_buttons(self) -> list[FakeButton]:
        return [FakeButton("MOVE_LEFT"), FakeButton("MOVE_RIGHT"), FakeButton("ATTACK")]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_vizdoom(monkeypatch: pytest.MonkeyPatch, tmp_path):
    FakeDoomGame.instances = []
    (tmp_path / "basic.cfg").write_text("# fake basic scenario\n", encoding="utf-8")
    (tmp_path / "custom.cfg").write_text("# fake custom scenario\n", encoding="utf-8")

    fake_module = SimpleNamespace(
        DoomGame=FakeDoomGame,
        ScreenResolution=FakeScreenResolution,
        scenarios_path=str(tmp_path),
    )
    monkeypatch.setitem(sys.modules, "vizdoom", fake_module)
    return fake_module


def test_doom_state_stores_structured_values():
    screen = np.zeros((2, 4, 3), dtype=np.uint8)

    state = DoomState(
        step=3,
        screen=screen,
        game_variables={"ammo": 5},
        done=False,
        reward=1.25,
    )

    assert state.step == 3
    assert state.screen is screen
    assert state.game_variables == {"ammo": 5}
    assert state.done is False
    assert state.reward == 1.25


def test_env_package_reexports_public_wrapper_classes():
    from env import DoomState as ReexportedDoomState
    from env import VizDoomEnv as ReexportedVizDoomEnv

    assert ReexportedDoomState is DoomState
    assert ReexportedVizDoomEnv is VizDoomEnv


def test_reset_builds_state_and_transposes_screen(fake_vizdoom):
    env = VizDoomEnv(scenario="basic", window_visible=False)

    state = env.reset()

    assert state.step == 0
    assert state.screen.shape == (2, 4, 3)
    assert state.game_variables == {"ammo": 7}
    assert state.done is False
    assert state.reward == 0.0

    game = FakeDoomGame.instances[-1]
    assert game.loaded_config.endswith("basic.cfg")
    assert game.window_visible is False
    assert game.screen_resolution == "RES_640X480"
    assert game.render_hud is False  # default off; perception capture opts in
    assert game.initialized is True
    assert game.new_episode_calls == 1


def test_render_hud_opt_in_is_forwarded(fake_vizdoom):
    VizDoomEnv(scenario="basic", window_visible=False, render_hud=True)

    assert FakeDoomGame.instances[-1].render_hud is True


def test_step_returns_reward_and_increments_counter(fake_vizdoom):
    env = VizDoomEnv(scenario="basic", window_visible=False)

    state = env.step([0, 0, 1])

    assert state.step == 1
    assert state.reward == 1.5
    assert state.game_variables == {"ammo": 7}
    assert FakeDoomGame.instances[-1].actions == [[0, 0, 1]]


def test_unknown_scenario_runs_with_empty_game_variables(fake_vizdoom):
    env = VizDoomEnv(scenario="custom", window_visible=False)

    state = env.reset()

    assert state.game_variables == {}


def test_button_names_and_close_delegate_to_vizdoom(fake_vizdoom):
    env = VizDoomEnv(scenario="basic", window_visible=False)

    assert env.get_button_names() == ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
    env.close()

    assert FakeDoomGame.instances[-1].closed is True
