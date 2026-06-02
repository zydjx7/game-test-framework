"""Unit tests for the Phase 2 action library (no ViZDoom launch).

FakeActionEnv models the measured firing cadence: one ammo decrement per 14
ATTACK tics; noop never changes ammo. This lets us verify that FIRE_TICS=16
yields exactly one shot and that composites compute before/after/delta and
wire the injected perceptor correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from actions import ActionPrimitives, TestActions
from perception import GroundTruthPerceptor


@dataclass
class FakeState:
    step: int
    screen: np.ndarray
    game_variables: Dict[str, int]
    done: bool = False
    reward: float = 0.0


class FakeActionEnv:
    """ammo drops by 1 every 14 ATTACK tics; noop leaves it unchanged."""

    BUTTONS = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    TICS_PER_SHOT = 14

    def __init__(self, start_ammo: int = 26) -> None:
        self.start_ammo = start_ammo
        self.ammo = start_ammo
        self._attack_tics = 0
        self._t = 0

    def get_button_names(self) -> List[str]:
        return list(self.BUTTONS)

    def _state(self, done: bool = False) -> FakeState:
        screen = np.full((4, 4, 3), min(self.ammo, 255), dtype=np.uint8)
        return FakeState(self._t, screen, {"ammo": self.ammo, "health": 100}, done)

    def reset(self) -> FakeState:
        self.ammo = self.start_ammo
        self._attack_tics = 0
        self._t = 0
        return self._state()

    def step(self, action: List[int]) -> FakeState:
        self._t += 1
        if len(action) > 2 and action[2] == 1:
            self._attack_tics += 1
            if self._attack_tics % self.TICS_PER_SHOT == 0:
                self.ammo = max(0, self.ammo - 1)
        return self._state()


class TestPrimitives:
    def test_fire_once_decrements_ammo_by_one(self):
        prim = ActionPrimitives(FakeActionEnv())
        prim.reset()
        prim.fire_once()
        assert prim.observe().game_variables["ammo"] == 25

    def test_wait_does_not_change_ammo(self):
        prim = ActionPrimitives(FakeActionEnv())
        prim.reset()
        prim.wait(20)
        assert prim.observe().game_variables["ammo"] == 26

    def test_four_tics_would_not_fire(self):
        # Guards the Stage 0 finding: a too-short fire shows no ammo change.
        prim = ActionPrimitives(FakeActionEnv())
        prim.reset()
        prim._hold("ATTACK", 4)
        assert prim.observe().game_variables["ammo"] == 26


class TestComposites:
    def _setup(self):
        prim = ActionPrimitives(FakeActionEnv())
        prim.reset()
        return TestActions(prim), GroundTruthPerceptor()

    def test_fire_and_check_ammo_reports_delta_one(self):
        actions, gt = self._setup()
        result = actions.fire_and_check_ammo(gt)
        assert result["ammo_before"] == 26
        assert result["ammo_after"] == 25
        assert result["delta"] == 1

    def test_idle_and_check_ammo_reports_zero_delta(self):
        actions, gt = self._setup()
        result = actions.idle_and_check_ammo(gt)
        assert result["ammo_before"] == 26
        assert result["ammo_after"] == 26
        assert result["delta"] == 0

    def test_repeated_fire_reduces_by_three(self):
        actions, gt = self._setup()
        last = None
        for _ in range(3):
            last = actions.fire_and_check_ammo(gt)
        assert last["ammo_after"] == 23

    def test_run_dispatches_by_name(self):
        actions, gt = self._setup()
        result = actions.run("fire_and_check_ammo", gt)
        assert result["delta"] == 1

    def test_run_rejects_unknown_template(self):
        actions, gt = self._setup()
        try:
            actions.run("nuke_everything", gt)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_list_templates(self):
        assert set(TestActions.list_templates()) == {
            "fire_and_check_ammo",
            "idle_and_check_ammo",
        }


class TestExpectations:
    def test_fire_expectation_met_returns_no_anomaly(self):
        assert TestActions.check_expectation("fire_and_check_ammo", {"delta": 1}) is None

    def test_fire_expectation_violated_returns_anomaly(self):
        anomaly = TestActions.check_expectation("fire_and_check_ammo", {"delta": 0})
        assert anomaly is not None
        assert anomaly["action"] == "fire_and_check_ammo"
        assert "decrease" in anomaly["expected"]
        assert anomaly["result"]["delta"] == 0

    def test_idle_expectation_met_when_unchanged(self):
        assert TestActions.check_expectation("idle_and_check_ammo", {"delta": 0}) is None

    def test_idle_expectation_violated_when_ammo_drops(self):
        anomaly = TestActions.check_expectation("idle_and_check_ammo", {"delta": 1})
        assert anomaly is not None
        assert anomaly["action"] == "idle_and_check_ammo"

    def test_unknown_template_has_no_expectation(self):
        assert TestActions.check_expectation("nuke_everything", {"delta": 99}) is None

    def test_none_delta_does_not_crash(self):
        # a perception failure can produce delta None -> treated as violation, no crash
        anomaly = TestActions.check_expectation("fire_and_check_ammo", {"delta": None})
        assert anomaly is not None
