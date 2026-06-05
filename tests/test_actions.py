"""Unit tests for the Phase 2 action library (no ViZDoom launch).

FakeActionEnv models the measured firing cadence: one ammo decrement per 14
ATTACK tics; noop never changes ammo. It also models health_gathering's
acid-floor cadence: health drops every 32 noop tics. This lets us verify that
FIRE_TICS=16 yields exactly one shot and that timing-aware composites use the
calibrated observation window before judging an effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from actions import ActionPrimitives, HEALTH_GATHERING_POLL_TICS, TestActions
from perception import GroundTruthPerceptor


@dataclass
class FakeState:
    step: int
    screen: np.ndarray
    game_variables: Dict[str, int]
    done: bool = False
    reward: float = 0.0


class FakeActionEnv:
    """ammo drops on ATTACK cadence; health drops on elapsed health ticks."""

    BUTTONS = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    TICS_PER_SHOT = 14
    HEALTH_DAMAGE_INTERVAL = 32
    HEALTH_DAMAGE_AMOUNT = 8

    def __init__(self, start_ammo: int = 26, start_health: int = 92) -> None:
        self.start_ammo = start_ammo
        self.start_health = start_health
        self.ammo = start_ammo
        self.health = start_health
        self._attack_tics = 0
        self._t = 0

    def get_button_names(self) -> List[str]:
        return list(self.BUTTONS)

    def _state(self, done: bool = False) -> FakeState:
        screen = np.full((4, 4, 3), min(self.ammo, 255), dtype=np.uint8)
        return FakeState(self._t, screen, {"ammo": self.ammo, "health": self.health}, done)

    def reset(self) -> FakeState:
        self.ammo = self.start_ammo
        self.health = self.start_health
        self._attack_tics = 0
        self._t = 0
        return self._state()

    def step(self, action: List[int]) -> FakeState:
        self._t += 1
        if len(action) > 2 and action[2] == 1:
            self._attack_tics += 1
            if self._attack_tics % self.TICS_PER_SHOT == 0:
                self.ammo = max(0, self.ammo - 1)
        self.health = max(
            0,
            self.start_health
            - (self._t // self.HEALTH_DAMAGE_INTERVAL) * self.HEALTH_DAMAGE_AMOUNT,
        )
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

    def test_calibrated_health_wait_observes_health_loss(self):
        prim = ActionPrimitives(FakeActionEnv())
        prim.reset()
        prim.wait(HEALTH_GATHERING_POLL_TICS)
        assert prim.observe().game_variables["health"] == 84

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

    def test_fire_and_check_ammo_reports_before_after(self):
        actions, gt = self._setup()
        result = actions.fire_and_check_ammo(gt)
        assert result["ammo_before"] == 26
        assert result["ammo_after"] == 25
        assert result["ammo_before"] - result["ammo_after"] == 1
        assert "delta" not in result  # canonical schema: before/after only

    def test_idle_and_check_ammo_reports_unchanged(self):
        actions, gt = self._setup()
        result = actions.idle_and_check_ammo(gt)
        assert result["ammo_before"] == 26
        assert result["ammo_after"] == 26

    def test_wait_and_check_health_reports_decrease(self):
        actions, gt = self._setup()
        result = actions.wait_and_check_health(gt)
        assert result["health_before"] == 92
        assert result["health_after"] == 84
        assert "health_delta" not in result  # canonical schema: before/after only

    def test_repeated_fire_reduces_by_three(self):
        actions, gt = self._setup()
        last = None
        for _ in range(3):
            last = actions.fire_and_check_ammo(gt)
        assert last["ammo_after"] == 23

    def test_run_dispatches_by_name(self):
        actions, gt = self._setup()
        result = actions.run("fire_and_check_ammo", gt)
        assert result["ammo_before"] - result["ammo_after"] == 1

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
            "wait_and_check_health",
        }


class TestExpectations:
    def test_fire_expectation_met_returns_no_anomaly(self):
        result = {"ammo_before": 26, "ammo_after": 25}
        assert TestActions.check_expectation("fire_and_check_ammo", result) is None

    def test_fire_expectation_violated_returns_anomaly(self):
        result = {"ammo_before": 26, "ammo_after": 26}
        anomaly = TestActions.check_expectation("fire_and_check_ammo", result)
        assert anomaly is not None
        assert anomaly["action"] == "fire_and_check_ammo"
        assert "decrease" in anomaly["expected"]
        assert anomaly["result"]["ammo_after"] == 26

    def test_idle_expectation_met_when_unchanged(self):
        result = {"ammo_before": 26, "ammo_after": 26}
        assert TestActions.check_expectation("idle_and_check_ammo", result) is None

    def test_idle_expectation_violated_when_ammo_drops(self):
        result = {"ammo_before": 26, "ammo_after": 25}
        anomaly = TestActions.check_expectation("idle_and_check_ammo", result)
        assert anomaly is not None
        assert anomaly["action"] == "idle_and_check_ammo"

    def test_health_expectation_met_when_decreased(self):
        result = {"health_before": 92, "health_after": 84}
        assert TestActions.check_expectation("wait_and_check_health", result) is None

    def test_health_expectation_violated_when_unchanged(self):
        result = {"health_before": 92, "health_after": 92}
        anomaly = TestActions.check_expectation("wait_and_check_health", result)
        assert anomaly is not None
        assert anomaly["action"] == "wait_and_check_health"

    def test_unknown_template_has_no_expectation(self):
        assert TestActions.check_expectation("nuke_everything", {"ammo_after": 99}) is None

    def test_none_read_is_violation_without_crashing(self):
        # a perception failure can produce a None read -> treated as violation
        result = {"ammo_before": 26, "ammo_after": None}
        anomaly = TestActions.check_expectation("fire_and_check_ammo", result)
        assert anomaly is not None
