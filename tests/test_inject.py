"""Unit tests for the Phase 3 failure injectors (no ViZDoom)."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from experiments.inject import ExecutionFailureInjector, PerceptionFailureInjector
from perception.base import GameState, GameStatePerceptor


class ScriptedPerceptor(GameStatePerceptor):
    """Returns a scripted ammo sequence (the TRUE values)."""

    def __init__(self, ammos: List[int]):
        self.ammos = ammos
        self.i = 0

    def perceive(self, screenshot: Any, **kwargs: Any) -> GameState:
        ammo = self.ammos[min(self.i, len(self.ammos) - 1)]
        self.i += 1
        return GameState(ammo=ammo, health=100)


class RecordingPrimitives:
    """Records which method ran so we can assert fire vs wait."""

    def __init__(self):
        self.calls: List[str] = []

    def fire_once(self):
        self.calls.append("fire")
        return "fired"

    def wait(self, tics: int = 0):
        self.calls.append(f"wait{tics}")
        return "waited"

    def observe(self):
        self.calls.append("observe")
        return "state"


class TestPerceptionInjector:
    def test_masks_first_ammo_change_then_truthful(self):
        # true sequence 26 (before), 25 (after, should be masked -> 26), 25, 24
        inj = PerceptionFailureInjector(ScriptedPerceptor([26, 25, 25, 24]), mask_changes=1)
        assert inj.perceive(None).ammo == 26   # before: truthful, sets prev=26
        assert inj.perceive(None).ammo == 26   # after: true 25 masked to 26 (delta looks 0)
        assert inj.perceive(None).ammo == 25   # budget spent: truthful again
        assert inj.perceive(None).ammo == 24   # truthful

    def test_records_true_ammo_in_raw_response(self):
        inj = PerceptionFailureInjector(ScriptedPerceptor([26, 25]), mask_changes=1)
        inj.perceive(None)
        masked = inj.perceive(None)
        assert masked.ammo == 26
        assert masked.raw_response["injected"] == "perception"
        assert masked.raw_response["true_ammo"] == 25

    def test_zero_budget_is_passthrough(self):
        inj = PerceptionFailureInjector(ScriptedPerceptor([26, 25, 24]), mask_changes=0)
        assert [inj.perceive(None).ammo for _ in range(3)] == [26, 25, 24]


class TestExecutionInjector:
    def test_first_fire_becomes_wait_then_real(self):
        prim = RecordingPrimitives()
        inj = ExecutionFailureInjector(prim, fail_fires=1)
        inj.fire_once()  # should NOT fire
        inj.fire_once()  # should fire
        assert prim.calls[0].startswith("wait")  # first was a no-op wait
        assert prim.calls[1] == "fire"

    def test_persistent_failure_never_fires(self):
        prim = RecordingPrimitives()
        inj = ExecutionFailureInjector(prim, fail_fires=99)
        for _ in range(3):
            inj.fire_once()
        assert "fire" not in prim.calls
        assert all(c.startswith("wait") for c in prim.calls)

    def test_delegates_other_methods(self):
        prim = RecordingPrimitives()
        inj = ExecutionFailureInjector(prim, fail_fires=1)
        inj.observe()
        assert prim.calls == ["observe"]
