"""Unit tests for agent.graph (reflective LangGraph) — no API, no ViZDoom.

A ScriptedActionLib returns preset step results by call count, so we can drive
the anomaly -> reflect -> recover branches deterministically with a FakeDecider
and a FakeReflector.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from actions import TestActions
from agent.goal import parse_goals
from agent.graph import run_reflective_agent
from agent.reflection import RECOVERY, FailureType, ReflectionCase

FIRE_GOAL = parse_goals(
    """
Scenario: fire
  Goal: Fire and consume ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 1
"""
)[0]

IDLE_GOAL = parse_goals(
    """
Scenario: idle
  Goal: Stay idle.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after == 0 and steps >= 1
"""
)[0]


class ScriptedActionLib:
    DESCRIPTIONS = TestActions.DESCRIPTIONS

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.i = 0
        self.prim = SimpleNamespace(reset=lambda: None)

    def run(self, name: str, perceptor: Any) -> Dict[str, Any]:
        r = self.results[min(self.i, len(self.results) - 1)]
        self.i += 1
        return dict(r)

    def check_expectation(self, name: str, result: Dict[str, Any]):
        return TestActions.check_expectation(name, result)


class FakeDecider:
    def __init__(self, action: str):
        self.action = action

    def decide(self, goal_description, history, tools_spec) -> str:
        return self.action


class FakeReflector:
    def __init__(self, failure_type: FailureType):
        self.failure_type = failure_type

    def reflect(self, anomaly, history) -> ReflectionCase:
        return ReflectionCase(
            anomaly=anomaly,
            failure_type=self.failure_type,
            recovery_action=RECOVERY[self.failure_type],
            confidence=0.9,
            reasoning="fake",
        )


def _run(goal, results, action, failure_type=FailureType.EXECUTION, **kw):
    return run_reflective_agent(
        goal,
        ScriptedActionLib(results),
        perceptor=None,
        decider=FakeDecider(action),
        reflector=FakeReflector(failure_type),
        **kw,
    )


class TestReflectiveGraph:
    def test_clean_fire_succeeds_without_reflection(self):
        out = _run(FIRE_GOAL, [{"ammo_before": 26, "ammo_after": 25, "delta": 1}], "fire_and_check_ammo")
        assert out["status"] == "success"
        assert out["cases"] == []  # no anomaly -> no reflection

    def test_clean_idle_succeeds(self):
        out = _run(IDLE_GOAL, [{"ammo_before": 26, "ammo_after": 26, "delta": 0}], "idle_and_check_ammo")
        assert out["status"] == "success"

    def test_execution_failure_then_recovered(self):
        # first fire does nothing (anomaly), reflect=execution -> retry -> redo succeeds
        results = [
            {"ammo_before": 26, "ammo_after": 26, "delta": 0},
            {"ammo_before": 26, "ammo_after": 25, "delta": 1},
        ]
        out = _run(FIRE_GOAL, results, "fire_and_check_ammo", FailureType.EXECUTION)
        assert out["status"] == "success"
        assert len(out["cases"]) == 1
        assert out["cases"][0]["failure_type"] == "execution"
        assert out["cases"][0]["recovered"] is True

    def test_logic_failure_is_reported_not_retried(self):
        results = [{"ammo_before": 26, "ammo_after": 26, "delta": 0}]
        out = _run(FIRE_GOAL, results, "fire_and_check_ammo", FailureType.LOGIC)
        assert out["status"] == "bug_reported"
        assert out["bug_report"] is not None
        assert out["cases"][0]["failure_type"] == "logic"

    def test_recovery_capped_then_maxsteps(self):
        # fire never works; one reflection+redo allowed, then it should not loop forever
        results = [{"ammo_before": 26, "ammo_after": 26, "delta": 0}]
        out = _run(
            FIRE_GOAL, results, "fire_and_check_ammo",
            FailureType.EXECUTION, max_steps=4, max_recoveries=1,
        )
        assert out["status"] == "max_steps_exceeded"
        assert len(out["cases"]) == 1  # reflected once, capped
