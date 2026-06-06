"""Unit tests for agent.graph (reflective LangGraph) — no API, no ViZDoom.

A ScriptedActionLib returns preset `run()` results by call count and preset
`observe()` ammo readings by call count, so we can drive the diagnostic recovery
ladder (re_observe -> retry -> report) deterministically with a FakeDecider and
a FakeReflector.

Key property under test (Stage-1 step 3): the recovery OUTCOME disambiguates the
failure. A perception fault recovers at re_observe (a fresh read sees the truth,
no extra action); an execution fault survives re_observe and needs retry; a
persistent fault survives both and is reported as suspected logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.goal import parse_goals
from agent.graph import run_reflective_agent
from agent.reflection import RECOVERY, FailureType, ReflectionCase
from actions import TestActions

FIRE_GOAL = parse_goals(
    """
Scenario: fire
  Goal: Fire and consume ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 1
"""
)[0]

FIRE_TWICE_GOAL = parse_goals(
    """
Scenario: fire twice
  Goal: Fire enough to consume two ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 2
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
    """run() returns preset results by call count; observe() returns preset ammo
    readings by call count. Both clamp to the last element when exhausted."""

    DESCRIPTIONS = TestActions.DESCRIPTIONS

    def __init__(self, results: List[Dict[str, Any]], obs_ammo: Optional[List[Optional[int]]] = None):
        self.results = results
        self.obs_ammo = obs_ammo or []
        self.run_calls = 0
        self.observe_calls = 0
        self.prim = SimpleNamespace(reset=lambda: None)

    def run(self, name: str, perceptor: Any) -> Dict[str, Any]:
        r = self.results[min(self.run_calls, len(self.results) - 1)]
        self.run_calls += 1
        return dict(r)

    def observe(self, perceptor: Any):
        if self.obs_ammo:
            v = self.obs_ammo[min(self.observe_calls, len(self.obs_ammo) - 1)]
        else:
            v = None
        self.observe_calls += 1
        return SimpleNamespace(ammo=v, health=None, score=None)

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


def _run(goal, results, action, failure_type=FailureType.EXECUTION, obs_ammo=None, **kw):
    lib = ScriptedActionLib(results, obs_ammo=obs_ammo)
    out = run_reflective_agent(
        goal,
        lib,
        perceptor=None,
        decider=FakeDecider(action),
        reflector=FakeReflector(failure_type),
        **kw,
    )
    return out, lib


def _recoveries(out) -> List[str]:
    return [h["recovery"] for h in out["history"] if "recovery" in h]


class TestReflectiveGraph:
    def test_clean_fire_succeeds_without_reflection(self):
        out, lib = _run(FIRE_GOAL, [{"ammo_before": 26, "ammo_after": 25}], "fire_and_check_ammo")
        assert out["status"] == "success"
        assert out["cases"] == []  # no anomaly -> no reflection
        assert lib.run_calls == 1
        assert _recoveries(out) == []

    def test_clean_idle_succeeds(self):
        out, _ = _run(IDLE_GOAL, [{"ammo_before": 26, "ammo_after": 26}], "idle_and_check_ammo")
        assert out["status"] == "success"

    def test_perception_fault_recovers_at_re_observe_without_acting(self):
        # The shot DID fire (perception masked it -> looked like delta 0). A fresh
        # read sees the true 25. re_observe alone recovers; NO extra run() call,
        # and `steps` is not inflated by the re-read.
        out, lib = _run(
            FIRE_GOAL,
            results=[{"ammo_before": 26, "ammo_after": 26}],
            action="fire_and_check_ammo",
            failure_type=FailureType.PERCEPTION,
            obs_ammo=[25],
        )
        assert out["status"] == "success"
        assert lib.run_calls == 1           # the action was NOT re-run
        assert lib.observe_calls == 1       # exactly one re-observation
        assert _recoveries(out) == ["re_observe"]
        assert out["steps"] == 1            # re_observe did not advance steps
        assert len(out["cases"]) == 1 and out["cases"][0]["recovered"] is True

    def test_execution_fault_survives_re_observe_then_recovers_at_retry(self):
        # The shot did NOT fire -> re-reading still sees 26 (re_observe fails),
        # so the ladder escalates to retry, which actually fires (-> 25).
        out, lib = _run(
            FIRE_GOAL,
            results=[{"ammo_before": 26, "ammo_after": 26}, {"ammo_before": 26, "ammo_after": 25}],
            action="fire_and_check_ammo",
            failure_type=FailureType.EXECUTION,
            obs_ammo=[26],  # re-read still stale -> re_observe cannot fix it
        )
        assert out["status"] == "success"
        assert lib.run_calls == 2           # act + one retry
        assert _recoveries(out) == ["re_observe", "retry"]
        assert len(out["cases"]) == 2       # reflected before re_observe AND before retry
        assert out["cases"][-1]["recovered"] is True

    def test_logic_diagnosis_does_not_skip_recovery(self):
        # Even when the LLM diagnoses logic, the ladder still runs: a fault that
        # is actually recoverable (perception-style: a fresh read sees 25) is
        # recovered, OVERRIDING the wrong logic guess. A bug is never reported
        # before the cheap recoveries have been tried.
        out, lib = _run(
            FIRE_GOAL,
            results=[{"ammo_before": 26, "ammo_after": 26}],
            action="fire_and_check_ammo",
            failure_type=FailureType.LOGIC,
            obs_ammo=[25],  # re-read reveals the true 25 -> re_observe recovers
        )
        assert out["status"] == "success"
        assert _recoveries(out) == ["re_observe"]
        assert lib.run_calls == 1
        assert out["cases"][0]["failure_type"] == "logic"  # diagnosis still recorded
        assert out["cases"][0]["recovered"] is True

    def test_persistent_fault_exhausts_ladder_then_reports(self):
        # Nothing ever fixes it: re_observe stale, retry still 26. The ladder
        # exhausts (1 re_observe + 1 retry) and escalates to SUSPECTED logic
        # instead of silently timing out. The LLM's logic diagnosis is recorded
        # as corroborating evidence but is not what triggers the report.
        out, lib = _run(
            FIRE_GOAL,
            results=[{"ammo_before": 26, "ammo_after": 26}],
            action="fire_and_check_ammo",
            failure_type=FailureType.LOGIC,
            obs_ammo=[26],
            max_steps=4,
        )
        assert out["status"] == "bug_reported"
        assert out["bug_report"]["suspected_logic"] is True
        assert out["bug_report"]["evidence"]["budgets_exhausted"] is True
        assert out["bug_report"]["evidence"]["diagnosed_logic_by_llm"] is True
        assert lib.run_calls == 2           # act + retry (re_observe does not run())
        assert _recoveries(out) == ["re_observe", "retry"]

    def test_budget_resets_per_anomaly(self):
        # Two separate execution faults, each recovered by retry. Without the
        # per-anomaly reset the second anomaly could not recover (the old global
        # counter would already be exhausted). Goal needs delta >= 2.
        out, lib = _run(
            FIRE_TWICE_GOAL,
            results=[
                {"ammo_before": 26, "ammo_after": 26},  # act 1: exec fault
                {"ammo_before": 26, "ammo_after": 25},  # retry 1: fixed (26->25)
                {"ammo_before": 25, "ammo_after": 25},  # act 2: exec fault
                {"ammo_before": 25, "ammo_after": 24},  # retry 2: fixed (25->24)
            ],
            action="fire_and_check_ammo",
            failure_type=FailureType.EXECUTION,
            obs_ammo=[26, 25],  # both re_observes fail -> each escalates to retry
            max_steps=8,
        )
        assert out["status"] == "success"
        assert out["cumulative"]["ammo_before"] - out["cumulative"]["ammo_after"] >= 2
        assert _recoveries(out) == ["re_observe", "retry", "re_observe", "retry"]
        # 4 cases = 2 per anomaly (failed re_observe rung, then the retry rung).
        # The retry rung of BOTH anomalies recovered -> the 2nd anomaly was still
        # recoverable, which only holds because the budget reset per-anomaly.
        assert len(out["cases"]) == 4
        assert out["cases"][1]["recovered"] is True  # 1st anomaly recovered at retry
        assert out["cases"][3]["recovered"] is True  # 2nd anomaly ALSO recovered
