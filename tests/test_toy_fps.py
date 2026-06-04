"""ToyFPS portability tests: the SAME agent layer runs on a second game.

These use the real `run_reflective_agent` / `run_agent_loop` (the agent layer)
against the ToyFPS adapter, with a scripted FakeDecider/FakeReflector so no API
is hit. They prove the generalized schema handles non-ammo metrics (score
increases, health increases) and that the agent layer is unchanged across games.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent import parse_goals, run_agent_loop, run_reflective_agent
from agent.reflection import RECOVERY, FailureType, ReflectionCase
from toy_fps import ToyActions, ToyFPS, ToyPerceptor, ToyPrimitives

GOALS = {g.metadata["name"]: g for g in parse_goals(
    (Path(__file__).resolve().parents[1] / "toy_fps" / "goals.feature").read_text(encoding="utf-8")
)}


class FakeDecider:
    def __init__(self, action: str):
        self.action = action

    def decide(self, *args, **kwargs) -> str:
        return self.action


class FakeReflector:
    def reflect(self, anomaly, history) -> ReflectionCase:
        ft = FailureType.EXECUTION
        return ReflectionCase(anomaly, ft, RECOVERY[ft], 0.9, "fake")


def _setup():
    return ToyActions(ToyPrimitives(ToyFPS())), ToyPerceptor()


def _run(goal_name: str, action: str, **kw):
    lib, perc = _setup()
    return run_reflective_agent(
        GOALS[goal_name], lib, perc, FakeDecider(action), FakeReflector(), **kw
    )


class TestToyFPSPortability:
    def test_ammo_goal_decreasing_metric(self):
        out = _run("Firing consumes ammo", "fire_and_check_ammo")
        assert out["status"] == "success"
        cum = out["cumulative"]
        assert cum["ammo_before"] - cum["ammo_after"] >= 1

    def test_score_goal_increasing_metric(self):
        out = _run("Firing increases score", "fire_and_check_score")
        assert out["status"] == "success"
        cum = out["cumulative"]
        assert cum["score_after"] - cum["score_before"] >= 1

    def test_health_goal_increasing_metric(self):
        out = _run("Healing restores health", "heal_and_check_health")
        assert out["status"] == "success"
        cum = out["cumulative"]
        assert cum["health_after"] - cum["health_before"] >= 5

    def test_same_run_agent_loop_runs_on_toy(self):
        # the Phase 2 while loop (no reflection) also runs unchanged on ToyFPS
        lib, perc = _setup()
        out = run_agent_loop(GOALS["Firing consumes ammo"], lib, perc, FakeDecider("fire_and_check_ammo"))
        assert out["status"] == "success"


class TestToyPerceptorReadsScore:
    def test_perceptor_reads_all_three_metrics(self):
        game = ToyFPS()
        state = game.reset()
        gs = ToyPerceptor().perceive(None, game_variables=state.game_variables)
        assert gs.ammo == 10 and gs.health == 50 and gs.score == 0

    def test_screenless_state_does_not_crash(self):
        # the Observation contract: ToyFPS has no screen
        lib, perc = _setup()
        lib.prim.reset()
        result = lib.run("fire_and_check_ammo", perc)
        assert result["ammo_before"] == 10 and result["ammo_after"] == 9
