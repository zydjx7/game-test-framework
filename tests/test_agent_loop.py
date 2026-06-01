"""Unit tests for agent.loop (agent loop + function-calling decider).

Uses the FakeActionEnv firing model + a scripted FakeDecider, and a FakeClient
to test FunctionCallingDecider without hitting DeepSeek.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from actions import ActionPrimitives, TestActions
from agent.goal import parse_goals
from agent.loop import FunctionCallingDecider, run_agent_loop
from perception import GroundTruthPerceptor
from tests.test_actions import FakeActionEnv

GOALS = parse_goals(
    """
Scenario: Firing consumes ammo
  Goal: Fire and consume ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 1

Scenario: Idle does not consume ammo
  Goal: Stay idle.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after == 0 and steps >= 1

Scenario: Repeated firing reduces ammo by three
  Goal: Fire until ammo drops by three.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 3
"""
)
GOAL_FIRE, GOAL_IDLE, GOAL_FIRE3 = GOALS


class FakeDecider:
    """Always returns the same action (scripted policy)."""

    def __init__(self, action: str) -> None:
        self.action = action
        self.calls = 0

    def decide(self, goal_description, history, tools_spec) -> str:
        self.calls += 1
        return self.action


def _lib():
    prim = ActionPrimitives(FakeActionEnv())
    return TestActions(prim), GroundTruthPerceptor()


class TestAgentLoop:
    def test_fire_goal_succeeds_in_one_step(self):
        lib, gt = _lib()
        out = run_agent_loop(GOAL_FIRE, lib, gt, FakeDecider("fire_and_check_ammo"))
        assert out["status"] == "success"
        assert out["steps"] == 1

    def test_idle_goal_succeeds_when_choosing_idle(self):
        lib, gt = _lib()
        out = run_agent_loop(GOAL_IDLE, lib, gt, FakeDecider("idle_and_check_ammo"))
        assert out["status"] == "success"

    def test_idle_goal_fails_if_agent_fires(self):
        # Wrong choice (fire) can never satisfy "delta == 0" -> maxes out.
        lib, gt = _lib()
        out = run_agent_loop(
            GOAL_IDLE, lib, gt, FakeDecider("fire_and_check_ammo"), max_steps=3
        )
        assert out["status"] == "max_steps_exceeded"

    def test_fire3_goal_needs_three_steps(self):
        lib, gt = _lib()
        out = run_agent_loop(GOAL_FIRE3, lib, gt, FakeDecider("fire_and_check_ammo"))
        assert out["status"] == "success"
        assert out["steps"] == 3
        assert out["cumulative"]["delta"] == 3


class TestFunctionCallingDecider:
    def _client(self, fn_name: str | None):
        tool_calls = (
            [SimpleNamespace(function=SimpleNamespace(name=fn_name, arguments="{}"))]
            if fn_name
            else None
        )
        message = SimpleNamespace(tool_calls=tool_calls, content="no tool")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        class FakeClient:
            def __init__(self, resp):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kw: resp)
                )
                self.last_kwargs: Dict[str, Any] = {}

        client = FakeClient(response)
        # capture kwargs to assert tools were passed
        orig = client.chat.completions.create

        def capturing(**kw):
            client.last_kwargs = kw
            return orig(**kw)

        client.chat.completions.create = capturing
        return client

    def test_returns_tool_call_name(self):
        client = self._client("fire_and_check_ammo")
        decider = FunctionCallingDecider(client=client, model="deepseek-chat")
        name = decider.decide(
            "fire the weapon",
            [],
            [{"name": "fire_and_check_ammo", "description": "fire"}],
        )
        assert name == "fire_and_check_ammo"
        # tools were actually passed to the API (native function calling)
        assert "tools" in client.last_kwargs
        assert client.last_kwargs["tools"][0]["function"]["name"] == "fire_and_check_ammo"

    def test_raises_when_no_tool_call(self):
        client = self._client(None)
        decider = FunctionCallingDecider(client=client, model="deepseek-chat")
        try:
            decider.decide("fire", [], [{"name": "fire_and_check_ammo", "description": "x"}])
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
