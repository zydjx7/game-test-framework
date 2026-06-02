"""Unit tests for agent.reflection (no real API calls).

A FakeClient returns a forced tool_call with classify_failure arguments, so we
test classification -> recovery-table lookup and the RAG-ready case shape.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.reflection import RECOVERY, FailureType, ReflectionCase, Reflector


def _client(failure_type: str | None, confidence: float = 0.9, reasoning: str = "because"):
    if failure_type is None:
        message = SimpleNamespace(tool_calls=None, content="no tool")
    else:
        args = json.dumps(
            {"failure_type": failure_type, "confidence": confidence, "reasoning": reasoning}
        )
        tool_call = SimpleNamespace(function=SimpleNamespace(name="classify_failure", arguments=args))
        message = SimpleNamespace(tool_calls=[tool_call], content=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self):
            self.last_kwargs = {}
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            self.last_kwargs = kwargs
            return response

    return FakeClient()


ANOMALY = {
    "action": "fire_and_check_ammo",
    "expected": "firing should decrease ammo by at least 1",
    "result": {"ammo_before": 26, "ammo_after": 26, "delta": 0},
}


class TestReflect:
    def test_perception_maps_to_re_observe(self):
        r = Reflector(client=_client("perception"), model="deepseek-chat")
        case = r.reflect(ANOMALY, history=[])
        assert case.failure_type is FailureType.PERCEPTION
        assert case.recovery_action == "re_observe"
        assert case.recovery_action == RECOVERY[FailureType.PERCEPTION]

    def test_execution_maps_to_retry(self):
        r = Reflector(client=_client("execution"), model="deepseek-chat")
        case = r.reflect(ANOMALY, history=[])
        assert case.failure_type is FailureType.EXECUTION
        assert case.recovery_action == "retry"

    def test_logic_maps_to_report(self):
        r = Reflector(client=_client("logic", confidence=0.8), model="deepseek-chat")
        case = r.reflect(ANOMALY, history=[])
        assert case.failure_type is FailureType.LOGIC
        assert case.recovery_action == "report"
        assert case.confidence == 0.8

    def test_case_is_rag_ready_dict(self):
        r = Reflector(client=_client("perception", reasoning="HUD looked noisy"), model="x")
        case = r.reflect(ANOMALY, history=[])
        d = case.to_dict()
        assert d["failure_type"] == "perception"
        assert d["recovery_action"] == "re_observe"
        assert d["reasoning"] == "HUD looked noisy"
        assert d["recovered"] is None
        assert d["anomaly"]["action"] == "fire_and_check_ammo"

    def test_passes_the_classify_tool(self):
        client = _client("perception")
        Reflector(client=client, model="deepseek-chat").reflect(ANOMALY, history=[])
        # structured output via function calling: the tool is passed. tool_choice
        # is "auto" (deepseek-v4-flash thinking mode rejects a forced choice).
        assert client.last_kwargs["tools"][0]["function"]["name"] == "classify_failure"
        assert client.last_kwargs["tool_choice"] == "auto"

    def test_history_is_included_in_prompt(self):
        client = _client("logic")
        history = [
            {"step": 0, "action": "fire_and_check_ammo", "result": {"delta": 0}},
            {"step": 1, "action": "fire_and_check_ammo", "result": {"delta": 0}},
        ]
        Reflector(client=client, model="x").reflect(ANOMALY, history=history)
        user_msg = client.last_kwargs["messages"][1]["content"]
        assert "step 0" in user_msg and "step 1" in user_msg

    def test_raises_when_no_tool_call(self):
        r = Reflector(client=_client(None), model="x")
        try:
            r.reflect(ANOMALY, history=[])
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
