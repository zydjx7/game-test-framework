"""Unity adapter over the Gate 2 runtime bridge.

This package is the Gate 3 adapter layer: it reuses the v1 agent core unchanged
and translates goal-level Unity action templates into bridge commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from actions.result import increased, snapshot_result


def _as_int(value: Any) -> int:
    return 1 if bool(value) else 0


@dataclass
class UnityAgentState:
    """State object consumed by Unity action composites and graph re_observe."""

    game_variables: Dict[str, int]
    screen: None = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    has_keycard: int = 0
    keycard_collected: int = 0
    door_open: int = 0
    checkpoint_active: int = 0
    respawn_count: int = 0
    extraction_reached: int = 0
    progression_softlock: int = 0
    failure_reason: str = ""
    player_zone: str = ""


class UnityStatePerceptor:
    """Convert bridge observations into a metric-bearing UnityAgentState."""

    def perceive(self, screenshot: Any, **kwargs: Any) -> UnityAgentState:
        game_variables = dict(kwargs.get("game_variables", {}) or {})
        raw = dict(kwargs.get("raw_response", {}) or {})
        observation = dict(kwargs.get("observation", {}) or raw)

        values = {
            "has_keycard": int(game_variables.get("has_keycard", _as_int(observation.get("has_keycard")))),
            "keycard_collected": int(
                game_variables.get("keycard_collected", _as_int(observation.get("keycard_collected")))
            ),
            "door_open": int(game_variables.get("door_open", _as_int(observation.get("door_open")))),
            "checkpoint_active": int(
                game_variables.get("checkpoint_active", _as_int(observation.get("checkpoint_active")))
            ),
            "respawn_count": int(game_variables.get("respawn_count", observation.get("respawn_count", 0) or 0)),
            "extraction_reached": int(
                game_variables.get("extraction_reached", _as_int(observation.get("extraction_reached")))
            ),
            "progression_softlock": int(
                game_variables.get("progression_softlock", _as_int(observation.get("progression_softlock")))
            ),
        }

        return UnityAgentState(
            game_variables=values,
            screen=None,
            raw_response=raw or observation,
            failure_reason=str(observation.get("failure_reason", "") or ""),
            player_zone=str(observation.get("player_zone", "") or ""),
            **values,
        )


class UnityPrimitives:
    """Layer-1 primitives over the Gate 2 JSONL/TCP bridge."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.last_state: Optional[UnityAgentState] = None
        self.perceptor = UnityStatePerceptor()

    def reset(self) -> UnityAgentState:
        self.last_state = self._state_from_response(self.client.request("reset"))
        return self.last_state

    def observe(self) -> UnityAgentState:
        self.last_state = self._state_from_response(self.client.request("observe"))
        return self.last_state

    def move_to_keycard(self) -> UnityAgentState:
        return self._action("move_to_keycard")

    def collect_keycard(self) -> UnityAgentState:
        return self._action("collect_keycard")

    def move_to_door(self) -> UnityAgentState:
        return self._action("move_to_door")

    def open_door(self) -> UnityAgentState:
        return self._action("open_door")

    def move_to_checkpoint(self) -> UnityAgentState:
        return self._action("move_to_checkpoint")

    def activate_checkpoint(self) -> UnityAgentState:
        return self._action("activate_checkpoint")

    def move_to_hazard(self) -> UnityAgentState:
        return self._action("move_to_hazard")

    def die_and_respawn(self) -> UnityAgentState:
        return self._action("die_and_respawn")

    def extract(self) -> UnityAgentState:
        return self._action("extract")

    def debug_state(self) -> Dict[str, Any]:
        response = self.client.request("debug_state")
        self.last_state = self._state_from_response(response)
        return dict(response.get("debug_state", {}) or {})

    def screenshot(self) -> str:
        response = self.client.request("screenshot")
        self.last_state = self._state_from_response(response)
        return str(response.get("screenshot_path", "") or "")

    def trace(self) -> List[str]:
        response = self.client.request("trace")
        self.last_state = self._state_from_response(response)
        return list(response.get("trace", []) or [])

    def _action(self, action_name: str) -> UnityAgentState:
        self.last_state = self._state_from_response(self.client.request("action", action_name))
        return self.last_state

    def _state_from_response(self, response: Dict[str, Any]) -> UnityAgentState:
        observation = dict(response.get("observation", {}) or {})
        game_variables = {
            "has_keycard": _as_int(observation.get("has_keycard")),
            "keycard_collected": _as_int(observation.get("keycard_collected")),
            "door_open": _as_int(observation.get("door_open")),
            "checkpoint_active": _as_int(observation.get("checkpoint_active")),
            "respawn_count": int(observation.get("respawn_count", 0) or 0),
            "extraction_reached": _as_int(observation.get("extraction_reached")),
            "progression_softlock": _as_int(observation.get("progression_softlock")),
        }
        return self.perceptor.perceive(
            None,
            game_variables=game_variables,
            observation=observation,
            raw_response=observation,
        )


class UnityCheckpointActions:
    """Goal-level action templates for the checkpoint-softlock fixture."""

    __test__ = False

    DESCRIPTIONS: Dict[str, str] = {
        "collect_keycard": "Navigate to the keycard and collect it.",
        "open_security_door": "Use the keycard to open the security door.",
        "activate_checkpoint": "Move beyond the door and activate the checkpoint.",
        "die_and_respawn": "Trigger death and respawn at the checkpoint.",
        "attempt_extraction": "Attempt to reach extraction after respawn.",
    }

    EXPECTATIONS: Dict[str, Dict[str, Any]] = {
        "collect_keycard": {
            "describe": "keycard should become collected",
            "check": increased("keycard_collected", 1),
        },
        "open_security_door": {
            "describe": "security door should become open",
            "check": increased("door_open", 1),
        },
        "activate_checkpoint": {
            "describe": "checkpoint should become active",
            "check": increased("checkpoint_active", 1),
        },
        "die_and_respawn": {
            "describe": "respawn count should increase",
            "check": increased("respawn_count", 1),
        },
        "attempt_extraction": {
            "describe": "extraction should be reached without progression_softlock",
            "check": lambda result: (
                increased("extraction_reached", 1)(result)
                and result.get("progression_softlock_after") == 0
            ),
        },
    }

    def __init__(self, primitives: UnityPrimitives) -> None:
        self.prim = primitives

    def collect_keycard(self, perceptor: UnityStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.move_to_keycard()
        self.prim.collect_keycard()
        after = self._read(perceptor)
        return self._result(before, after, "keycard_collected")

    def open_security_door(self, perceptor: UnityStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.move_to_door()
        self.prim.open_door()
        after = self._read(perceptor)
        return self._result(before, after, "door_open")

    def activate_checkpoint(self, perceptor: UnityStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.move_to_checkpoint()
        self.prim.activate_checkpoint()
        after = self._read(perceptor)
        return self._result(before, after, "checkpoint_active")

    def die_and_respawn(self, perceptor: UnityStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.move_to_hazard()
        self.prim.die_and_respawn()
        after = self._read(perceptor)
        return self._result(before, after, "respawn_count", "door_open")

    def attempt_extraction(self, perceptor: UnityStatePerceptor) -> Dict[str, Any]:
        before = self._read(perceptor)
        self.prim.extract()
        after = self._read(perceptor)
        result = self._result(before, after)
        result["failure_reason"] = after.failure_reason
        if after.progression_softlock:
            result["failure_type"] = "progression_softlock"
        return result

    def run(self, template_name: str, perceptor: UnityStatePerceptor) -> Dict[str, Any]:
        if template_name not in self.list_templates():
            raise ValueError(f"unknown Unity test template: {template_name}")
        return getattr(self, template_name)(perceptor)

    def observe(self, perceptor: UnityStatePerceptor) -> UnityAgentState:
        return self._read(perceptor)

    @classmethod
    def list_templates(cls) -> List[str]:
        return list(cls.DESCRIPTIONS.keys())

    @classmethod
    def check_expectation(
        cls, template_name: str, result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        spec = cls.EXPECTATIONS.get(template_name)
        if spec is None:
            return None

        check: Callable[[Dict[str, Any]], bool] = spec["check"]
        if check(result):
            return None

        anomaly = {
            "action": template_name,
            "expected": spec["describe"],
            "result": result,
        }
        if result.get("failure_type") == "progression_softlock":
            anomaly["failure_type"] = "progression_softlock"
            anomaly["failure_reason"] = result.get("failure_reason", "")
        return anomaly

    def _read(self, perceptor: UnityStatePerceptor) -> UnityAgentState:
        state = self.prim.observe()
        return perceptor.perceive(
            getattr(state, "screen", None),
            game_variables=state.game_variables,
            observation=state.raw_response,
            raw_response=state.raw_response,
        )

    @staticmethod
    def _result(
        before: UnityAgentState,
        after: UnityAgentState,
        *metrics: str,
    ) -> Dict[str, Any]:
        metric_names = set(metrics) | {"extraction_reached", "progression_softlock"}
        return snapshot_result(
            {metric: getattr(before, metric) for metric in sorted(metric_names)},
            {metric: getattr(after, metric) for metric in sorted(metric_names)},
        )
