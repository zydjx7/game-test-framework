"""Goal-level Gherkin: parse test goals the agent must achieve.

The paradigm shift (Phase 2 novelty): a scenario describes the test GOAL, not
the per-step actions. The agent decides HOW. A Scenario block looks like:

    Scenario: Firing consumes ammo
      Goal: Fire the weapon and confirm it consumes ammo.
      Available actions: fire_and_check_ammo, idle_and_check_ammo
      Success: ammo_before - ammo_after >= 1

`Success:` compiles to a predicate over the agent loop's cumulative result
dict (keys: ammo_before, ammo_after, delta, steps, last_action, last_delta).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Goal:
    description: str
    available_actions: List[str]
    success_criteria: Callable[[Dict[str, Any]], bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_satisfied(self, result: Dict[str, Any]) -> bool:
        return self.success_criteria(result)


def compile_success(expression: str) -> Callable[[Dict[str, Any]], bool]:
    """Compile a `Success:` expression into a predicate over the result dict.

    The expression may reference result keys (ammo_before, ammo_after, delta,
    steps, ...) and standard comparison/arithmetic operators. It is evaluated
    with no builtins. Gherkin here is trusted researcher-authored input, not
    untrusted user input, so a sandboxed eval is acceptable and keeps the
    parser tiny.
    """

    code = compile(expression.strip(), "<success>", "eval")

    def criteria(result: Dict[str, Any]) -> bool:
        return bool(eval(code, {"__builtins__": {}}, dict(result)))  # noqa: S307

    return criteria


def parse_goals(text: str) -> List[Goal]:
    """Parse one or more Scenario blocks into Goal objects."""

    goals: List[Goal] = []
    current: Dict[str, Any] = {}

    def flush() -> None:
        if not current:
            return
        if "success" not in current:
            raise ValueError(f"Scenario '{current.get('name')}' has no Success: line")
        goals.append(
            Goal(
                description=current.get("goal", current.get("name", "")),
                available_actions=current.get("available_actions", []),
                success_criteria=compile_success(current["success"]),
                metadata={"name": current.get("name", ""), "raw_success": current["success"]},
            )
        )
        current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key == "scenario":
            flush()
            current["name"] = value
        elif key == "goal":
            current["goal"] = value
        elif key in ("available actions", "available_actions"):
            current["available_actions"] = [a.strip() for a in value.split(",") if a.strip()]
        elif key == "success":
            current["success"] = value
        # unknown lines are ignored (keeps the parser forgiving)

    flush()
    return goals
