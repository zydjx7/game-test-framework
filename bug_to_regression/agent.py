"""Deterministic bug-to-regression planner for Gate 6.

Gate 6 supports the checkpoint-softlock report family only. It validates the
source report and emits a small IR consumed by a fixed Unity template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .ir import RegressionCaseIR, RegressionPlanIR


EXPECTED_REPRO_ACTIONS = [
    "move_to_keycard",
    "collect_keycard",
    "move_to_door",
    "open_door",
    "move_to_checkpoint",
    "activate_checkpoint",
    "move_to_hazard",
    "die_and_respawn",
    "extract",
]


class BugToRegressionAgent:
    """Convert a Gate 3 progression-softlock report into Regression Plan IR."""

    def plan(self, report: Dict[str, Any], source_report_path: Path) -> RegressionPlanIR:
        self._validate_checkpoint_softlock_report(report)
        repro_actions = self._validated_repro_actions(report)

        return RegressionPlanIR(
            plan_id="gate6_checkpoint_softlock_regression_plan",
            source_report_path=str(source_report_path),
            source_failure_type=str(report.get("failure_type")),
            source_goal=str(report.get("goal", "")),
            cases=[
                RegressionCaseIR(
                    id="checkpoint_softlock_regression_playmode",
                    layer="playmode",
                    title="Checkpoint respawn must not softlock extraction",
                    template="checkpoint_softlock_regression_playmode",
                    repro_actions=repro_actions,
                    assertions=[
                        "final observation reaches extraction",
                        "final observation does not report progression_softlock",
                        "debug_state records door_open true",
                        "debug_state records extraction_reached true",
                        "debug_state records progression_softlock false",
                    ],
                )
            ],
        )

    @staticmethod
    def _validate_checkpoint_softlock_report(report: Dict[str, Any]) -> None:
        if report.get("status") != "bug_reported":
            raise ValueError(f"expected bug_reported source status, got {report.get('status')}")
        if report.get("failure_type") != "progression_softlock":
            raise ValueError(f"expected progression_softlock, got {report.get('failure_type')}")

        debug_state = dict(report.get("debug_state", {}) or {})
        required_debug = {
            "bug_door_not_persisted": True,
            "door_open": False,
            "extraction_reached": False,
            "progression_softlock": True,
        }
        for field, expected in required_debug.items():
            if debug_state.get(field) is not expected:
                raise ValueError(f"debug_state.{field} expected {expected!r}, got {debug_state.get(field)!r}")

        if "checkpoint" not in str(report.get("goal", "")).lower():
            raise ValueError("source report goal does not look like the checkpoint-softlock goal")

    @staticmethod
    def _validated_repro_actions(report: Dict[str, Any]) -> List[str]:
        trace_actions = [
            str(entry).removeprefix("action:")
            for entry in list(report.get("trace", []) or [])
            if str(entry).startswith("action:")
        ]

        position = 0
        for action in trace_actions:
            if position < len(EXPECTED_REPRO_ACTIONS) and action == EXPECTED_REPRO_ACTIONS[position]:
                position += 1

        if position != len(EXPECTED_REPRO_ACTIONS):
            missing = EXPECTED_REPRO_ACTIONS[position]
            raise ValueError(f"source trace does not contain ordered repro action {missing!r}")

        return list(EXPECTED_REPRO_ACTIONS)
