"""Deterministic Spec-to-Test agent for the checkpoint requirement.

Gate 4 deliberately avoids free-form C# generation. This agent maps a known
requirement family to a structured Test Plan IR that fixed templates render.
"""

from __future__ import annotations

import re

from .ir import TestCaseIR, TestPlanIR


class SpecToTestAgent:
    """Produce a minimal Test Plan IR from a checkpoint-softlock requirement."""

    def plan(self, requirement_text: str) -> TestPlanIR:
        requirement_id = self._requirement_id(requirement_text)
        self._validate_checkpoint_requirement(requirement_text)

        return TestPlanIR(
            plan_id=f"{requirement_id}_gate4_plan",
            requirement_id=requirement_id,
            requirement=requirement_text.strip(),
            cases=[
                TestCaseIR(
                    id="checkpoint_state_persistence_unit",
                    layer="unit",
                    title="Checkpoint restore preserves opened-door progression state",
                    template="checkpoint_state_persistence_unit",
                    assertions=[
                        "door_open remains true after checkpoint restore",
                        "player respawns in checkpoint_room",
                        "consumed keycard remains consumed",
                        "collected keycard remains collected",
                    ],
                ),
                TestCaseIR(
                    id="checkpoint_respawn_extraction_playmode",
                    layer="playmode",
                    title="Checkpoint respawn allows extraction without softlock",
                    template="checkpoint_respawn_extraction_playmode",
                    assertions=[
                        "checkpoint flow reaches extraction",
                        "debug_state records door_open true",
                        "debug_state records extraction_reached true",
                        "debug_state records progression_softlock false",
                    ],
                ),
            ],
        )

    @staticmethod
    def _requirement_id(requirement_text: str) -> str:
        match = re.search(r"(?im)^ID:\s*([a-zA-Z0-9_-]+)\s*$", requirement_text)
        if match:
            return match.group(1)
        return "checkpoint_no_softlock"

    @staticmethod
    def _validate_checkpoint_requirement(requirement_text: str) -> None:
        lowered = requirement_text.lower()
        required_terms = ["checkpoint", "respawn", "extraction"]
        missing = [term for term in required_terms if term not in lowered]
        if missing:
            raise ValueError(
                "Gate 4 Spec-to-Test only supports the checkpoint requirement; "
                f"missing terms: {', '.join(missing)}"
            )
