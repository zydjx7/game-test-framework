"""Structured VLM evidence bundle for Gate 5.

The smoke uses ``FixtureVlmBackend`` as a deterministic local test double. It
keeps Gate 5 machine-checkable without API keys while preserving the backend
contract a real VLM can implement later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


VERDICT_SOURCE = "debug_state+trace+agent_report"
EVIDENCE_ROLE = "supporting_evidence_only"


@dataclass(frozen=True)
class VisualQuestion:
    id: str
    prompt: str


@dataclass(frozen=True)
class VisualAnswer:
    question_id: str
    prompt: str
    answer: str
    confidence: float
    backend: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualEvidenceReport:
    schema_version: int
    source_report_path: str
    screenshot_path: str
    verdict_source: str
    failure_type: str | None
    visual_evidence_role: str
    questions: List[VisualAnswer]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_report_path": self.source_report_path,
            "screenshot_path": self.screenshot_path,
            "verdict_source": self.verdict_source,
            "failure_type": self.failure_type,
            "visual_evidence_role": self.visual_evidence_role,
            "questions": [question.to_dict() for question in self.questions],
        }


DEFAULT_QUESTIONS = [
    VisualQuestion(
        id="door_blocks_extraction",
        prompt="Is the extraction route blocked by a closed door?",
    ),
    VisualQuestion(
        id="player_view_blocked",
        prompt="Does the evidence suggest the player cannot reach extraction after respawn?",
    ),
    VisualQuestion(
        id="state_visual_consistency",
        prompt="Is the visual evidence consistent with the debug_state softlock diagnosis?",
    ),
]


class FixtureVlmBackend:
    """Deterministic Gate 5 VLM backend used only for local smoke verification."""

    name = "fixture-vlm"

    def answer(
        self,
        question: VisualQuestion,
        screenshot_path: Path,
        context: Dict[str, Any],
    ) -> VisualAnswer:
        debug_state = dict(context.get("debug_state", {}) or {})
        failure_type = context.get("failure_type")
        trace = list(context.get("trace", []) or [])
        failure_reason = str(debug_state.get("failure_reason", "") or "")
        softlocked = bool(debug_state.get("progression_softlock"))
        door_closed = debug_state.get("door_open") is False
        extraction_missing = debug_state.get("extraction_reached") is False

        if question.id == "door_blocks_extraction":
            answer = "yes" if softlocked and door_closed else "no"
            confidence = 0.95 if answer == "yes" else 0.6
            rationale = (
                "The screenshot artifact exists and the synchronized debug context "
                "reports the security door closed after respawn."
            )
        elif question.id == "player_view_blocked":
            answer = "yes" if softlocked and extraction_missing else "no"
            confidence = 0.9 if answer == "yes" else 0.55
            rationale = (
                "The agent attempted extraction, but the synchronized runtime state "
                "reports extraction_reached=false and progression_softlock=true."
            )
        elif question.id == "state_visual_consistency":
            expected_trace = "action:extract" in trace and "action:die_and_respawn" in trace
            answer = "consistent" if failure_type == "progression_softlock" and softlocked and expected_trace else "unclear"
            confidence = 0.93 if answer == "consistent" else 0.5
            rationale = (
                "The final trace includes respawn then extract, and the report "
                f"failure reason is {failure_reason!r}."
            )
        else:
            answer = "unknown"
            confidence = 0.0
            rationale = "The fixture backend has no rule for this question."

        return VisualAnswer(
            question_id=question.id,
            prompt=question.prompt,
            answer=answer,
            confidence=confidence,
            backend=self.name,
            rationale=f"{rationale} Screenshot: {screenshot_path.name}.",
        )


def build_gate5_evidence(
    source_report_path: Path,
    agent_report: Dict[str, Any],
    backend: Any | None = None,
) -> VisualEvidenceReport:
    screenshot_path = Path(str(agent_report.get("screenshot_path", "")))
    context = {
        "debug_state": dict(agent_report.get("debug_state", {}) or {}),
        "trace": list(agent_report.get("trace", []) or []),
        "failure_type": agent_report.get("failure_type"),
        "bug_report": agent_report.get("bug_report"),
    }
    active_backend = backend or FixtureVlmBackend()
    answers = [
        active_backend.answer(question, screenshot_path, context)
        for question in DEFAULT_QUESTIONS
    ]

    return VisualEvidenceReport(
        schema_version=1,
        source_report_path=str(source_report_path),
        screenshot_path=str(screenshot_path),
        verdict_source=VERDICT_SOURCE,
        failure_type=agent_report.get("failure_type"),
        visual_evidence_role=EVIDENCE_ROLE,
        questions=answers,
    )
