"""Structured Regression Test Plan IR for Gate 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RegressionCaseIR:
    id: str
    layer: str
    title: str
    template: str
    repro_actions: List[str]
    assertions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionPlanIR:
    plan_id: str
    source_report_path: str
    source_failure_type: str
    source_goal: str
    cases: List[RegressionCaseIR]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source_report_path": self.source_report_path,
            "source_failure_type": self.source_failure_type,
            "source_goal": self.source_goal,
            "cases": [case.to_dict() for case in self.cases],
        }
