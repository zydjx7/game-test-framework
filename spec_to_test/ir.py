"""Structured Test Plan IR for Gate 4 Spec-to-Test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class TestCaseIR:
    id: str
    layer: str
    title: str
    template: str
    assertions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestPlanIR:
    plan_id: str
    requirement_id: str
    requirement: str
    cases: List[TestCaseIR]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "requirement_id": self.requirement_id,
            "requirement": self.requirement,
            "cases": [case.to_dict() for case in self.cases],
        }
