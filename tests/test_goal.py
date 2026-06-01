"""Unit tests for agent.goal (Goal dataclass + goal-level Gherkin parser)."""

from __future__ import annotations

from pathlib import Path

from agent.goal import Goal, compile_success, parse_goals

FEATURE = """
Scenario: Firing consumes ammo
  Goal: Fire the weapon and confirm it consumes ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 1

Scenario: Idle does not consume ammo
  Goal: Confirm idle leaves ammo unchanged.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after == 0 and steps >= 1
"""


class TestCompileSuccess:
    def test_true_and_false(self):
        crit = compile_success("ammo_before - ammo_after >= 1")
        assert crit({"ammo_before": 26, "ammo_after": 25}) is True
        assert crit({"ammo_before": 26, "ammo_after": 26}) is False

    def test_compound_expression(self):
        crit = compile_success("delta == 0 and steps >= 1")
        assert crit({"delta": 0, "steps": 1}) is True
        assert crit({"delta": 1, "steps": 1}) is False

    def test_no_builtins_available(self):
        # __import__ must not be reachable from the sandbox.
        crit = compile_success("delta >= 1")
        try:
            compile_success("__import__('os')")({"delta": 0})
            assert False, "expected NameError for __import__"
        except NameError:
            pass


class TestParseGoals:
    def test_parses_two_scenarios(self):
        goals = parse_goals(FEATURE)
        assert len(goals) == 2
        assert isinstance(goals[0], Goal)

    def test_fields(self):
        g = parse_goals(FEATURE)[0]
        assert g.available_actions == ["fire_and_check_ammo", "idle_and_check_ammo"]
        assert "consumes ammo" in g.description
        assert g.metadata["name"] == "Firing consumes ammo"

    def test_success_criteria_works(self):
        g = parse_goals(FEATURE)[0]
        assert g.is_satisfied({"ammo_before": 26, "ammo_after": 25}) is True
        assert g.is_satisfied({"ammo_before": 26, "ammo_after": 26}) is False

    def test_project_feature_file_parses(self):
        path = Path(__file__).resolve().parents[1] / "agent" / "goals.feature"
        goals = parse_goals(path.read_text(encoding="utf-8"))
        assert len(goals) == 3
        assert [g.metadata["name"] for g in goals] == [
            "Firing consumes ammo",
            "Idle does not consume ammo",
            "Repeated firing reduces ammo by three",
        ]
