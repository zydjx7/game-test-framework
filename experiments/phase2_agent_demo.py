"""Phase 2 Stage D demo: goal-level Gherkin -> agent loop -> ViZDoom.

For each goal in agent/goals.feature, the agent uses DeepSeek native function
calling to choose an action template and drive defend_the_center until the
goal's success criterion holds. No per-step functions are hand-written.

Run (from project root, .venv active, DEEPSEEK_API_KEY in .env):
    python experiments/phase2_agent_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from actions import ActionPrimitives, TestActions  # noqa: E402
from agent import FunctionCallingDecider, parse_goals, run_agent_loop  # noqa: E402
from env import VizDoomEnv  # noqa: E402
from perception import GroundTruthPerceptor  # noqa: E402


def main() -> None:
    feature = (PROJECT_ROOT / "agent" / "goals.feature").read_text(encoding="utf-8")
    goals = parse_goals(feature)

    decider = FunctionCallingDecider()  # real DeepSeek function calling
    perceptor = GroundTruthPerceptor()  # deterministic v1 (isolate decision from perception)

    passed = 0
    with VizDoomEnv(scenario="defend_the_center", window_visible=False, render_hud=True) as env:
        action_lib = TestActions(ActionPrimitives(env))
        for goal in goals:
            print(f"\n=== GOAL: {goal.metadata['name']} ===")
            print(f"  criterion: {goal.metadata['raw_success']}")
            out = run_agent_loop(goal, action_lib, perceptor, decider, max_steps=8)
            chosen = [h["action"] for h in out["history"]]
            print(f"  agent chose: {chosen}")
            print(f"  result: {out['status']} in {out['steps']} step(s)")
            if out["status"] == "success":
                passed += 1

    print(f"\n===== {passed}/{len(goals)} goals achieved =====")


if __name__ == "__main__":
    main()
