"""Phase 3 Stage E: evaluate reflection (baseline vs proposed) under injected
failures, and tabulate the honest metrics (Doc/phase3-design.md §6).

Design choices:
- The DECIDER is fixed (always the goal's action). Reflection, not decision, is
  under test; a fixed decider removes decision variance and cost, and makes the
  baseline deterministic. The REFLECTOR is the real DeepSeek model.
- baseline  = run_agent_loop (Phase 2 while loop, NO reflection)
- proposed  = run_reflective_agent (LangGraph reflection)
- Goal success is cumulative, so for a ONE-SHOT failure even the baseline can
  brute-force its way to success by repeating; the decisive contrast is the
  PERSISTENT failure, where the baseline silently times out (bug MISSED) while
  the proposed agent reports a bug (bug DETECTED) -- Claim 1 in miniature.

Usage (from project root, .venv active, DEEPSEEK_API_KEY in .env):
    python experiments/eval_reflection.py --runs 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from actions import ActionPrimitives, TestActions  # noqa: E402
from agent import Reflector, parse_goals, run_agent_loop, run_reflective_agent  # noqa: E402
from env import VizDoomEnv  # noqa: E402
from experiments.inject import ExecutionFailureInjector, PerceptionFailureInjector  # noqa: E402
from perception import GroundTruthPerceptor  # noqa: E402

GOAL = parse_goals(
    """Scenario: fire
  Goal: Fire the weapon and confirm it consumes ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 1"""
)[0]

CONDITIONS = ["perception", "execution", "persistent"]


class FixedDecider:
    def __init__(self, action: str):
        self.action = action

    def decide(self, *args, **kwargs) -> str:
        return self.action


def make_components(env, condition):
    """Fresh action_lib + perceptor (fresh injection budget) for one run."""
    prim = ActionPrimitives(env)
    if condition == "perception":
        return TestActions(prim), PerceptionFailureInjector(GroundTruthPerceptor(), mask_changes=1)
    if condition == "execution":
        return TestActions(ExecutionFailureInjector(prim, fail_fires=1)), GroundTruthPerceptor()
    # persistent: action never goes through -> LOGIC-like
    return TestActions(ExecutionFailureInjector(prim, fail_fires=99)), GroundTruthPerceptor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6)
    # Diagnostic recovery ladder budgets (Stage-1 step 3). Pre-declare these per
    # experiment; do not tune per case. 1/1 is the minimal, cleanest ladder.
    parser.add_argument("--max-reobserves", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    decider = FixedDecider("fire_and_check_ammo")
    reflector = Reflector()
    rows = []

    with VizDoomEnv(scenario="defend_the_center", window_visible=False, render_hud=True) as env:
        for condition in CONDITIONS:
            for run in range(args.runs):
                # baseline (no reflection)
                lib_b, perc_b = make_components(env, condition)
                b = run_agent_loop(GOAL, lib_b, perc_b, decider, max_steps=args.max_steps)
                rows.append({
                    "condition": condition, "run": run, "mode": "baseline",
                    "status": b["status"], "steps": b["steps"],
                    "cases": 0, "recovered": "", "bug_reported": 0,
                })
                # proposed (reflection)
                lib_p, perc_p = make_components(env, condition)
                p = run_reflective_agent(
                    GOAL, lib_p, perc_p, decider, reflector,
                    max_steps=args.max_steps,
                    max_reobserves=args.max_reobserves, max_retries=args.max_retries,
                )
                recovered = any(c.get("recovered") for c in p["cases"])
                rows.append({
                    "condition": condition, "run": run, "mode": "proposed",
                    "status": p["status"], "steps": p["steps"],
                    "cases": len(p["cases"]),
                    "recovered": recovered,
                    "bug_reported": int(p["status"] == "bug_reported"),
                })
                print(f"{condition:11} run{run} | baseline={b['status']:18} | proposed={p['status']:14} cases={len(p['cases'])}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "experiments" / f"reflection_results_{stamp}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    _summary(rows, args.runs)
    print(f"\nCSV: {out}")


def _summary(rows, runs):
    print("\n===== Phase 3 Reflection Summary =====")
    print(f"(runs per condition = {runs})")
    for cond in CONDITIONS:
        base = [r for r in rows if r["condition"] == cond and r["mode"] == "baseline"]
        prop = [r for r in rows if r["condition"] == cond and r["mode"] == "proposed"]
        b_succ = sum(r["status"] == "success" for r in base)
        p_succ = sum(r["status"] == "success" for r in prop)
        p_bug = sum(r["bug_reported"] for r in prop)
        p_rec = sum(1 for r in prop if r["recovered"] is True)
        n = len(base)
        print(f"\n[{cond}]")
        print(f"  baseline : success {b_succ}/{n}, bugs_reported 0/{n} (cannot report)")
        if cond == "persistent":
            print(f"  proposed : bugs_reported {p_bug}/{n}  <-- DETECTED (baseline missed)")
        else:
            print(f"  proposed : success {p_succ}/{n}, recovered {p_rec}/{n}")


if __name__ == "__main__":
    main()
