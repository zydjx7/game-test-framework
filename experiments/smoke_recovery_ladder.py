"""Live smoke for the diagnostic recovery ladder (Stage-1 step 3).

Real ViZDoom + real DeepSeek reflector. Injects a PERCEPTION fault vs an
EXECUTION fault and shows the SAME agent takes two DIFFERENT recovery paths, both
reaching success:

    perception fault -> re_observe alone recovers (a fresh read sees the truth;
                        NO extra shot fired, `steps` not inflated)
    execution  fault -> re_observe fails (re-reading still shows no change), so
                        the ladder escalates to retry, which actually re-fires

The difference is driven by the LADDER + what each fault actually is, NOT by the
LLM's perception-vs-execution classification (which ADR-0003 says is unreliable
at the moment of the anomaly). The real DeepSeek reflector's 3-type diagnosis is
recorded in `cases` but does not steer the recovery.

Usage (project root, .venv active, DEEPSEEK_API_KEY in .env):
    python experiments/smoke_recovery_ladder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from actions import ActionPrimitives, TestActions  # noqa: E402
from agent import Reflector, parse_goals, run_reflective_agent  # noqa: E402
from env import VizDoomEnv  # noqa: E402
from experiments.inject import ExecutionFailureInjector, PerceptionFailureInjector  # noqa: E402
from perception import GroundTruthPerceptor  # noqa: E402

GOAL = parse_goals(
    """Scenario: fire
  Goal: Fire the weapon and confirm it consumes ammo.
  Available actions: fire_and_check_ammo, idle_and_check_ammo
  Success: ammo_before - ammo_after >= 1"""
)[0]


class FixedDecider:
    """Fix the decision so the recovery path is what is under test, not choice."""

    def decide(self, *args, **kwargs) -> str:
        return "fire_and_check_ammo"


def _components(env, condition):
    prim = ActionPrimitives(env)
    if condition == "perception":
        return TestActions(prim), PerceptionFailureInjector(GroundTruthPerceptor(), mask_changes=1)
    return TestActions(ExecutionFailureInjector(prim, fail_fires=1)), GroundTruthPerceptor()


def _recoveries(out):
    return [h["recovery"] for h in out["history"] if "recovery" in h]


def main() -> None:
    decider = FixedDecider()
    reflector = Reflector()  # real DeepSeek
    results = {}

    with VizDoomEnv(scenario="defend_the_center", window_visible=False, render_hud=True) as env:
        for condition in ("perception", "execution"):
            lib, perc = _components(env, condition)
            out = run_reflective_agent(
                GOAL, lib, perc, decider, reflector,
                max_steps=6, max_reobserves=1, max_retries=1,
            )
            rec = _recoveries(out)
            diag = [c["failure_type"] for c in out["cases"]]
            results[condition] = (out, rec)
            print(f"\n[{condition}] status={out['status']} steps={out['steps']}")
            print(f"  cumulative : {out['cumulative']}")
            print(f"  recoveries : {rec}")
            print(f"  llm_diag   : {diag} (recorded, does not steer recovery)")

    p_out, p_rec = results["perception"]
    e_out, e_rec = results["execution"]

    print("\n===== assertions =====")
    ok = True
    def check(label, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {label}")

    check("perception recovered (success)", p_out["status"] == "success")
    check("perception path = re_observe only (no extra shot)", p_rec == ["re_observe"])
    check("execution recovered (success)", e_out["status"] == "success")
    check("execution path = re_observe then retry", e_rec == ["re_observe", "retry"])
    check("the two recovery paths differ", p_rec != e_rec)
    print("\nSMOKE", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
