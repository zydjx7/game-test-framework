"""Portability demo: the SAME agent layer drives a second game (ToyFPS).

No ViZDoom needed. Shows run_reflective_agent / the goal parser / the decider /
reflection running UNCHANGED on a pure-Python game with multi-metric goals
(ammo decreases, score increases, health increases).

Run (from project root, .venv active, DEEPSEEK_API_KEY in .env):
    python experiments/toy_fps_demo.py

(For an API-free proof, run `python -m pytest tests/test_toy_fps.py`.)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import FunctionCallingDecider, Reflector, parse_goals, run_reflective_agent  # noqa: E402
from toy_fps import ToyActions, ToyFPS, ToyPerceptor, ToyPrimitives  # noqa: E402


def main() -> None:
    goals = parse_goals((PROJECT_ROOT / "toy_fps" / "goals.feature").read_text(encoding="utf-8"))
    decider = FunctionCallingDecider()
    reflector = Reflector()
    perceptor = ToyPerceptor()

    passed = 0
    for goal in goals:
        lib = ToyActions(ToyPrimitives(ToyFPS()))
        out = run_reflective_agent(goal, lib, perceptor, decider, reflector, max_steps=6)
        chose = [h["action"] for h in out["history"]]
        metrics = {k: v for k, v in out["cumulative"].items() if k.endswith(("_before", "_after"))}
        print(f"{goal.metadata['name']:28} -> {out['status']:9} chose={chose} {metrics}")
        passed += out["status"] == "success"

    print(f"\n{passed}/{len(goals)} ToyFPS goals via the SAME agent layer (no engine, no edits).")


if __name__ == "__main__":
    main()
