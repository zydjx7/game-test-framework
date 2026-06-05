"""Probe effect latency for ViZDoom adapter calibration.

This is an experiment helper, not agent logic. Use it before adding a new
composite so the observation window is based on measured scenario timing
instead of a guessed number.

Examples:
    python experiments/vizdoom/probe_timing.py --scenario health_gathering --metric health --policy noop
    python experiments/vizdoom/probe_timing.py --scenario defend_the_center --metric health --policy noop
    python experiments/vizdoom/probe_timing.py --scenario defend_the_center --metric ammo --policy attack
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import VizDoomEnv  # noqa: E402


def _action(policy: str, buttons: List[str]) -> List[int]:
    vec = [0] * len(buttons)
    for name in _policy_buttons(policy):
        if name in buttons:
            vec[buttons.index(name)] = 1
    return vec


def _policy_buttons(policy: str) -> List[str]:
    if policy == "noop":
        return []
    if policy == "attack":
        return ["ATTACK"]
    if policy == "turn_left":
        return ["TURN_LEFT"]
    if policy == "attack_turn_left":
        return ["ATTACK", "TURN_LEFT"]
    if policy == "move_forward":
        return ["MOVE_FORWARD"]
    raise ValueError(f"unknown policy: {policy}")


def probe_once(
    scenario: str,
    metric: str,
    policy: str,
    max_tics: int,
    render_hud: bool,
) -> Dict[str, Any]:
    with VizDoomEnv(scenario=scenario, window_visible=False, render_hud=render_hud) as env:
        state = env.reset()
        buttons = env.get_button_names()
        start = state.game_variables.get(metric)
        last = start

        for tic in range(1, max_tics + 1):
            if state.done:
                return {
                    "changed": False,
                    "done": True,
                    "tic": tic - 1,
                    "start": start,
                    "before": last,
                    "after": None,
                    "buttons": buttons,
                }

            state = env.step(_action(policy, buttons))
            current = state.game_variables.get(metric)
            if current != last:
                return {
                    "changed": True,
                    "done": state.done,
                    "tic": tic,
                    "start": start,
                    "before": last,
                    "after": current,
                    "buttons": buttons,
                }
            last = current

        return {
            "changed": False,
            "done": state.done,
            "tic": max_tics,
            "start": start,
            "before": last,
            "after": last,
            "buttons": buttons,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--policy", default="noop")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-tics", type=int, default=300)
    parser.add_argument("--render-hud", action="store_true")
    args = parser.parse_args()

    print(
        f"scenario={args.scenario} metric={args.metric} policy={args.policy} "
        f"episodes={args.episodes} max_tics={args.max_tics}"
    )
    for ep in range(args.episodes):
        result = probe_once(
            args.scenario,
            args.metric,
            args.policy,
            args.max_tics,
            args.render_hud,
        )
        print(
            f"ep{ep}: changed={result['changed']} done={result['done']} "
            f"tic={result['tic']} {result['before']}->{result['after']} "
            f"start={result['start']} buttons={result['buttons']}"
        )


if __name__ == "__main__":
    main()
