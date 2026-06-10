"""Gate 3 Unity Gameplay Agent smoke.

Runs the goal-level checkpoint scenario through the reused v1 reflective agent
core, using the Gate 2 Unity runtime bridge as the game adapter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_agent_runtime() -> None:
    if importlib.util.find_spec("langgraph") is not None:
        return

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    current = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if current == target:
        return

    completed = subprocess.run([str(target), *sys.argv], check=False)
    raise SystemExit(completed.returncode)


_ensure_agent_runtime()

from agent import parse_goals, run_reflective_agent
from agent.reflection import RECOVERY, FailureType, ReflectionCase
from run_unity_tests import (
    DEFAULT_PROJECT,
    DEFAULT_UNITY_EXE,
    parse_results,
    print_log_tail,
)
from unity_agent import UnityCheckpointActions, UnityPrimitives, UnityStatePerceptor
from unity_smoke import (
    BridgeClient,
    SmokeFailure,
    assert_png,
    run_unity,
    wait_for_ready,
    wait_for_unity,
)


GOALS_PATH = REPO_ROOT / "unity_agent" / "goals.feature"
DEFAULT_RESULTS = REPO_ROOT / "results" / "unity" / "agent_results.xml"
DEFAULT_LOG = REPO_ROOT / "results" / "unity" / "agent_unity.log"
DEFAULT_READY = REPO_ROOT / "results" / "unity" / "agent_bridge_ready.json"
DEFAULT_REPORT = REPO_ROOT / "results" / "unity" / "agent_report.json"
GOAL_NAME = "gameplay_checkpoint_no_softlock"

EXPECTED_BRIDGE_ACTIONS = [
    "action:move_to_keycard",
    "action:collect_keycard",
    "action:move_to_door",
    "action:open_door",
    "action:move_to_checkpoint",
    "action:activate_checkpoint",
    "action:move_to_hazard",
    "action:die_and_respawn",
    "action:extract",
]


class ScriptedCheckpointDecider:
    """Deterministic decider for the Gate 3 smoke.

    It keeps the gate local and repeatable while still exercising the real v1
    agent loop, action library contract, and reflection ladder.
    """

    SEQUENCE = [
        "collect_keycard",
        "open_security_door",
        "activate_checkpoint",
        "die_and_respawn",
        "attempt_extraction",
    ]

    def decide(
        self,
        goal_description: str,
        history: List[Dict[str, Any]],
        tools_spec: List[Dict[str, str]],
    ) -> str:
        available = {tool["name"] for tool in tools_spec}
        completed = {
            entry.get("action")
            for entry in history
            if not entry.get("recovery")
        }

        for action in self.SEQUENCE:
            if action in available and action not in completed:
                return action

        return self.SEQUENCE[-1]


class ProgressionSoftlockReflector:
    """Deterministic reflector for local Gate 3 verification."""

    def reflect(self, anomaly: Dict[str, Any], history: List[Dict[str, Any]]) -> ReflectionCase:
        if anomaly.get("failure_type") == "progression_softlock":
            failure_type = FailureType.LOGIC
            reasoning = (
                "Extraction failed because Unity reported progression_softlock "
                "after checkpoint respawn."
            )
            confidence = 1.0
        else:
            failure_type = FailureType.EXECUTION
            reasoning = "Local smoke fallback: retry the action before reporting."
            confidence = 0.6

        return ReflectionCase(
            anomaly=anomaly,
            failure_type=failure_type,
            recovery_action=RECOVERY[failure_type],
            confidence=confidence,
            reasoning=reasoning,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect",
        choices=["normal", "progression_softlock"],
        default="normal",
        help="Expected outcome. Default requires a normal extraction success.",
    )
    parser.add_argument(
        "--unity",
        default=os.environ.get("UNITY_EXE", str(DEFAULT_UNITY_EXE)),
        help="Path to Unity.exe. Defaults to UNITY_EXE or the pinned editor.",
    )
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
        help="Unity project path. Defaults to unity/GameTestFixture.",
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="NUnit3 XML results output path for the bridge host test.",
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG),
        help="Unity Editor log output path for the bridge host test.",
    )
    parser.add_argument(
        "--ready",
        default=str(DEFAULT_READY),
        help="Bridge ready JSON path written by Unity.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Gameplay Agent report JSON output path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for Unity bridge startup and shutdown.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def load_checkpoint_goal() -> Any:
    goals = parse_goals(GOALS_PATH.read_text(encoding="utf-8"))
    for goal in goals:
        if goal.metadata.get("name") == GOAL_NAME:
            return goal
    raise SmokeFailure(f"Unity goal missing: {GOAL_NAME}")


def run_agent(prim: UnityPrimitives) -> Dict[str, Any]:
    goal = load_checkpoint_goal()
    actions = UnityCheckpointActions(prim)
    perceptor = UnityStatePerceptor()
    return run_reflective_agent(
        goal,
        actions,
        perceptor,
        ScriptedCheckpointDecider(),
        ProgressionSoftlockReflector(),
        max_steps=8,
        max_reobserves=1,
        max_retries=1,
    )


def classify_failure(agent_result: Dict[str, Any], debug_state: Dict[str, Any]) -> str | None:
    bug_report = agent_result.get("bug_report") or {}
    anomaly = bug_report.get("anomaly") or {}
    failure_type = anomaly.get("failure_type")
    if failure_type:
        return str(failure_type)
    if debug_state.get("progression_softlock"):
        return "progression_softlock"
    return None


def write_report(
    path: Path,
    agent_result: Dict[str, Any],
    debug_state: Dict[str, Any],
    screenshot_path: str,
    trace: List[str],
) -> Dict[str, Any]:
    failure_type = classify_failure(agent_result, debug_state)
    report = {
        "status": agent_result.get("status"),
        "failure_type": failure_type,
        "goal": agent_result.get("goal"),
        "steps": agent_result.get("steps"),
        "cumulative": agent_result.get("cumulative", {}),
        "history": agent_result.get("history", []),
        "cases": agent_result.get("cases", []),
        "bug_report": agent_result.get("bug_report"),
        "debug_state": debug_state,
        "screenshot_path": screenshot_path,
        "trace": trace,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def assert_trace_has_checkpoint_flow(trace: List[str]) -> None:
    for event in EXPECTED_BRIDGE_ACTIONS:
        require(event in trace, f"Agent report trace missing {event}.")


def assert_report(report: Dict[str, Any], expected: str) -> None:
    screenshot_path = Path(str(report.get("screenshot_path", "")))
    assert_png(screenshot_path)
    trace = list(report.get("trace", []) or [])
    assert_trace_has_checkpoint_flow(trace)

    debug_state = dict(report.get("debug_state", {}) or {})
    if expected == "normal":
        require(report.get("status") == "success", f"Expected success, got {report.get('status')}.")
        require(debug_state.get("extraction_reached") is True, "debug_state should show extraction_reached.")
        require(
            debug_state.get("progression_softlock") is False,
            "Normal run should not report progression_softlock.",
        )
        require(report.get("failure_type") is None, "Normal run should not include a failure_type.")
        print("PASS agent reached extraction")
        return

    require(
        report.get("status") == "bug_reported",
        f"Expected bug_reported, got {report.get('status')}.",
    )
    require(
        report.get("failure_type") == "progression_softlock",
        f"Expected progression_softlock, got {report.get('failure_type')}.",
    )
    require(
        debug_state.get("progression_softlock") is True,
        "debug_state should show progression_softlock: true.",
    )
    require(report.get("bug_report") is not None, "Bug run should include bug_report.")
    print("PASS agent reported progression_softlock")


def main() -> int:
    args = parse_args()
    report_path = Path(args.report)
    results_path = Path(args.results)
    log_path = Path(args.log)
    ready_path = Path(args.ready)

    process = None
    client: BridgeClient | None = None
    connected = False
    smoke_failed = False

    try:
        if report_path.exists():
            report_path.unlink()

        process = run_unity(args)
        host, port = wait_for_ready(process, ready_path, args.timeout)
        print(f"PASS bridge ready {host}:{port}")

        client = BridgeClient(host, port, args.timeout)
        connected = True
        prim = UnityPrimitives(client)
        agent_result = run_agent(prim)
        print(f"PASS agent completed status={agent_result.get('status')}")

        debug_state = prim.debug_state()
        screenshot_path = prim.screenshot()
        trace = prim.trace()
        report = write_report(report_path, agent_result, debug_state, screenshot_path, trace)
        print(f"PASS report {report_path}")

        assert_report(report, args.expect)
    except SmokeFailure as exc:
        smoke_failed = True
        print(f"FAIL {exc}")
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        smoke_failed = True
        print(f"FAIL agent_smoke_error: {exc}")
    finally:
        if client is not None:
            try:
                client.request("shutdown")
            except (SmokeFailure, OSError, json.JSONDecodeError) as exc:
                smoke_failed = True
                print(f"FAIL bridge_shutdown_error: {exc}")
            finally:
                client.close()

        if process is not None:
            try:
                if connected:
                    unity_exit = wait_for_unity(process, args.timeout, log_path)
                else:
                    if process.poll() is None:
                        process.terminate()
                    unity_exit = wait_for_unity(process, 10.0, log_path)
            except SmokeFailure as exc:
                smoke_failed = True
                unity_exit = -1
                print(f"FAIL {exc}")

            _, unity_failures = parse_results(results_path)
            if unity_exit != 0:
                smoke_failed = True
                print(f"FAIL unity_exit_code={unity_exit}")
                print_log_tail(log_path)
            if unity_failures:
                smoke_failed = True

    if smoke_failed:
        print("FAIL Unity Gameplay Agent smoke failed.")
        return 1

    print(f"PASS Unity Gameplay Agent smoke passed. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
