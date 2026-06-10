"""Gate 2 Unity runtime bridge smoke.

Starts the Unity PlayMode bridge host, drives the checkpoint fixture over a
local JSONL/TCP bridge, and prints PASS/FAIL from Python.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from run_unity_tests import (
    DEFAULT_PROJECT,
    DEFAULT_UNITY_EXE,
    parse_results,
    print_log_tail,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "unity" / "bridge_results.xml"
DEFAULT_LOG = REPO_ROOT / "results" / "unity" / "bridge_unity.log"
DEFAULT_READY = REPO_ROOT / "results" / "unity" / "bridge_ready.json"
DEFAULT_SCREENSHOT = REPO_ROOT / "results" / "unity" / "bridge_screenshot.png"

ACTION_SEQUENCE = [
    "move_to_keycard",
    "collect_keycard",
    "move_to_door",
    "open_door",
    "move_to_checkpoint",
    "activate_checkpoint",
    "move_to_hazard",
    "die_and_respawn",
    "extract",
]


class SmokeFailure(RuntimeError):
    """A machine-checkable smoke assertion failed."""


class BridgeClient:
    def __init__(self, host: str, port: int, timeout_seconds: float) -> None:
        self._socket = socket.create_connection((host, port), timeout=timeout_seconds)
        self._reader = self._socket.makefile("r", encoding="utf-8", newline="\n")
        self._writer = self._socket.makefile("w", encoding="utf-8", newline="\n")

    def request(self, command: str, action: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"command": command}
        if action is not None:
            payload["action"] = action

        self._writer.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._writer.flush()

        line = self._reader.readline()
        if not line:
            raise SmokeFailure(f"Bridge closed before responding to {command}.")

        response = json.loads(line)
        if not response.get("ok", False):
            raise SmokeFailure(
                f"Bridge command {command!r} failed: {response.get('error', '<no error>')}"
            )

        return response

    def close(self) -> None:
        for handle in (self._reader, self._writer):
            try:
                handle.close()
            except OSError:
                pass

        try:
            self._socket.close()
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for Unity bridge startup and shutdown.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def observation(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("observation")
    require(isinstance(value, dict), "Bridge response missing observation.")
    return value


def wait_for_ready(
    process: subprocess.Popen[bytes],
    ready_path: Path,
    timeout_seconds: float,
) -> tuple[str, int]:
    deadline = time.monotonic() + timeout_seconds
    last_json_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeFailure(f"Unity exited before bridge ready: code={process.returncode}")

        if ready_path.exists():
            try:
                data = json.loads(ready_path.read_text(encoding="utf-8"))
                host = data["host"]
                port = int(data["port"])
                return host, port
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_json_error = exc

        time.sleep(0.2)

    suffix = f" Last ready-file error: {last_json_error}" if last_json_error else ""
    raise SmokeFailure(f"Timed out waiting for bridge ready file: {ready_path}.{suffix}")


def assert_png(path: Path) -> None:
    require(path.exists(), f"Screenshot missing: {path}")
    header = path.read_bytes()[:8]
    require(header == b"\x89PNG\r\n\x1a\n", f"Screenshot is not a PNG: {path}")


def drive_checkpoint_flow(client: BridgeClient) -> None:
    reset = client.request("reset")
    reset_obs = observation(reset)
    require(reset_obs.get("player_zone") == "start", "Reset should start player in start zone.")
    require(reset_obs.get("door_open") is False, "Reset should start with the door closed.")
    print("PASS bridge reset")

    final_obs: dict[str, Any] | None = None
    for action in ACTION_SEQUENCE:
        response = client.request("action", action)
        final_obs = observation(response)
        print(f"PASS action {action}")

    require(final_obs is not None, "Action sequence produced no final observation.")
    require(
        final_obs.get("extraction_reached") is True,
        "Expected extraction after checkpoint respawn; "
        f"reason={final_obs.get('failure_reason', '')}",
    )
    require(
        final_obs.get("progression_softlock") is False,
        f"Unexpected progression_softlock: {final_obs.get('failure_reason', '')}",
    )
    print("PASS extraction reached")

    debug_response = client.request("debug_state")
    debug_state = debug_response.get("debug_state")
    require(isinstance(debug_state, dict), "debug_state response missing debug_state object.")
    require(debug_state.get("door_open") is True, "debug_state should record door_open: true.")
    require(
        debug_state.get("extraction_reached") is True,
        "debug_state should record extraction_reached: true.",
    )
    require(
        debug_state.get("progression_softlock") is False,
        "debug_state should record progression_softlock: false.",
    )
    print("PASS debug_state")

    screenshot_response = client.request("screenshot")
    screenshot_path = Path(screenshot_response.get("screenshot_path") or DEFAULT_SCREENSHOT)
    assert_png(screenshot_path)
    print(f"PASS screenshot {screenshot_path}")

    trace_response = client.request("trace")
    trace = trace_response.get("trace")
    require(isinstance(trace, list), "trace response missing trace list.")
    for action in ACTION_SEQUENCE:
        require(f"action:{action}" in trace, f"Trace missing action:{action}")
    print(f"PASS trace events={len(trace)}")


def run_unity(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    unity_exe = Path(args.unity)
    project_path = Path(args.project)
    results_path = Path(args.results)
    log_path = Path(args.log)
    ready_path = Path(args.ready)

    if not unity_exe.exists():
        raise SmokeFailure(f"unity_exe_missing: {unity_exe}")
    if not project_path.exists():
        raise SmokeFailure(f"project_missing: {project_path}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (results_path, log_path, ready_path, DEFAULT_SCREENSHOT):
        if path.exists():
            path.unlink()

    env = os.environ.copy()
    env["GATE2_BRIDGE_HOST"] = "1"
    env["GATE2_BRIDGE_READY_FILE"] = str(ready_path)
    env["GATE2_BRIDGE_TIMEOUT_SECONDS"] = str(max(args.timeout, 1.0))

    command = [
        str(unity_exe),
        "-batchmode",
        "-runTests",
        "-projectPath",
        str(project_path),
        "-testPlatform",
        "PlayMode",
        "-testCategory",
        "Gate2Bridge",
        "-testResults",
        str(results_path),
        "-logFile",
        str(log_path),
    ]

    print("Running Unity runtime bridge smoke...")
    print(" ".join(command))
    return subprocess.Popen(command, env=env)


def wait_for_unity(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
    log_path: Path,
) -> int:
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        print_log_tail(log_path)
        raise SmokeFailure("Unity bridge host did not exit before timeout.")


def main() -> int:
    args = parse_args()
    ready_path = Path(args.ready)
    results_path = Path(args.results)
    log_path = Path(args.log)

    process: subprocess.Popen[bytes] | None = None
    client: BridgeClient | None = None
    smoke_failed = False
    connected = False

    try:
        process = run_unity(args)
        host, port = wait_for_ready(process, ready_path, args.timeout)
        print(f"PASS bridge ready {host}:{port}")
        client = BridgeClient(host, port, args.timeout)
        connected = True
        drive_checkpoint_flow(client)
    except SmokeFailure as exc:
        smoke_failed = True
        print(f"FAIL {exc}")
    except (OSError, json.JSONDecodeError) as exc:
        smoke_failed = True
        print(f"FAIL bridge_io_error: {exc}")
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
        print("FAIL Unity runtime bridge smoke failed.")
        return 1

    print(f"PASS Unity runtime bridge smoke passed. Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
