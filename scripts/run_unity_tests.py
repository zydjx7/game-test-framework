"""Run the Gate 0 Unity PlayMode smoke suite and print machine-readable verdicts."""

from __future__ import annotations

import argparse
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = REPO_ROOT / "unity" / "GameTestFixture"
DEFAULT_UNITY_EXE = Path(r"E:\unity\2022.3.12f1\Editor\Unity.exe")
DEFAULT_RESULTS = REPO_ROOT / "results" / "unity" / "results.xml"
DEFAULT_LOG = REPO_ROOT / "results" / "unity" / "unity.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unity",
        default=os.environ.get("UNITY_EXE", str(DEFAULT_UNITY_EXE)),
        help="Path to Unity.exe. Defaults to UNITY_EXE or the Gate 0 pinned editor.",
    )
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
        help="Unity project path. Defaults to unity/GameTestFixture.",
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="NUnit3 XML results output path.",
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG),
        help="Unity Editor log output path.",
    )
    return parser.parse_args()


def print_log_tail(log_path: Path, line_count: int = 80) -> None:
    if not log_path.exists():
        return

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"\n--- Unity log tail ({log_path}) ---")
    for line in lines[-line_count:]:
        print(line)


def parse_results(results_path: Path) -> tuple[int, int]:
    if not results_path.exists():
        print(f"FAIL results_xml_missing: {results_path}")
        return 0, 1

    root = ET.parse(results_path).getroot()
    cases = root.findall(".//test-case")
    failures = 0

    for case in cases:
        name = case.attrib.get("fullname") or case.attrib.get("name", "<unnamed>")
        result = case.attrib.get("result", "Unknown")
        if result == "Passed":
            print(f"PASS {name}")
            continue

        failures += 1
        message = case.findtext("./failure/message", default="").strip()
        suffix = f": {message}" if message else ""
        print(f"FAIL {name} ({result}){suffix}")

    total = int(root.attrib.get("total", len(cases) or 0))
    passed = int(root.attrib.get("passed", 0))
    failed = int(root.attrib.get("failed", failures))
    skipped = int(root.attrib.get("skipped", 0))
    result = root.attrib.get("result", "Unknown")

    if not cases:
        failures += 1
        print("FAIL no_test_cases_found")

    print(
        "SUMMARY "
        f"result={result} total={total} passed={passed} failed={failed} skipped={skipped}"
    )

    if result != "Passed" or failed > 0:
        failures += max(failed, 1)

    return len(cases), failures


def main() -> int:
    args = parse_args()
    unity_exe = Path(args.unity)
    project_path = Path(args.project)
    results_path = Path(args.results)
    log_path = Path(args.log)

    if not unity_exe.exists():
        print(f"FAIL unity_exe_missing: {unity_exe}")
        return 2
    if not project_path.exists():
        print(f"FAIL project_missing: {project_path}")
        return 2

    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()
    if log_path.exists():
        log_path.unlink()

    command = [
        str(unity_exe),
        "-batchmode",
        "-runTests",
        "-projectPath",
        str(project_path),
        "-testPlatform",
        "PlayMode",
        "-testResults",
        str(results_path),
        "-logFile",
        str(log_path),
    ]

    print("Running Unity PlayMode tests...")
    print(" ".join(command))
    completed = subprocess.run(command, check=False)

    _, failures = parse_results(results_path)
    if completed.returncode != 0:
        print(f"FAIL unity_exit_code={completed.returncode}")
        failures += 1
        print_log_tail(log_path)

    if failures:
        print("FAIL Unity PlayMode suite failed.")
        return 1

    print(f"PASS Unity PlayMode suite passed. Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
