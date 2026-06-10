"""Gate 5 VLM visual-evidence smoke.

Runs the Gate 3 injected-bug scenario, verifies the authoritative
``progression_softlock`` report, then appends structured visual evidence beside
the report. The VLM evidence is supporting context only; it never owns the
verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_unity_tests import DEFAULT_PROJECT, DEFAULT_UNITY_EXE  # noqa: E402
from unity_smoke import SmokeFailure, assert_png  # noqa: E402
from vlm_evidence import FixtureVlmBackend, build_gate5_evidence  # noqa: E402


DEFAULT_AGENT_REPORT = REPO_ROOT / "results" / "unity" / "gate5_agent_report.json"
DEFAULT_AGENT_RESULTS = REPO_ROOT / "results" / "unity" / "gate5_agent_results.xml"
DEFAULT_AGENT_LOG = REPO_ROOT / "results" / "unity" / "gate5_agent_unity.log"
DEFAULT_AGENT_READY = REPO_ROOT / "results" / "unity" / "gate5_agent_bridge_ready.json"
DEFAULT_EVIDENCE = REPO_ROOT / "results" / "unity" / "vlm_evidence_report.json"
DEFAULT_COMBINED = REPO_ROOT / "results" / "unity" / "agent_report_with_vlm_evidence.json"
BUG_ENV_VAR = "GATE1_BUG_DOOR_NOT_PERSISTED"


class VlmEvidenceSmokeFailure(RuntimeError):
    """A machine-checkable Gate 5 assertion failed."""


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
        "--agent-report",
        default=str(DEFAULT_AGENT_REPORT),
        help="Gate 3 source report path produced by this smoke.",
    )
    parser.add_argument(
        "--evidence",
        default=str(DEFAULT_EVIDENCE),
        help="Standalone Gate 5 visual evidence JSON path.",
    )
    parser.add_argument(
        "--combined-report",
        default=str(DEFAULT_COMBINED),
        help="Copy of the Gate 3 report with visual_evidence appended.",
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
        raise VlmEvidenceSmokeFailure(message)


def run_gate3_bug_smoke(args: argparse.Namespace) -> None:
    report_path = Path(args.agent_report)
    for path in (
        report_path,
        Path(args.evidence),
        Path(args.combined_report),
        DEFAULT_AGENT_RESULTS,
        DEFAULT_AGENT_LOG,
        DEFAULT_AGENT_READY,
    ):
        if path.exists():
            path.unlink()

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "unity_agent_smoke.py"),
        "--expect",
        "progression_softlock",
        "--unity",
        str(args.unity),
        "--project",
        str(args.project),
        "--results",
        str(DEFAULT_AGENT_RESULTS),
        "--log",
        str(DEFAULT_AGENT_LOG),
        "--ready",
        str(DEFAULT_AGENT_READY),
        "--report",
        str(report_path),
        "--timeout",
        str(args.timeout),
    ]

    env = os.environ.copy()
    env[BUG_ENV_VAR] = "1"
    print("Running Gate 3 injected-bug smoke for Gate 5 evidence...", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        raise VlmEvidenceSmokeFailure(
            f"Gate 3 bug smoke failed with exit code {completed.returncode}."
        )


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VlmEvidenceSmokeFailure(f"Could not read JSON artifact {path}: {exc}") from exc


def validate_agent_report(report_path: Path, report: Dict[str, Any]) -> Path:
    require(report_path.exists(), f"source agent report missing: {report_path}")
    require(
        report.get("failure_type") == "progression_softlock",
        f"Expected progression_softlock source report, got {report.get('failure_type')}.",
    )
    require(
        report.get("status") == "bug_reported",
        f"Expected bug_reported source status, got {report.get('status')}.",
    )

    debug_state = dict(report.get("debug_state", {}) or {})
    require(debug_state.get("progression_softlock") is True, "debug_state must own softlock verdict.")
    require(debug_state.get("door_open") is False, "debug_state should show the door closed.")
    require(
        debug_state.get("extraction_reached") is False,
        "debug_state should show extraction was not reached.",
    )

    trace = list(report.get("trace", []) or [])
    require("action:die_and_respawn" in trace, "trace missing action:die_and_respawn.")
    require("action:extract" in trace, "trace missing action:extract.")

    screenshot_path = Path(str(report.get("screenshot_path", "")))
    assert_png(screenshot_path)
    print(f"PASS Gate 3 source report {report_path}", flush=True)
    print(f"PASS screenshot {screenshot_path}", flush=True)
    return screenshot_path


def validate_evidence(evidence: Dict[str, Any], report: Dict[str, Any]) -> None:
    require(evidence.get("schema_version") == 1, "visual evidence schema_version must be 1.")
    require(
        evidence.get("verdict_source") == "debug_state+trace+agent_report",
        "visual evidence must keep verdict source outside the VLM.",
    )
    require(
        evidence.get("visual_evidence_role") == "supporting_evidence_only",
        "visual evidence role must be supporting_evidence_only.",
    )
    require(
        evidence.get("failure_type") == report.get("failure_type"),
        "visual evidence must preserve the source failure_type.",
    )

    questions = list(evidence.get("questions", []) or [])
    answers = {entry.get("question_id"): entry for entry in questions}
    for question_id in (
        "door_blocks_extraction",
        "player_view_blocked",
        "state_visual_consistency",
    ):
        entry = answers.get(question_id)
        require(isinstance(entry, dict), f"visual evidence missing {question_id}.")
        require(entry.get("backend") == FixtureVlmBackend.name, f"{question_id} backend mismatch.")
        require(bool(entry.get("answer")), f"{question_id} missing answer.")
        confidence = entry.get("confidence")
        require(isinstance(confidence, (int, float)), f"{question_id} missing numeric confidence.")
        require(0.0 <= float(confidence) <= 1.0, f"{question_id} confidence out of range.")

    require(
        answers["door_blocks_extraction"].get("answer") == "yes",
        "expected visual answer door_blocks_extraction=yes.",
    )
    require(
        answers["player_view_blocked"].get("answer") == "yes",
        "expected visual answer player_view_blocked=yes.",
    )
    require(
        answers["state_visual_consistency"].get("answer") == "consistent",
        "expected visual answer state_visual_consistency=consistent.",
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_path = Path(args.agent_report)
    evidence_path = Path(args.evidence)
    combined_path = Path(args.combined_report)

    try:
        run_gate3_bug_smoke(args)
        report = load_json(report_path)
        validate_agent_report(report_path, report)

        evidence = build_gate5_evidence(
            report_path,
            report,
            backend=FixtureVlmBackend(),
        ).to_dict()
        validate_evidence(evidence, report)
        write_json(evidence_path, evidence)
        print(f"PASS visual_evidence {evidence_path}", flush=True)

        combined = dict(report)
        combined["visual_evidence"] = evidence
        require(
            combined.get("failure_type") == "progression_softlock",
            "combined report must preserve the Gate 3 failure_type.",
        )
        write_json(combined_path, combined)
        print(f"PASS combined_report {combined_path}", flush=True)
    except (VlmEvidenceSmokeFailure, SmokeFailure) as exc:
        print(f"FAIL {exc}", flush=True)
        return 1
    except OSError as exc:
        print(f"FAIL vlm_evidence_io_error: {exc}", flush=True)
        return 1

    print("PASS VLM visual evidence smoke passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
