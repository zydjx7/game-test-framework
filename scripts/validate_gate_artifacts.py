"""Gate 8 artifact/schema validator.

Validates committed golden fixtures and the latest local Unity-track artifacts
against the small JSON Schema subset used by this repository. The script has no
third-party dependency so it can run under both system Python and .venv.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = REPO_ROOT / "schemas"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "artifacts"
UNITY_RESULTS = REPO_ROOT / "results" / "unity"


class ValidationError(ValueError):
    """Raised when an artifact violates its schema."""


@dataclass(frozen=True)
class ArtifactCheck:
    label: str
    artifact_path: Path | None
    schema_path: Path
    loader: str = "json"
    pointer: str | None = None


FIXTURE_CHECKS: Sequence[ArtifactCheck] = (
    ArtifactCheck("fixture:gate3_progression_softlock_report", FIXTURES / "gate3_progression_softlock_report.json", SCHEMAS / "bug_report.schema.json"),
    ArtifactCheck("fixture:gate5_visual_evidence", FIXTURES / "gate5_visual_evidence.json", SCHEMAS / "visual_evidence.schema.json"),
    ArtifactCheck("fixture:gate6_regression_plan", FIXTURES / "gate6_regression_plan.json", SCHEMAS / "regression_plan_ir.schema.json"),
    ArtifactCheck("fixture:gate7_presentation_mismatch_report", FIXTURES / "gate7_presentation_mismatch_report.json", SCHEMAS / "bug_report.schema.json"),
    ArtifactCheck("fixture:gate7_embedded_visual_evidence", FIXTURES / "gate7_presentation_mismatch_report.json", SCHEMAS / "visual_evidence.schema.json", pointer="visual_evidence"),
    ArtifactCheck("fixture:gate7_regression_plan", FIXTURES / "gate7_regression_plan.json", SCHEMAS / "regression_plan_ir.schema.json"),
    ArtifactCheck("fixture:gate4_test_plan", FIXTURES / "gate4_test_plan.json", SCHEMAS / "test_plan_ir.schema.json"),
    ArtifactCheck("fixture:unity_playmode_run_result", FIXTURES / "unity_playmode_run_result.json", SCHEMAS / "run_result.schema.json"),
    ArtifactCheck("fixture:validate_artifacts_tool_result", FIXTURES / "validate_artifacts_tool_result.json", SCHEMAS / "tool_result.schema.json"),
)

LATEST_CHECKS: Sequence[ArtifactCheck] = (
    ArtifactCheck("latest:gate6_source_agent_report", UNITY_RESULTS / "gate6_source_agent_report.json", SCHEMAS / "bug_report.schema.json"),
    ArtifactCheck("latest:gate7_presentation_bug_report", UNITY_RESULTS / "gate7_presentation_bug_report.json", SCHEMAS / "bug_report.schema.json"),
    ArtifactCheck("latest:gate7_embedded_visual_evidence", UNITY_RESULTS / "gate7_presentation_bug_report.json", SCHEMAS / "visual_evidence.schema.json", pointer="visual_evidence"),
    ArtifactCheck("latest:gate5_visual_evidence", UNITY_RESULTS / "vlm_evidence_report.json", SCHEMAS / "visual_evidence.schema.json"),
    ArtifactCheck("latest:gate4_test_plan", UNITY_RESULTS / "spec_to_test_plan.json", SCHEMAS / "test_plan_ir.schema.json"),
    ArtifactCheck("latest:gate6_regression_plan", UNITY_RESULTS / "bug_to_regression_plan.json", SCHEMAS / "regression_plan_ir.schema.json"),
    ArtifactCheck("latest:gate7_regression_plan", UNITY_RESULTS / "gate7_presentation_regression_plan.json", SCHEMAS / "regression_plan_ir.schema.json"),
    ArtifactCheck("latest:unity_playmode_run_result", UNITY_RESULTS / "results.xml", SCHEMAS / "run_result.schema.json", loader="unity_xml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Validate committed fixtures only; skip results/unity latest artifacts.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Validate latest results/unity artifacts only; skip committed fixtures.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_pointer(instance: Any, pointer: str | None) -> Any:
    if not pointer:
        return instance

    current = instance
    for part in pointer.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValidationError(f"missing pointer field '{pointer}'")
        current = current[part]
    return current


def load_artifact(check: ArtifactCheck) -> Any:
    if check.artifact_path is None:
        raise ValidationError("artifact path is missing")
    if not check.artifact_path.exists():
        raise ValidationError(f"missing artifact: {relative(check.artifact_path)}")

    if check.loader == "json":
        return resolve_pointer(load_json(check.artifact_path), check.pointer)
    if check.loader == "unity_xml":
        return parse_unity_results(check.artifact_path)

    raise ValidationError(f"unknown loader: {check.loader}")


def parse_unity_results(path: Path) -> Dict[str, Any]:
    root = ET.parse(path).getroot()
    cases = []
    for case in root.findall(".//test-case"):
        cases.append(
            {
                "name": case.attrib.get("fullname") or case.attrib.get("name", "<unnamed>"),
                "result": case.attrib.get("result", "Unknown"),
                "message": case.findtext("./failure/message", default="").strip(),
            }
        )

    return {
        "runner": "Unity",
        "test_platform": find_unity_property(root, "platform") or "Unknown",
        "test_category": None,
        "result": root.attrib.get("result", "Unknown"),
        "total": int(root.attrib.get("total", len(cases))),
        "passed": int(root.attrib.get("passed", count_cases(cases, "Passed"))),
        "failed": int(root.attrib.get("failed", count_cases(cases, "Failed"))),
        "skipped": int(root.attrib.get("skipped", count_cases(cases, "Skipped"))),
        "cases": cases,
    }


def find_unity_property(root: ET.Element, name: str) -> str | None:
    for prop in root.findall(".//property"):
        if prop.attrib.get("name") == name:
            return prop.attrib.get("value")
    return None


def count_cases(cases: Iterable[Dict[str, str]], result: str) -> int:
    return sum(1 for case in cases if case.get("result") == result)


def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> None:
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                validate(instance, option, path)
                return
            except ValidationError as exc:
                errors.append(str(exc))
        raise ValidationError(f"{path}: did not match anyOf: {'; '.join(errors)}")

    if "const" in schema and instance != schema["const"]:
        raise ValidationError(f"{path}: expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise ValidationError(f"{path}: expected one of {schema['enum']!r}, got {instance!r}")

    if "type" in schema:
        require_type(instance, schema["type"], path)

    if "minimum" in schema and is_number(instance) and instance < schema["minimum"]:
        raise ValidationError(f"{path}: expected >= {schema['minimum']}, got {instance!r}")

    if isinstance(instance, dict):
        validate_object(instance, schema, path)
    elif isinstance(instance, list):
        validate_array(instance, schema, path)


def validate_object(instance: Dict[str, Any], schema: Dict[str, Any], path: str) -> None:
    for field in schema.get("required", []):
        if field not in instance:
            raise ValidationError(f"{path}: missing required field '{field}'")

    properties = schema.get("properties", {})
    for key, value in instance.items():
        if key in properties:
            validate(value, properties[key], f"{path}.{key}")
        elif schema.get("additionalProperties", True) is False:
            raise ValidationError(f"{path}: additional property '{key}' is not allowed")


def validate_array(instance: List[Any], schema: Dict[str, Any], path: str) -> None:
    if "items" not in schema:
        return
    item_schema = schema["items"]
    for index, item in enumerate(instance):
        validate(item, item_schema, f"{path}[{index}]")


def require_type(instance: Any, expected: str | Sequence[str], path: str) -> None:
    expected_types = [expected] if isinstance(expected, str) else list(expected)
    if any(matches_type(instance, expected_type) for expected_type in expected_types):
        return
    raise ValidationError(f"{path}: expected type {expected_types!r}, got {type(instance).__name__}")


def matches_type(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return is_number(instance)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    raise ValidationError(f"unsupported schema type: {expected}")


def is_number(instance: Any) -> bool:
    return isinstance(instance, (int, float)) and not isinstance(instance, bool)


def run_check(check: ArtifactCheck) -> Dict[str, str]:
    try:
        schema = load_json(check.schema_path)
        artifact = load_artifact(check)
        validate(artifact, schema)
    except (OSError, json.JSONDecodeError, ET.ParseError, ValidationError, ValueError) as exc:
        message = str(exc)
        print(f"FAIL {check.label}: {message}", flush=True)
        return {
            "artifact": check.label,
            "schema": relative(check.schema_path),
            "status": "failed",
            "message": message,
        }

    print(f"PASS {check.label} -> {relative(check.schema_path)}", flush=True)
    return {
        "artifact": check.label,
        "schema": relative(check.schema_path),
        "status": "passed",
        "message": "",
    }


def relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def selected_checks(args: argparse.Namespace) -> List[ArtifactCheck]:
    if args.fixtures_only and args.latest_only:
        raise SystemExit("--fixtures-only and --latest-only cannot be combined")
    if args.fixtures_only:
        return list(FIXTURE_CHECKS)
    if args.latest_only:
        return list(LATEST_CHECKS)
    return list(FIXTURE_CHECKS) + list(LATEST_CHECKS)


def main() -> int:
    args = parse_args()
    checks = [run_check(check) for check in selected_checks(args)]
    failed = [check for check in checks if check["status"] != "passed"]

    tool_result = {
        "tool": "validate_gate_artifacts",
        "status": "failure" if failed else "success",
        "exit_code": 1 if failed else 0,
        "checks": checks,
    }

    try:
        validate(tool_result, load_json(SCHEMAS / "tool_result.schema.json"))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"FAIL runtime_tool_result_contract: {exc}", flush=True)
        return 1

    print("PASS runtime_tool_result_contract", flush=True)
    if failed:
        print(f"FAIL Gate 8 artifact schema validation failed checks={len(checks)} failures={len(failed)}", flush=True)
        return 1

    print(f"PASS Gate 8 artifact schema validation passed checks={len(checks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
