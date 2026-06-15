import copy

import pytest

from scripts import validate_gate_artifacts as gate8


def test_gate8_golden_fixtures_validate():
    for check in gate8.FIXTURE_CHECKS:
        schema = gate8.load_json(check.schema_path)
        artifact = gate8.load_artifact(check)
        gate8.validate(artifact, schema)


def test_bug_report_schema_requires_trace():
    schema = gate8.load_json(gate8.SCHEMAS / "bug_report.schema.json")
    artifact = gate8.load_json(gate8.FIXTURES / "gate3_progression_softlock_report.json")
    artifact = copy.deepcopy(artifact)
    artifact.pop("trace")

    with pytest.raises(gate8.ValidationError, match="trace"):
        gate8.validate(artifact, schema)


def test_visual_evidence_role_must_stay_supporting_only():
    schema = gate8.load_json(gate8.SCHEMAS / "visual_evidence.schema.json")
    artifact = gate8.load_json(gate8.FIXTURES / "gate5_visual_evidence.json")
    artifact = copy.deepcopy(artifact)
    artifact["visual_evidence_role"] = "sole_oracle"

    with pytest.raises(gate8.ValidationError, match="supporting_evidence_only"):
        gate8.validate(artifact, schema)


def test_parse_unity_results_normalizes_nunit_xml():
    normalized = gate8.parse_unity_results(gate8.FIXTURES / "unity_minimal_results.xml")

    assert normalized["runner"] == "Unity"
    assert normalized["test_platform"] == "PlayMode"
    assert normalized["result"] == "Passed"
    assert normalized["total"] == 1
    assert normalized["cases"] == [
        {
            "name": "Example.Test",
            "result": "Passed",
            "message": "",
        }
    ]
