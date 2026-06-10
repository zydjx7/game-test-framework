# Gate 5 Design - VLM visual evidence

> Scope: Gate 5 of `Doc/project-direction.md`. Gate 3 already produces a
> developer-facing Gameplay Agent report with `debug_state`, screenshot, and
> trace. Gate 5 adds structured visual evidence beside that report. The visual
> layer never owns pass/fail or failure classification.

## Goal

Build the smallest VLM evidence path:

```text
Gate 3 bug report + screenshot -> structured visual questions
                               -> structured visual answers
                               -> evidence bundle appended beside debug_state
                               -> command-line PASS/FAIL smoke
```

The evidence path must prove the contract that a VLM can answer player-visible
questions about the screenshot while the authoritative verdict remains grounded
in `debug_state`, trace, and the Gameplay Agent report.

## Hard PASS Criteria

All criteria are command-line verifiable with the Editor closed:

1. `python scripts/vlm_evidence_smoke.py` runs the Gate 3 bug scenario, requires a
   `progression_softlock` report, verifies the screenshot PNG exists, appends
   visual evidence, prints `PASS`, and exits `0`.
2. The output evidence JSON includes:
   - the source `agent_report` path;
   - the source screenshot path;
   - the verdict source (`debug_state+trace+agent_report`, not VLM);
   - structured question/answer records with confidence and backend metadata.
3. The smoke asserts that VLM evidence is present, but that the report's
   `failure_type` is still `progression_softlock` from Gate 3.
4. Existing gate checks remain green:
   - `python scripts/run_unity_tests.py`;
   - `python scripts/unity_smoke.py`;
   - `python scripts/unity_agent_smoke.py`;
   - `python scripts/spec_to_test_smoke.py`.
5. The v1 Python baseline remains green:
   `.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp\gate5-v1`.

## Evidence Schema

Gate 5 writes a JSON artifact under `results/unity/`:

```json
{
  "schema_version": 1,
  "source_report_path": "results/unity/agent_report.json",
  "screenshot_path": "results/unity/bridge_screenshot.png",
  "verdict_source": "debug_state+trace+agent_report",
  "failure_type": "progression_softlock",
  "visual_evidence_role": "supporting_evidence_only",
  "questions": [
    {
      "id": "door_blocks_extraction",
      "prompt": "Is the extraction marker/player route blocked by a closed door?",
      "answer": "yes",
      "confidence": 0.95,
      "backend": "fixture-vlm",
      "rationale": "The debug context reports the door closed after respawn."
    }
  ]
}
```

The evidence can also be embedded into a copy of the Gate 3 report, but the Gate
5 smoke's hard artifact is the standalone evidence JSON so it stays easy to
inspect and diff.

## VLM Backend Policy

The implementation uses a small backend interface:

```text
answer(question, screenshot_path, context) -> structured answer
```

Gate 5's required smoke uses a deterministic local fixture backend. This avoids
making the pass/fail path depend on API keys, network latency, model drift, or
regional provider availability. A real VLM backend can be swapped in later for
demonstration or richer evaluation, but it must feed the same schema and must not
replace the runtime oracle.

## Structured Questions

Initial questions target the checkpoint softlock screenshot and its context:

- `door_blocks_extraction`: Is the extraction route blocked by a closed door?
- `player_view_blocked`: Does the evidence suggest the player cannot reach the
  extraction marker after respawn?
- `state_visual_consistency`: Is the visual evidence consistent with the
  `debug_state` softlock diagnosis?

The smoke requires all three answers to be present. It does not require the VLM
to discover the bug independently.

## Scope Fence

Gate 5 does not add:

- VLM-only verdicts or image-only pass/fail.
- New Unity mechanics, scenes, prefabs, or gameplay behavior.
- Real provider credentials, network calls, or model-selection policy.
- Multi-agent orchestration.
- Bug-to-regression generation; that is Gate 6.
- Coverage or mutation infrastructure.

Gate 6 may consume the Gate 3/5 report bundle to render a permanent regression
test that fails on the injected bug and passes on the fixed build.
