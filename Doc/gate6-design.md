# Gate 6 Design - Bug-to-regression loop

> Scope: Gate 6 of `Doc/project-direction.md`. Gate 3/5 already produce a
> machine-readable checkpoint-softlock bug report with debug state, screenshot,
> trace, and optional VLM evidence. Gate 6 closes the first loop: feed that bug
> report back into Spec-to-Test and render a permanent Unity regression test.

## Goal

Build the smallest bug-to-regression path:

```text
Gate 3 bug report -> Regression Test Plan IR -> templated Unity regression test
                  -> run with injected bug -> FAIL
                  -> run fixed build       -> PASS
```

This proves the report is not just a narrative artifact. It becomes executable
regression protection.

## Hard PASS Criteria

All criteria are command-line verifiable with the Editor closed:

1. `python scripts/bug_to_regression_smoke.py` runs the Gate 3 injected-bug
   scenario and requires a source report with `failure_type:
   progression_softlock`.
2. The script emits a Regression Test Plan IR JSON under `results/unity/`.
3. The script template-renders a Unity PlayMode regression test under
   `unity/GameTestFixture/Assets/Tests/PlayMode/` with:
   - a generated-file header;
   - Unity category `Gate6Regression`;
   - assertions that extraction is reached and `progression_softlock` is false.
4. The same generated test is run twice through Unity CLI:
   - with `GATE1_BUG_DOOR_NOT_PERSISTED=1`, it must FAIL nonzero;
   - with the bug env unset, it must PASS.
5. Existing gate checks remain green:
   - `python scripts/run_unity_tests.py`;
   - `python scripts/unity_smoke.py`;
   - `python scripts/unity_agent_smoke.py`;
   - `python scripts/spec_to_test_smoke.py`;
   - `python scripts/vlm_evidence_smoke.py`.
6. The v1 Python baseline remains green:
   `.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp\gate6-v1`.

## IR Boundary

Gate 6 defines only the fields the regression template needs:

- `plan_id`
- `source_report_path`
- `source_failure_type`
- `source_goal`
- `cases[]`
  - `id`
  - `layer`: `playmode`
  - `title`
  - `template`
  - `repro_actions[]`
  - `assertions[]`

The planner validates the source report but does not synthesize arbitrary C#.
Only the checkpoint-softlock report family is supported at this gate.

## Generated Regression Test

The generated PlayMode test reproduces the same action flow recorded in the bug
report:

```text
move_to_keycard -> collect_keycard -> move_to_door -> open_door ->
move_to_checkpoint -> activate_checkpoint -> move_to_hazard ->
die_and_respawn -> extract
```

It then asserts the fixed behavior:

- `finalObservation.extraction_reached == true`
- `finalObservation.progression_softlock == false`
- `debugState.door_open == true`
- `debugState.extraction_reached == true`
- `debugState.progression_softlock == false`

Therefore the injected door-not-persisted bug makes the generated regression
test fail, and the fixed/default build passes.

## Relation to Gate 4

Gate 4 proved requirement -> IR -> tests. Gate 6 proves bug report -> IR ->
regression test. Both use deterministic templates. Gate 6 may use a separate
`bug_to_regression/` package to keep the source-report validation and regression
IR distinct from requirement planning.

## Scope Fence

Gate 6 does not add:

- Coverage infrastructure.
- Mutation testing framework or dashboards. The Gate 1 env toggle is only the
  controlled bug/fixed build switch for this one regression.
- Free-form C# generation.
- New Unity mechanics, scenes, prefabs, or gameplay behavior.
- VLM-only verdicts.
- Multi-agent orchestration.

After Gate 6, the project has the first complete vertical slice of the intended
dual-agent loop. Later work can broaden bug classes, add real provider-backed
VLM demos, or add orchestration, but not as part of this gate.
