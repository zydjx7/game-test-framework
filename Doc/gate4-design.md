# Gate 4 Design - Spec-to-Test template slice

> Scope: Gate 4 of `Doc/project-direction.md`. Gate 3 proved the Gameplay Agent
> can reach extraction normally and report `progression_softlock` on the injected
> bug. Gate 4 introduces the other side of the final loop: a deterministic
> Spec-to-Test slice that turns the checkpoint requirement into a structured Test
> Plan IR and template-rendered Unity tests.

## Goal

Build the smallest Spec-to-Test Agent path:

```text
checkpoint requirement -> Test Plan IR -> templated NUnit/EditMode test
                                      -> templated PlayMode test
                                      -> Unity CLI PASS/FAIL
```

The output must be **template-rendered**, not free-form C# generation. This keeps
Gate 4 deterministic and reviewable while proving the system can cross the
requirement-to-tests boundary.

## Hard PASS Criteria

All criteria are command-line verifiable with the Editor closed:

1. `python scripts/spec_to_test_smoke.py` reads the checkpoint requirement,
   emits a Test Plan IR JSON artifact, renders Unity test files, runs generated
   EditMode + PlayMode tests through Unity CLI, prints `PASS`, and exits `0`.
2. The Test Plan IR includes at least:
   - one `unit` or component-level case checking checkpoint state persistence;
   - one `playmode` case checking checkpoint respawn -> extraction.
3. The rendered Unity tests are committed under `unity/GameTestFixture/Assets/Tests/`
   and are not handwritten ad hoc tests; they include a generated-file header.
4. `python scripts/run_unity_tests.py` remains green for the existing PlayMode
   suite.
5. `python scripts/unity_agent_smoke.py` remains green for the Gate 3 vertical
   slice.
6. The v1 Python baseline remains green:
   `.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp\gate4-v1`.

## IR Boundary

Gate 4 defines only the fields the templates need:

- `plan_id`
- `requirement_id`
- `requirement`
- `cases[]`
  - `id`
  - `layer`: `unit` or `playmode`
  - `title`
  - `template`
  - `assertions[]`

No natural-language C# synthesis, no code repair loop, no mutation coverage, and
no LLM dependency are added at this gate. A later LLM can choose or fill IR, but
the compiler-facing surface remains template-rendered.

## Rendered Tests

Gate 4 renders:

- **EditMode/component test**: checkpoint capture/restore preserves the
  post-door-open state and the consumed keycard state.
- **PlayMode test**: the same checkpoint flow reaches extraction and records
  non-softlocked debug state.

The generated tests may duplicate a small part of Gate 1/2 behavior; that is
acceptable because Gate 4's proof is the pipeline from requirement -> IR ->
compiled tests, not new game behavior.

## Scope Fence

Gate 4 does not add:

- VLM evidence or image interpretation.
- Bug-to-regression FAIL-on-bug/PASS-on-fix semantics; that is Gate 6.
- Coverage or mutation infrastructure.
- Multi-agent orchestration.
- MCP as runtime channel.
- Authored scenes, FPS templates, or new gameplay mechanics.

Gate 5 may add VLM visual evidence after this template-generated test path is
stable.
