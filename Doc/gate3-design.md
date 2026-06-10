# Gate 3 Design - Gameplay Agent checkpoint report

> Scope: Gate 3 of `Doc/project-direction.md`. Gate 2 proved Python can drive
> Unity through the runtime bridge. Gate 3 wraps that bridge with the reused v1
> Gameplay Agent shape: goal-level Gherkin, action library, decider, reflection
> ladder, and bug report.

## Goal

Run a Unity checkpoint goal through the agent layer:

```text
goal-level Gherkin -> decider -> Unity action library -> Gate 2 bridge ->
observe/debug_state/screenshot/trace -> reflective bug report
```

Normal build:

```text
agent reaches extraction -> PASS
```

Injected bug build:

```text
GATE1_BUG_DOOR_NOT_PERSISTED=1 ->
agent reports progression_softlock with debug_state + screenshot + trace -> PASS
```

## Hard PASS Criteria

All criteria are command-line verifiable with the Editor closed:

1. `python scripts/unity_agent_smoke.py` prints `PASS`, exits `0`, and records a
   normal agent run that reaches extraction.
2. `GATE1_BUG_DOOR_NOT_PERSISTED=1 python scripts/unity_agent_smoke.py
   --expect progression_softlock` prints `PASS`, exits `0`, and records a bug
   report with `failure_type: progression_softlock`.
3. The bug report includes:
   - `debug_state` with `progression_softlock: true`;
   - a PNG screenshot path that exists;
   - bridge trace entries for the checkpoint flow.
4. Running the bug build without `--expect progression_softlock` must fail
   nonzero, proving the normal gate cannot silently accept the bug.
5. Existing Gate 0-2 Unity checks remain green:
   - `python scripts/run_unity_tests.py`
   - `python scripts/unity_smoke.py`
6. The v1 Python baseline remains green:
   `.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp\gate3-v1`.

## Agent Boundary

Gate 3 reuses the v1 agent core by import only:

- `agent.parse_goals`
- `agent.run_reflective_agent`
- the diagnostic recovery ladder from `agent.graph`
- `actions.result.snapshot_result`

Do not edit `agent/`, `actions/`, `perception/`, `env/`, `toy_fps/`, `Code/`, or
`src/` for this gate.

## Unity Adapter

Add a new Unity-specific adapter package outside the v1 source directories:

- `unity_agent/goals.feature`: one goal, `gameplay_checkpoint_no_softlock`.
- `unity_agent/adapter.py`: bridge-backed primitives, a Unity state perceptor,
  and high-level action templates.

The action templates stay goal-level rather than raw bridge calls:

- `collect_keycard`
- `open_security_door`
- `activate_checkpoint`
- `die_and_respawn`
- `attempt_extraction`

The smoke may use a deterministic decider/reflector so Gate 3 remains local and
repeatable. This still exercises the real agent loop and reflection ladder; LLM
API reliability is not the oracle for this Unity gate.

## Bug Report Shape

`scripts/unity_agent_smoke.py` writes a JSON report under `results/unity/` with:

- `status`: `success` or `bug_reported`
- `failure_type`: `progression_softlock` when applicable
- `goal`
- `history`
- `debug_state`
- `screenshot_path`
- `trace`

The report is a developer-facing evidence bundle, not a VLM judgment. VLM visual
evidence starts at Gate 5.

## Scope Fence

Gate 3 does not add:

- Spec-to-Test / Test Plan IR / generated tests.
- VLM interpretation.
- Coverage or mutation infrastructure.
- MCP as runtime channel.
- Multi-agent orchestration.
- Authored scene or FPS template.

Gate 4 may start Spec-to-Test only after this Gate 3 vertical slice is stable.
