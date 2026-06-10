# Gate 2 Design - Python runtime bridge smoke

> Scope: Gate 2 of `Doc/project-direction.md`. Gate 1 proved the
> checkpoint-softlock mechanic can pass normally and fail with the injected
> `door-not-persisted` bug. Gate 2 proves Python can drive that same running
> Unity fixture through a runtime bridge, without LLMs or the Gameplay Agent.

## Goal

Implement the smallest project-owned Unity runtime bridge:

```text
Python script -> localhost JSONL/TCP -> Unity PlayMode bridge host -> checkpoint fixture
```

The Python smoke must drive the full Gate 1 flow step by step:

```text
reset -> collect keycard -> open door -> activate checkpoint ->
die/respawn -> extract -> assert debug_state + trace + screenshot
```

## Hard PASS Criteria

All criteria are command-line verifiable with the Editor closed:

1. `python scripts/unity_smoke.py` starts Unity, connects to the bridge, drives the
   checkpoint flow, prints `PASS`, and exits `0`.
2. The smoke asserts `debug_state` fields:
   `door_open: true`, `extraction_reached: true`, and
   `progression_softlock: false`.
3. The smoke requests `screenshot` and verifies a PNG artifact exists.
4. The smoke requests `trace` and verifies the driven action sequence is recorded.
5. With `GATE1_BUG_DOOR_NOT_PERSISTED=1`, the same smoke reports `FAIL` and exits
   nonzero because extraction becomes a `progression_softlock`.
6. Existing Gate 0/1 PlayMode tests remain green via `python scripts/run_unity_tests.py`.
7. The v1 Python baseline remains green: `.venv\Scripts\python.exe -m pytest
   --basetemp .pytest_tmp\gate2-v1`.

## Bridge Contract

Gate 2 implements only the runtime primitives needed by the adapter contract:

- `reset`: rebuild the in-memory checkpoint fixture.
- `action`: execute one named mechanic action.
- `observe`: return the current Unity state without changing it.
- `debug_state`: export and return `results/unity/debug_state.json`.
- `screenshot`: render a deterministic camera/RenderTexture PNG artifact.
- `trace`: return the bridge action/event trace.

Commands and responses are newline-delimited JSON. The protocol is intentionally
thin and local-only; it is not a general game automation framework yet.

Allowed Gate 2 action names:

- `move_to_keycard`
- `collect_keycard`
- `move_to_door`
- `open_door`
- `move_to_checkpoint`
- `activate_checkpoint`
- `move_to_hazard`
- `die_and_respawn`
- `extract`

## Unity Host

Unity exposes the bridge through a PlayMode host test that only starts when
`GATE2_BRIDGE_HOST=1`. In normal `scripts/run_unity_tests.py` runs the host test
returns immediately, so the regular PlayMode suite stays non-interactive.

The host test is only a command-line process wrapper. The fixture state machine,
debug export, screenshot export, and trace live in runtime code under
`unity/GameTestFixture/Assets/Scripts/`.

## Python Smoke

`scripts/unity_smoke.py` is the only Gate 2 entry point. It:

1. Launches the pinned Unity editor in batchmode PlayMode with
   `GATE2_BRIDGE_HOST=1`.
2. Waits for a bridge-ready file and opens the TCP connection.
3. Sends the fixed checkpoint action sequence.
4. Asserts observe/debug_state/screenshot/trace results.
5. Sends `shutdown`, parses Unity test results, and exits nonzero on any Unity or
   smoke assertion failure.

## Scope Fence

Gate 2 does not add:

- Gameplay Agent / reflection / LLM / VLM.
- Goal-level Gherkin for Unity.
- MCP dependency or editor automation as a runtime channel.
- Coverage, mutation framework, bug-to-regression generation, or multi-agent
  orchestration.
- Authored scene or FPS template.

Gate 3 may wrap this bridge with the reused v1 Gameplay Agent. Gate 2 only proves
the live runtime channel is real.
