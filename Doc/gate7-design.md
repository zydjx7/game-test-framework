# Gate 7 Design - Presentation/state-visual mismatch

> Scope: Gate 7 of `Doc/post-gate6-roadmap.md`. Gate 6 closed the first loop for
> a progression-softlock bug. Gate 7 adds a second bug class: player-visible
> presentation mismatch where gameplay logic is correct but the rendered state is
> wrong.

## Goal

Build the smallest presentation bug loop:

```text
logic door open + extraction reachable
visual door appears closed
  -> Gameplay/visual report: presentation_mismatch
  -> screenshot + visual_state + trace evidence
  -> generated regression test
  -> bug build FAILS, fixed build PASSES
```

This proves the system can detect more than progression logic. It also gives
the VLM evidence path a real job while preserving the rule that VLM is never the
sole oracle.

## Bug Toggle

Use a new env var:

```text
GATE7_BUG_DOOR_VISUAL_STUCK_CLOSED=1
```

Expected behavior:

- Gameplay logic stays fixed:
  - `debug_state.door_open == true`
  - `debug_state.extraction_reached == true`
  - `debug_state.progression_softlock == false`
- Presentation is wrong:
  - `visual_state.door_visual_open == false`
  - screenshot shows the door in its closed visual state

The bug must not reuse `GATE1_BUG_DOOR_NOT_PERSISTED`; Gate 7 is a separate bug
family.

## Hard PASS Criteria

All criteria are command-line verifiable with the Editor closed:

1. Existing gates remain green before Gate 7 implementation starts:
   - `python scripts/run_unity_tests.py`
   - `python scripts/unity_smoke.py`
   - `python scripts/unity_agent_smoke.py`
   - `python scripts/bug_to_regression_smoke.py`
2. A new smoke, tentatively `python scripts/presentation_bug_smoke.py`, runs the
   checkpoint flow and prints PASS/FAIL.
3. Fixed/default build:
   - reaches extraction;
   - exports `debug_state`;
   - exports screenshot;
   - exports `visual_state`;
   - reports no `presentation_mismatch`.
4. Bug build with `GATE7_BUG_DOOR_VISUAL_STUCK_CLOSED=1`:
   - reaches extraction;
   - keeps `progression_softlock == false`;
   - reports exactly `failure_type: presentation_mismatch`;
   - never reports `progression_softlock`;
   - includes screenshot, trace, debug_state, and visual_state evidence.
5. VLM visual evidence is appended beside the report, not used as the verdict.
6. A generated Gate 7 regression test is committed under
   `unity/GameTestFixture/Assets/Tests/PlayMode/` with a generated-file header
   and category `Gate7Regression`.
7. The generated Gate 7 regression test:
   - FAILS with `GATE7_BUG_DOOR_VISUAL_STUCK_CLOSED=1`;
   - PASSES with the env unset.
8. The v1 Python baseline remains green:
   `.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp\gate7-v1`.

## Required Unity Surface

Add only the smallest runtime surface needed:

- A visual-state snapshot type, e.g. `CheckpointVisualState`.
- A visual-state exporter/bridge command, e.g. `visual_state`.
- A door visual mismatch toggle that affects the rendered door state but not
  gameplay logic.

`visual_state` must come from the presentation layer, not from a copy or
inference of gameplay `debug_state`. Keep the sources separate:

```text
debug_state.door_open
    comes from gameplay/passability/progression logic

visual_state.door_visual_open
    comes from DoorVisualController / Renderer / material / transform /
    animation-visible state
```

Recommended component boundary:

```text
SecurityDoor / DoorLogicState
    owns passability, door_open, extraction reachability

DoorVisualController / DoorVisualState
    owns mesh/material/color/transform/visible door state
```

`GATE7_BUG_DOOR_VISUAL_STUCK_CLOSED=1` may affect only the visual component. It
must not affect passability, extraction, checkpoint persistence, or
`progression_softlock`.

The fixture may keep using programmatic GameObjects and the existing fixed
camera/RenderTexture screenshot path. Do not add authored scenes, FPS templates,
weapons, enemies, or HUD systems.

## Oracle Boundary

Gate 7 uses three evidence layers:

1. **Gameplay state**: `debug_state` proves logic still succeeds.
2. **Presentation state**: `visual_state` is the machine-checkable oracle for the
   rendered door state.
3. **VLM evidence**: answers structured visual questions from screenshot/context.

The verdict source is:

```text
debug_state + visual_state + trace
```

The VLM evidence source is:

```text
screenshot + debug_state + visual_state
```

VLM may support the report, but it must not be the only thing that detects the
bug.

The Gate 7 smoke should explicitly guard against confusing this bug with Gate 1:

```python
assert report["failure_type"] == "presentation_mismatch"
assert report["failure_type"] != "progression_softlock"
assert report["debug_state"]["extraction_reached"] is True
assert report["debug_state"]["progression_softlock"] is False
assert report["visual_state"]["door_visual_open"] is False
```

The trace/evidence should make the distinction obvious: `open_door` succeeded,
extraction was reached, and only the visible door state stayed closed.

## Suggested Report Shape

Gate 7 can write a report under `results/unity/`:

```json
{
  "status": "bug_reported",
  "failure_type": "presentation_mismatch",
  "goal": "Reach extraction and verify the opened door is visually open.",
  "debug_state": {
    "door_open": true,
    "extraction_reached": true,
    "progression_softlock": false
  },
  "visual_state": {
    "door_visual_open": false,
    "door_visual_color": "closed"
  },
  "screenshot_path": "results/unity/bridge_screenshot.png",
  "trace": ["..."],
  "visual_evidence": {
    "visual_evidence_role": "supporting_evidence_only"
  }
}
```

## Regression Boundary

Gate 7 should mirror Gate 6:

```text
presentation_mismatch report -> Regression Plan IR -> templated PlayMode test
                              -> bug build FAIL
                              -> fixed build PASS
```

The generated regression should assert both logic and presentation:

- `debugState.door_open == true`
- `debugState.extraction_reached == true`
- `debugState.progression_softlock == false`
- `visualState.door_visual_open == true`

## Scope Fence

Gate 7 does not add:

- Multi-agent orchestration.
- LangGraph coordinator.
- Coverage or mutation infrastructure.
- Real provider-backed VLM as a required smoke dependency.
- RAG.
- Free-form C# generation.
- Authored scenes or imported FPS templates.

Gate 8 may normalize shared schemas only after this second bug class is green.
Gate 9 may start the formal multi-agent orchestrator only after Gate 8.
