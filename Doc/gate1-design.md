# Gate 1 Design - checkpoint-softlock fixture

> Scope: Gate 1 of `Doc/project-direction.md`. Gate 0 proved Unity CLI
> PlayMode + `debug_state.json`; Gate 1 proves a progression-softlock fixture can
> be judged by the same live-smoke discipline.

## Goal

Build the first multi-object Unity gameplay regression fixture:

```text
player -> keycard -> security door -> checkpoint -> death/respawn -> extraction
```

The fixed-script PlayMode test must pass in the normal configuration and fail
when the injected `door-not-persisted` bug is enabled.

## Hard PASS Criteria

All criteria are command-line verifiable with the Editor closed:

1. `python scripts/run_unity_tests.py` PASS in the normal configuration.
2. The fixed-script Gate 1 test reaches extraction after death/respawn.
3. `results/unity/debug_state.json` records checkpoint state, including
   `door_open: true`, `extraction_reached: true`, and
   `progression_softlock: false`.
4. Negative check: running the same PlayMode suite with
   `GATE1_BUG_DOOR_NOT_PERSISTED=1` must FAIL because the fixed script expects
   extraction to remain reachable.

## Fixture Model

Gate 1 remains intentionally small and deterministic. No FPS template, weapon,
enemy, HUD, bridge, agent, LLM, VLM, coverage, or mutation framework is added.

Runtime components:

- `PlayerState`: tracks the player's logical zone, keycard inventory, and respawn
  count.
- `KeycardPickup`: grants the keycard once and records world pickup state.
- `SecurityDoor`: wraps `DoorController`; opens only with a keycard and consumes
  the keycard.
- `CheckpointMarker`: captures the state that must persist across respawn.
- `DeathRespawn`: simulates death and restores checkpoint state.
- `ExtractionPoint`: succeeds only if the player can still progress through the
  persisted-open door.

The PlayMode test creates GameObjects programmatically. An authored scene can be
added later for human preview, but it is not the Gate 1 oracle.

## Injected Bug

The negative path is controlled by an environment variable:

```powershell
$env:GATE1_BUG_DOOR_NOT_PERSISTED='1'
python scripts\run_unity_tests.py
Remove-Item Env:\GATE1_BUG_DOOR_NOT_PERSISTED
```

When enabled, `DeathRespawn` deliberately restores the checkpoint without the
door-open state. The fixed-script test then observes the door closed after
respawn and fails before extraction, proving the gate can go red.

## Debug State

Gate 1 extends the debug export with a checkpoint snapshot:

```json
{
  "scene": "...",
  "door_open": true,
  "player_zone": "checkpoint_room",
  "has_keycard": false,
  "keycard_collected": true,
  "checkpoint_active": true,
  "respawn_count": 1,
  "extraction_reached": true,
  "progression_softlock": false,
  "failure_reason": ""
}
```

The JSON remains a machine oracle beside the PlayMode assertions. It is not a
chat or editor-only observation.

## MCP Policy

Funplay/Coplay MCP may help inspect the Editor or author simple objects, but
Gate 1 completion is still only the CLI PlayMode result. Do not commit MCP
package dependencies to `master` as part of this gate.

## Verification Checklist

1. Normal run: `python scripts/run_unity_tests.py` -> PASS.
2. Negative run with `GATE1_BUG_DOOR_NOT_PERSISTED=1` -> FAIL.
3. Restore environment and rerun normal -> PASS.
4. v1 baseline: `.venv\Scripts\python.exe -m pytest --basetemp
   .pytest_tmp\gate1-v1` -> 117 passed, 4 deselected.
