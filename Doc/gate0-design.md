# Gate 0 Design — Unity verifiable pipeline (test-fixture skeleton)

> Scope: Gate 0 of `Doc/project-direction.md`. Written at gate start, per the v1
> per-phase design-doc convention. Later gates get their own design docs when they
> start (hard rule 8: no pre-writing).

## Goal

Prove the **Unity → PlayMode test → debug_state → CLI PASS/FAIL** pipeline is real
and command-line verifiable, on ONE trivial mechanic (a door), before any game
content exists.

## Hard PASS criteria (all three, from the command line, Editor closed)

1. `python scripts/run_unity_tests.py` runs the PlayMode suite via Unity CLI and
   prints per-test PASS/FAIL, exiting nonzero on failure.
2. The door test asserts: closed → `Open()` → open.
3. `results/unity/debug_state.json` is exported and the test asserts its content
   (`door_open: true`).

Screenshot is **best-effort** (fixed Camera → RenderTexture → `EncodeToPNG` if it
cooperates; defer without blocking). It becomes a hard requirement by Gate 3 —
prove it works no later than Gate 2.

## Non-goals (do NOT build at Gate 0)

- **No authored scene.** The PlayMode test constructs its GameObjects
  programmatically (`new GameObject` + `AddComponent<DoorController>`). This avoids
  the Build Settings registration trap and scene authoring entirely. The first
  authored scene arrives at Gate 1.
- No player / weapon / camera / HUD / enemy / checkpoint. No FPS-template import.
- No runtime bridge, no LLM, no VLM.

## Division of labor

**Human (one-time GUI steps agents cannot do):**

- Install Unity Hub + ONE LTS editor (Unity 6 LTS or 2022.3 LTS — pick one).
- Sign in via Hub once so the Personal license is activated (batchmode fails on a
  never-activated machine).
- Record the exact editor version + full `Unity.exe` path in WORKLOG.

**Agent (everything else):**

1. Create the EMPTY 3D project at `unity/GameTestFixture/` (via Hub or CLI
   `-createProject`). Verify `git status` shows only `Assets/ Packages/
   ProjectSettings/` (the `.gitignore` Unity block is already in place). Commit the
   skeleton — first Unity commit is skeleton only, no mechanics.
2. CLI sanity smoke BEFORE any C#:
   `Unity -batchmode -quit -projectPath <abs> -logFile -` must exit 0.
3. `DoorController.cs` (Runtime): `IsOpen` + `Open()/Close()`. State only; no
   visuals required.
4. `DebugStateExporter.cs`: writes `{scene, door_open, timestamp}` to
   `results/unity/debug_state.json` (repo-root `results/` is already gitignored).
5. Assemblies — default path: a Runtime asmdef (`GameTestFixture.Runtime`) + a
   PlayMode test asmdef referencing it. A test asmdef **cannot** reference the
   default `Assembly-CSharp` — that is *why* the Runtime asmdef is the default
   path, and the fix for reference errors is mechanical (game code into its own
   asmdef). **But asmdef must not become the Gate 0 blocker**: if references
   misbehave beyond one focused fix attempt, get the pipeline green with the
   simplest structure that compiles and tidy assemblies in a follow-up commit
   ("Gate 0.5"). The gate is CLI PASS/FAIL, not assembly elegance.
6. `DoorSmokeTest` (PlayMode, `[UnityTest]`): create GameObject → add
   DoorController → assert closed → `Open()` → assert open → export debug_state →
   assert file exists + content.
7. Iterate in the Editor Test Runner if convenient, but the GATE is the CLI run:
   `-runTests -batchmode -testPlatform PlayMode -testResults <abs>\results.xml`.
8. `scripts/run_unity_tests.py`: subprocess → Unity CLI → parse `results.xml`
   (NUnit3 format) → print each test + summary → exit nonzero on failure. **This
   script is the Unity live-smoke for ALL later gates.**
9. **Negative check (required):** intentionally break the assertion once, confirm
   the CLI run reports FAIL, then revert. A gate that cannot fail is not a gate.
10. WORKLOG entry + commit + push (small commits, per AGENTS.md protocol).

## Known traps (read before debugging in circles)

- **License**: batchmode on a never-activated machine errors out — activate via a
  one-time Hub sign-in.
- **Do NOT pass `-quit` together with `-runTests`** (it can kill the run before
  results are written); the test runner exits by itself.
- **The Editor must not have the project open** during a CLI run (process lock —
  the CLI invocation fails or hangs).
- Test asmdef cannot reference `Assembly-CSharp` → Runtime code needs its own
  asmdef (step 5 has the fallback policy).
- `-testResults` should be an **absolute** path; the output is NUnit3 XML.
- If a screenshot is attempted, do not pass `-nographics` (PlayMode needs a
  graphics context to render).
- All artifacts (`results.xml`, `debug_state.json`, `*.png`) go under gitignored
  `results/unity/` — never commit artifacts.
- **v1 stays green**: `python -m pytest` must still pass (117); do not touch
  `agent/ perception/ actions/ env/ toy_fps/ Code/ src/`.
