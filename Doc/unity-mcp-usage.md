# Unity MCP Usage Rules

> Scope: Unity-track development helper policy. This document explains how AI
> agents may use Unity Editor MCP tools without weakening the project's
> live-smoke discipline.

## Position

Unity MCP is an **authoring assistant**, not project infrastructure.

- MCP may help inspect or edit the Unity Editor state: hierarchy, components,
  scene objects, prefabs, Console logs, screenshots, and simple Editor actions.
- MCP must not become the Gameplay Agent runtime channel.
- MCP must not become the oracle for pass/fail.
- MCP output never replaces `scripts/run_unity_tests.py`, PlayMode tests,
  `debug_state.json`, screenshots, or traces.

The runtime bridge remains the project-owned interface from
`Doc/adapter-contract.md`: `reset / action / observe / debug_state / screenshot /
trace` against the live game. It appears at Gate 2, not earlier.

## Allowed MCP Uses

MCP is allowed for:

- Inspecting hierarchy, selected objects, components, assets, and Console logs.
- Creating or modifying simple scene objects and prefabs during fixture
  authoring.
- Attaching existing scripts and setting serialized fields.
- Running safe Editor queries, compile checks, and screenshots for human review.
- Helping a Unity learner understand why an object, serialized field, or
  component reference is in a given state.
- Running PlayMode from the Editor as a convenience, as long as the final gate is
  still the command-line smoke.

## Required After Every MCP-Assisted Unity Change

After an MCP-assisted change touches Unity project state or C# code:

1. Run `python scripts/run_unity_tests.py`.
2. Read the printed PASS/FAIL and, when needed, `results/unity/results.xml`.
3. Fix or revert failures before claiming the work is done.
4. If the change could affect the reused Python core, also run
   `.venv\Scripts\python.exe -m pytest`.
5. Record non-trivial MCP lessons in `WORKLOG.md` or an ADR; do not leave them
   only in chat.

## Forbidden Uses

Do not use MCP to:

- Replace the Gate 2 runtime bridge or the adapter contract.
- Decide pass/fail from "the Editor looks right" or "Console has no obvious
  errors".
- Add Gameplay Agent, bridge, VLM, coverage, mutation, or multi-agent
  orchestration before their gates.
- Import large templates or full FPS starter projects when the gate calls for a
  minimal fixture.
- Make broad file deletions, moves, or package changes without a focused reason
  and a live-smoke verification.
- Directly edit scene or prefab YAML when a Unity Editor operation, C# test, or
  small script can express the change more safely.
- Leave hidden local probe scripts, generated scenes, or MCP settings
  undocumented.

## Candidate Tools

Treat tool choice as replaceable. The project should not depend on one MCP
implementation unless an evaluation branch proves it useful and low-risk.

Current candidates to evaluate:

- **CoplayDev/unity-mcp** (`com.coplaydev.unity-mcp`, Unity `2021.3+`):
  mature ecosystem, broad AI-client support, but includes Python/uv setup and a
  wider tool surface.
  <https://github.com/CoplayDev/unity-mcp>
- **FunplayAI/funplay-unity-mcp** (`com.gamebooom.unity.mcp`, Unity `2022.3+`):
  Unity-side package with core/full tool profiles, logs, compile checks,
  screenshots, PlayMode control, and input simulation; newer and less proven.
  <https://github.com/FunplayAI/funplay-unity-mcp>

Verify package metadata from the upstream repos before installing; do not rely on
old chat summaries for versions or setup instructions.

## Evaluation Plan

Run MCP experiments on a separate branch, never as an unreviewed dependency on
`master`.

Evaluation branch name:

```text
codex/unity-mcp-eval
```

Experiment order:

1. Keep `master` green first: `python scripts/run_unity_tests.py`.
2. Try one MCP package at a time.
3. Record install friction, Unity version compatibility, package-lock changes,
   whether Codex can actually see tools, and whether logs/compile errors are
   reliable.
4. Test only authoring tasks that are already safe for the current gate:
   hierarchy/Console inspection, simple object/component edits, screenshot, and
   PlayMode convenience runs.
5. Re-run `python scripts/run_unity_tests.py` after every package trial.
6. If a package is useful, document the exact setup and risks before proposing a
   mainline commit.
7. If a package is not useful, remove it from the branch and record why.

Success criteria for accepting an MCP package into the normal workflow:

- It connects reliably on Windows with the pinned Unity editor.
- It improves authoring/debugging without weakening CLI live-smoke verification.
- It does not require committed local user settings or fragile generated files.
- It does not tempt agents to bypass PlayMode tests, `debug_state`, screenshots,
  traces, or the Gate order.

## Default Stance

Use MCP as a Unity learning aid and Editor authoring helper. Keep the project's
real guarantees in code, tests, debug state, screenshots, traces, and the
runtime bridge.
