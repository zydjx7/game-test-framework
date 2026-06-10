# Unity MCP Evaluation Log

> Branch-only working log for `codex/unity-mcp-eval`. Do not treat this as a
> mainline decision until an evaluation result is promoted back to `master`.

## 2026-06-10 Preflight

Baseline:

- Mainline policy is recorded in `Doc/unity-mcp-usage.md`.
- Evaluation branch: `codex/unity-mcp-eval`.
- Pinned Unity editor: `E:\unity\2022.3.12f1\Editor\Unity.exe`.

Repository access:

- `git ls-remote https://github.com/CoplayDev/unity-mcp.git HEAD` succeeded:
  `c0908b88d6ec2d7152df2a8fc9c1590270856390`.
- `git ls-remote https://github.com/FunplayAI/funplay-unity-mcp.git HEAD`
  succeeded: `0b26bf84a847fb9526faa1f443cc5405fd5162b6`.

Local tool check:

- `uv`: not found.
- `uvx`: not found.
- `node`: found at `F:\nodejs\node.exe`.
- `npm`: found at `F:\nodejs\npm.ps1`.
- `python`: found at `C:\Program Files\Python310\python.exe`.

Initial interpretation:

- CoplayDev remains the mature candidate, but this machine needs `uv/uvx` setup
  before a fair trial.
- FunplayAI is the lower-friction first trial for this machine because it is a
  Unity-side package targeting Unity `2022.3+`.

Current blocker:

- `python scripts\run_unity_tests.py` cannot run while
  `F:\game-testing-main\unity\GameTestFixture` is open in Unity Editor. Unity
  reported `ProjectAlreadyOpenInAnotherInstance` and no `results.xml` was
  produced.

Next step:

1. Save and close the open Unity Editor instance for `GameTestFixture`.
2. Confirm baseline CLI smoke: `python scripts\run_unity_tests.py`.
3. Trial FunplayAI package on this branch only.
4. Run the CLI smoke after the package import.
5. Record package-lock/user-settings noise before deciding whether anything is
   worth promoting to `master`.

## 2026-06-10 Funplay Trial

Baseline after closing the Editor:

- `python scripts\run_unity_tests.py` PASS.
- Closing the Editor without saving did not affect Gate 0 source files or tests.
- Unity had auto-changed `PackageManagerSettings.asset` back to
  `https://packages.unity.com`; the branch was restored to the committed
  `https://packages.unity.cn` setting before the baseline smoke.

Package install attempt:

- Added Git dependency:
  `com.gamebooom.unity.mcp = https://github.com/FunplayAI/funplay-unity-mcp.git`.
- Unity resolved it to commit/hash `0b26bf84a847fb9526faa1f443cc5405fd5162b6`.
- First import failed to compile because Funplay's UI/input-interaction tools use
  `UnityEngine.UI` and `UnityEngine.EventSystems`, but this minimal Gate 0
  project did not include `com.unity.ugui`.

Fix needed for this fixture:

- Added `com.unity.ugui = 1.0.0`.
- Unity then compiled Funplay successfully.
- `python scripts\run_unity_tests.py` PASS after adding Funplay + `com.unity.ugui`.

Local-only server enablement:

- Created ignored file `unity/GameTestFixture/UserSettings/FunplayMcpSettings.json`
  with `enabled=true`, port `8765`, `toolExportProfile=core`, and safety checks
  enabled. This file is intentionally not committed because `UserSettings/` is
  gitignored.
- Starting non-batch Unity Editor auto-started Funplay MCP:
  `http://127.0.0.1:8765/`.

MCP protocol smoke:

- `initialize` returned server name `Funplay MCP Server - GameTestFixture` and
  version `0.4.3`.
- `tools/list` returned the expected `core` profile with 29 tools, including
  `execute_code`, `get_scene_info`, `get_compilation_errors`, `get_console_logs`,
  `get_hierarchy`, PlayMode controls, input simulation, and screenshots.
- `resources/list` returned project/scene/selection/error/console resources.
- `tools/call get_scene_info` reported the default unsaved scene with Main Camera
  and Directional Light.
- `tools/call get_compilation_errors` reported no compilation errors or warnings.
- `resources/read unity://project/context` reported Unity `2022.3.12f1`, package
  version `0.4.3`, Edit Mode, no compilation errors, and no recent console errors.
- `tools/call get_editor_state` reported Edit Mode, not compiling, not updating.

Current Codex integration note:

- Manual HTTP JSON-RPC works in this running session.
- The current Codex tool list did not dynamically reload Funplay tools after the
  server started. A future session should add or let Funplay write the Codex MCP
  config:

  ```toml
  [mcp_servers.funplay]
  url = "http://127.0.0.1:8765/"
  ```

Assessment:

- Funplay is viable as a local authoring helper on this machine.
- It is not zero-footprint for this minimal fixture: it adds `com.unity.inputsystem`,
  `com.unity.nuget.newtonsoft-json`, and requires explicit `com.unity.ugui`.
- Keep it on the evaluation branch for now. Do not promote it to `master` until
  we decide the extra dependencies are worth the authoring/debugging benefits.
