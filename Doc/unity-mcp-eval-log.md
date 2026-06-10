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
