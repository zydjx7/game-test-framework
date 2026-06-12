# GameTest Agent System — Project Direction (real-engine / Unity track)

> **Forward-authoritative.** This supersedes `Doc/research-plan.md` as the project's
> go-forward plan. `research-plan.md` (the ViZDoom track, Phases 0–3.5) is **DONE,
> 117 tests green, and retained** as the source of the reused Python agent core +
> the portability baseline — see "Reused assets". Do not delete it.
>
> Naming note: `research-plan.md` calls itself "v2.0" because it was the ViZDoom
> rewrite of the old AssaultCube plan. That is a *different* "v2" from this pivot.
> To avoid collision this document is named the **GameTest Agent System** direction,
> not "v2".

## Why the pivot (read first)

The ViZDoom track (Phases 0–3.5, all DONE) proved the **agent core**:
observe→decide→act→check loop, reflection + diagnostic recovery ladder, DeepSeek
function-calling decider, goal-level Gherkin, GroundTruth-vs-VLM perception, a clean
adapter contract, and a disciplined live-smoke habit.

But its **test target** — `ammo -= 1` / `health -= n` on toy ViZDoom scenarios — has
little industrial value: in a real engine those are three-line unit-test assertions.
An LLM+VLM agent only earns its keep where unit tests **cannot** reach: integration,
presentation, and progression-softlock bugs that appear only when real systems are
wired together and that need a **player-visible oracle**.

So we keep the agent core and move the target to a **real engine (Unity)**.

## North star (one sentence)

> In a real engine, generate layered tests from a requirement, run a real gameplay
> bug — especially the presentation / progression-softlock bugs unit tests miss —
> capture screenshot + trace + debug_state, and emit a developer-reproducible report.

## End-state architecture — what this is ultimately building

The Gates below are the **build order**, not the full picture. The intended end
state is a **dual-agent GameTest Agent System** that closes the loop from
requirement to permanent regression protection:

```text
Game requirement / GDD / bug report / PR diff
        |
        v
Spec-to-Test Agent
  reads specs / code / bug reports -> structured Test Plan IR ->
  template-renders tests. It does NOT play the game.
        |
        v
Layered tests
  +- Unit tests (NUnit, engine-side)
  +- Component / integration tests
  +- PlayMode / functional tests (fixed-script scene tests)
  +- Gameplay Agent goals (goal-level Gherkin, NOT fixed scripts)
        |
        v
Test Runner / Unity runtime
  compile, run, pass/fail, debug_state.json, screenshots, trace/replay,
  logs. Agents call these tools; they never invent pass/fail.
        |
        v
Gameplay QA Agent (reuses the v1 agent core)
  executes goal-level scenarios via the runtime bridge;
  observes debug_state + screenshots; VLM = visual evidence only;
  classifies failures (incl. progression_softlock); emits the bug report.
        |
        v
Bug report (repro steps + debug_state + screenshots + trace)
        |
        v
Bug-to-Regression loop
  the report feeds BACK into the Spec-to-Test Agent, which renders a
  permanent unit/PlayMode regression test:
  it must FAIL on the bug build and PASS on the fixed build.
```

**Why layered (the system's core selling point):** which layer fails localizes
the bug. Unit green + PlayMode red → scene wiring/config problem; all scripted
layers green + Gameplay Agent red → a cross-system or player-visible issue that
only goal-level play exposes. The layers are complementary, not redundant.

Module responsibilities:

- **Spec-to-Test Agent** — requirements in; Test Plan IR + template-rendered
  tests out. Never executes gameplay.
- **Test Runner / evaluation tools** — the only source of pass/fail truth.
- **Gameplay QA Agent** — goal-level execution + failure attribution + report.
  The RESEARCH core (failure attribution / false-positive suppression — the
  thesis) lives HERE; Spec-to-Test is system breadth, not the thesis.
- **VLM visual inspection** — player-visible evidence for presentation /
  progression bugs. Never the sole oracle (hard rule 4).
- **Bug-to-Regression** — turns one discovered bug into a permanent guard.

What this system is **NOT** (scope fence — cite this against drift):

- Not a replacement for unit tests (it complements them at layers they miss).
- Not a pure black-box game-playing bot, and not a VLM-only QA system.
- Not full game QA: performance, network, localization, compliance, balance,
  and subjective playtest are OUT of scope unless explicitly added later.

The Gates are deliberately narrower than this end state. Do not pre-build later
modules (hard rule 8) — but never mistake Gate 0–2 plumbing for the project.

## Cardinal rule: live-smoke first (the most important line in this doc)

The ViZDoom track succeeded because every change was verifiable: edit → `python -m
pytest` → run demo → read trace/state → **KNOW whether the AI's code actually works**.
Unity threatens that loop — failures hide in editor / scene / prefab state the AI
can't fully see and a learner can't debug. Therefore:

> **Nothing lands in Unity without a machine-checkable PASS.** Unity's `pytest`
> equivalent is the command-line PlayMode runner:
>
> ```
> Unity -runTests -batchmode -projectPath <proj> -testPlatform PlayMode -testResults results.xml
> ```
>
> If a change cannot be confirmed by a PlayMode test or a smoke script that writes
> PASS/FAIL, it is **not done**.

## Reused assets (v1 → keep GREEN, do not rewrite)

v2 reuses these; it does not touch them. They stay green as the portability proof:

- `agent/` — loop, goal (goal-level Gherkin), reflection, graph (LangGraph),
  diagnostic recovery ladder.
- `perception/` — GroundTruthPerceptor, VLMPerceptor (Qwen3-VL-Flash backend).
- `actions/result.py` — flat `<metric>_before/_after` schema + accumulate.
- `Doc/adapter-contract.md` — the runtime-bridge spec a new game must implement.
- ViZDoom + ToyFPS adapters + their tests.

## New in v2 (build order = the Gates)

- A **Unity runtime adapter** (Python ↔ *running* game) implementing the adapter
  contract.
- A **Unity test-fixture sandbox** at repo `unity/GameTestFixture/` (path pinned;
  named "fixture", not "MiniFPS", so no agent mistakes Gate 0 for FPS-building) — a
  QA fixture that grows into a MiniFPS across the Gates, NOT a fun game. Gate 0 is
  an empty skeleton + one door (design: `Doc/gate0-design.md`).
- **Presentation / progression** bug classes + **VLM as visual evidence** (not sole
  oracle).
- (Gate 4) **Spec-to-Test Agent**; (Gate 6) the **bug→regression** loop that
  closes the system.

## Runtime bridge ≠ editor MCP (do not conflate)

Editor automation / MCP servers help **author** the game (create GameObjects, attach
components, edit scenes). They are NOT the channel the *running* agent uses to act and
observe. The Gameplay Agent needs a **runtime bridge**: reset / action / observe /
debug_state / screenshot / trace against the LIVE game = `Doc/adapter-contract.md`
implemented in C# + a thin RPC.

- **Gates 0–1 are pure C# PlayMode tests — NO bridge** (they run inside Unity).
- **The bridge appears at Gate 2.** Don't build it earlier; don't equate it with MCP.
- Operational rules for MCP-assisted authoring live in `Doc/unity-mcp-usage.md`.

## Roadmap — Gates (do NOT start Gate N+1 before Gate N passes)

| Gate | Deliverable | PASS criterion |
|---|---|---|
| **0** | Unity **test-fixture skeleton** (empty/minimal 3D project — NOT a game; no player/weapon/camera/HUD/enemy) + ONE trivial mechanic (door open/close) + a command-line PlayMode test + exported `debug_state.json`; screenshot best-effort | **Hard gate:** CLI PlayMode run returns PASS/FAIL **and** door-state asserted in the test **and** `debug_state.json` exported & checked — all without clicking the Editor. Screenshot is a best-effort artifact (see notes); do not let it block "pipeline verified". |
| **1** | Checkpoint-softlock fixture (player / keycard / door / checkpoint / death-respawn / objective / extraction) + a fixed-script PlayMode test | Normal build PASS; with injected `door-not-persisted` bug, FAIL. |
| **2** | Python runtime bridge (`reset/action/observe/debug_state/screenshot/trace`) + `scripts/unity_smoke.py` | A **no-LLM** Python script drives the full checkpoint flow end-to-end and asserts state. (= adapter-contract realised in Unity.) |
| **3** | Gameplay Agent (reuse v1 core) on a `gameplay_checkpoint_no_softlock` goal | Normal → agent reaches extraction; bug build → agent reports `progression_softlock` with screenshot + trace + debug_state. |
| **4** | Spec-to-Test Agent: requirement → Test Plan IR → **templated** unit/PlayMode tests | From the checkpoint requirement it emits a layered plan and template-renders at least a **compilable** unit + PlayMode test. |
| **5** | VLM as visual evidence | VLM answers structured questions (marker behind closed door? player view blocked?) appended as evidence — never the sole verdict. |
| **6** | Bug-to-regression loop | Given the Gate 3 bug report, the Spec-to-Test Agent renders ≥1 **compilable** unit/component/PlayMode regression test that **FAILS on the bug build and PASSES on the fixed build** (Gate 1's bug toggle provides both builds). |

**First valuable bug = checkpoint softlock (Gate 1+).** Gate 0's mechanic is
deliberately trivial: its only job is to prove the Unity→test→state pipeline is real
and command-line-controllable before any multi-system mechanic.

**Start from an EMPTY/minimal Unity 3D project, not an FPS template.** Gate 0 needs
only a `DoorController`, a `DebugStateExporter`, and a PlayMode test — no player,
weapon, camera, enemy, or HUD. Do NOT import the FPS Microgame wholesale (it drags in
systems you don't need and can't yet debug). A character controller is **cherry-picked**
in at Gate 1 (from Unity Starter Assets / the Microgame), never imported as a whole.

**Screenshot note.** Unity batchmode screenshots are flaky (no GameView; `-nographics`
can't render). Do NOT depend on `ScreenCapture`/GameView. The robust path is a fixed
Camera → RenderTexture → `Texture2D.EncodeToPNG`. Screenshot is best-effort at Gate 0
but becomes a **hard requirement by Gate 3** (the VLM gate needs real frames) — prove
it works no later than Gate 2.

## Post-Gate6 continuation

Gates 0-6 completed the first vertical slice. The next forward-authoritative
planning docs are:

- `Doc/post-gate6-roadmap.md` - why the project should add a second bug class
  before formal multi-agent orchestration, and the intended Gate 7-9 sequence.
- `Doc/gate7-design.md` - Gate 7 presentation/state-visual mismatch bug.

Post-Gate6 build order:

| Gate | Deliverable | PASS criterion |
|---|---|---|
| **7** | Second bug class: presentation/state-visual mismatch (`debug_state` logic succeeds, rendered door state is wrong) | Fixed build PASS; `GATE7_BUG_DOOR_VISUAL_STUCK_CLOSED=1` reports `presentation_mismatch` with `debug_state` + `visual_state` + screenshot + trace, then generated regression FAILS on bug build and PASSES on fixed build. |
| **8** | Artifact/schema/tool-contract normalization | Common schemas for run results, bug reports, visual evidence, Test Plan IR, Regression Plan IR, and CLI tool results; Gates 0-7 remain green. |
| **9** | Multi-agent orchestration MVP | LangGraph or equivalent orchestrates the existing tools and emits a final summary; CLI tools remain the only source of PASS/FAIL truth. |

Do not start Gate 9 before Gate 7 and Gate 8 are complete. Do not treat Gate 7's
bug toggle as a general mutation framework.

## Deferred — do NOT start before its gate

coverage / mutation / CI dashboard; Spec-to-Test free C# generation (templates first);
multi-agent orchestration before Gate 9; RAG; the full 7-class failure taxonomy (the v1 honest
boundary — logic vs non-logic — holds until richer mechanics force the split); MCP
server; Unreal. (Absorbs the old `Doc/v2-roadmap.md` extension ladder.)

## The three differentiators are PRESERVED (anti-forgery)

1. **Goal-level Gherkin** → the Gate 3 goal file.
2. **Failure reflection + diagnostic recovery** → reused ladder + a new
   `progression_softlock` class.
3. **Injected-bug evaluation + report oracle** → Gate 1 `door-not-persisted` bug as a
   mutation; Gate 3/5 bug report + VLM evidence as the oracle.

Any "redesign" proposal must keep these three.

## Hard rules for any AI coding agent (Claude Code / Codex)

1. **Never add a feature before the Unity live-smoke passes.**
2. **Never claim a Unity change works without a PlayMode test or smoke script that
   prints PASS/FAIL.** "It should work" is not done.
3. **No formal multi-agent orchestration before Gate 7 and Gate 8 are green.**
   Before Gate 9, use the existing scripts/tools directly.
4. **VLM is visual evidence beside `debug_state`, never the sole oracle.**
5. **No coverage / mutation infrastructure before Gate 8.** Gate toggles are
   controlled bug switches, not a general mutation framework.
6. **Keep the ViZDoom v1 project GREEN and untouched** — reused Python core +
   portability proof.
7. **The runtime bridge implements `Doc/adapter-contract.md`; editor MCP is authoring
   only, never the runtime oracle.**
8. **Do not pre-write specs/code for a later Gate** (e.g. bridge protocol detail at
   Gate 2, not now) — the same "don't pre-build" discipline that served v1.

## What success of the FIRST milestone looks like

Not "the whole dual-agent system." One thread, end to end:

```
Unity scene → Python controls it → Agent executes the checkpoint goal →
softlock bug detected → screenshot + debug_state + trace exported →
report explains the softlock
```

Punch Gates 0–3 through and the project stands up. Until then, more agent design is
just an architecture diagram.
