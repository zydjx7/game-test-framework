# Post-Gate6 Roadmap

> Scope: the forward plan after Gates 0-6. Gate 6 completed the first vertical
> slice: Unity fixture -> Gameplay Agent bug report -> VLM supporting evidence ->
> generated regression test -> bug build red / fixed build green. This document
> decides what to do before building a formal multi-agent framework.

## Why this roadmap exists

Gate 0-6 proved feasibility for one bug family: checkpoint progression softlock.
That is a real milestone, but it is still one example. Starting orchestration,
RAG, coverage, mutation dashboards, or a broad multi-agent shell immediately
would risk abstracting from a single data point.

The next work should answer one question:

```text
Does the loop generalize beyond one progression bug?
```

Only after that should the project invest in a formal multi-agent framework.

## Current baseline

Already complete:

- Unity CLI test pipeline with PASS/FAIL.
- Runtime bridge: reset/action/observe/debug_state/screenshot/trace.
- Gameplay Agent over the bridge.
- Checkpoint progression-softlock bug report.
- VLM visual evidence as supporting evidence only.
- Spec-to-Test template path.
- Bug-to-regression red/green path.
- v1 Python portability baseline remains green.

This is the first complete vertical slice, not the final system.

## Recommended next gates

| Gate | Theme | Purpose | Start condition | Done when |
|---|---|---|---|---|
| **7** | Second bug class: presentation/state-visual mismatch | Prove the loop handles a player-visible bug, not only progression logic | Gate 6 green | Bug toggle -> report -> visual evidence -> generated regression; bug build FAILS, fixed build PASSES |
| **8** | Artifact and tool contract normalization | Stabilize the shared language between future agents | Gate 7 green | Common schemas for run results, bug reports, evidence bundles, test plans, and regression plans; existing gates still green |
| **9** | Multi-agent orchestration MVP | Introduce the real framework only after two bug families and stable schemas | Gate 8 green | One orchestrated command calls existing tools and produces a final summary; PASS/FAIL still comes only from CLI tools |

## Gate 7 choice

Gate 7 should add a presentation/state-visual mismatch bug:

```text
Logic state: door is open and extraction is reachable.
Visual state: the door still appears closed.
```

Suggested bug toggle:

```text
GATE7_BUG_DOOR_VISUAL_STUCK_CLOSED=1
```

Why this bug:

- It is a different class from `progression_softlock`.
- It is player-visible and therefore justifies VLM evidence.
- It can reuse the checkpoint flow and door/extraction fixture.
- It can remain machine-checkable by exporting a deterministic `visual_state`
  beside the screenshot, so VLM does not become the sole oracle.

Gate 7 should not depend on a real VLM API. The required smoke can keep using a
deterministic local evidence backend while preserving the same interface a real
VLM provider can implement later.

## When multi-agent starts

Do not start the formal multi-agent framework immediately after Gate 6.

Start it after:

1. Gate 7 proves a second bug class.
2. Gate 8 normalizes shared artifacts and tool contracts.

At that point LangGraph becomes useful because the system has a real state flow:

```text
requirement_or_bug
  -> test_plan_ir
  -> unity_run_result
  -> gameplay_report
  -> visual_evidence
  -> regression_plan_ir
  -> red_green_result
  -> final_summary
```

The orchestrator should coordinate existing tools. It must not replace them as
the source of truth.

## Gate 9 multi-agent MVP shape

Minimum agents/nodes:

- **Coordinator**: owns workflow state and routing.
- **Spec-to-Test Agent**: requirement -> Test Plan IR.
- **Gameplay QA Agent**: goal execution -> bug report.
- **Visual Evidence Agent**: screenshot + state -> supporting evidence.
- **Bug-to-Regression Agent**: bug report -> Regression Plan IR/test.
- **Unity Test Runner Tool**: only source of Unity PASS/FAIL.
- **Report Writer**: final human-readable summary.

Use LangGraph only for orchestration/state passing. Unity CLI and smoke scripts
remain the oracle.

Gate 9 must be a thin orchestrator:

```text
python scripts/orchestrate_gate9.py --scenario checkpoint_presentation
```

It should call existing tools that already passed Gates 0-8. It must not
introduce new bug detection logic, new Unity mechanics, or a new oracle. If a new
bug class is needed, that is a separate gate before orchestration.

## Gate 8 schema criteria

Gate 8 should make existing artifacts reliable and composable; it should not add
new gameplay behavior.

Minimum Done Criteria:

1. Existing Gate 3/5/6/7 artifacts validate against schemas.
2. Each schema has at least one golden fixture.
3. `scripts/validate_gate_artifacts.py` validates the latest local artifacts and
   prints PASS/FAIL.
4. Fields used by only one gate start optional unless a later consumer requires
   them.
5. Schema normalization does not change behavior; Gates 0-7 remain green.

Expected shape:

```text
schemas/
  run_result.schema.json
  bug_report.schema.json
  visual_evidence.schema.json
  test_plan_ir.schema.json
  regression_plan_ir.schema.json
  tool_result.schema.json

tests/fixtures/artifacts/
  gate3_progression_softlock_report.json
  gate5_visual_evidence.json
  gate6_regression_plan.json
  gate7_presentation_mismatch_report.json

scripts/validate_gate_artifacts.py
```

## Guardrails

- No coverage or mutation infrastructure before Gate 8. The Gate 1 and Gate 7
  toggles are controlled bug switches, not a general mutation framework.
- No multi-agent orchestration before Gate 7 and Gate 8 are complete.
- No free-form C# generation; keep templates until the template boundary becomes
  the bottleneck.
- No VLM-only verdict. Visual evidence is supporting evidence beside exported
  state and CLI results.
- No scene-heavy FPS template import. Continue extending the fixture with
  programmatic tests and narrowly scoped runtime components.
- No v1 core rewrites. The ViZDoom/ToyFPS baseline remains the portability proof.

## Deferred until after Gate 9

- Real provider-backed VLM demos and model comparisons.
- RAG for test generation or case-based reflection.
- CI dashboard and trend reporting.
- Broader mutation/evaluation matrix.
- Additional game engines or MCP-as-runtime experiments.

A lightweight Python-only CI can be considered after Gate 8 or during packaging:
run the v1 pytest baseline and schema validators. Unity PlayMode CI can remain
deferred until license/runner setup is worth the cost.

These are valuable, but they need the Gate 7-9 foundation first.
