# LLM Adapter-Generation Prompt Template

> The cheapest form of "LLM-assisted onboarding" (see `Doc/reviews/2026-06-05-…`).
> Paste this to Claude/Codex with a game spec to get an adapter DRAFT. It is also
> the spec for a future `adapter-generator` skill. **The draft is not final**:
> smoke tests + a human must calibrate real engine timing/perception (the
> `FIRE_TICS=16` class of facts an LLM cannot infer).

## Prompt

```
You are generating a game adapter for an agent-based FPS testing framework.
The AGENT LAYER MUST NOT BE TOUCHED. You only produce the adapter + content.

Read these first (provided): Doc/adapter-contract.md, toy_fps/ (worked example),
actions/result.py (snapshot_result, decreased/increased/unchanged).

GAME SPEC (filled by the user):
  game: <name>
  state metrics: <e.g. ammo:int, health:int, score:int>   # become game_variables
  has screen: <yes/no>
  primitives (intent -> engine call): <e.g. fire_once -> game.fire()>
  test templates: <name -> (primitive, metric, expectation: decreased|increased|unchanged, by)>

GENERATE:
  <game>/game.py        # state machine OR engine wrapper exposing a state with
                        #   .game_variables (dict) and optional .screen / .done
  <game>/adapter.py     # Primitives (reset/observe + mechanic methods) and an
                        #   action library: composites returning snapshot_result,
                        #   plus DESCRIPTIONS, EXPECTATIONS (shared helpers),
                        #   run/list_templates/check_expectation, a screen-tolerant _read
  <game>/goals.feature  # Success over <metric>_before/<metric>_after/steps
  tests/test_<game>.py  # drive run_reflective_agent + run_agent_loop with a
                        #   scripted FakeDecider/FakeReflector (NO API)
  experiments/<game>_demo.py  # same agent, real decider/reflector

HARD RULES:
  - results ONLY via snapshot_result (never hand-build before_x / xStart / x_delta)
  - reuse GroundTruthPerceptor unless reading pixels (then a VLMPerceptor)
  - do not import or edit agent/ ; the agent layer is fixed
  - end by listing a SMOKE CHECKLIST of facts that MUST be measured, not guessed:
    fire-to-decrement tic count, reset stability, HUD-vs-state lag, screen crop, etc.
```

## After generation (mandatory calibration)

1. `python -m pytest tests/test_<game>.py` — contract/unit pass (no API).
2. Smoke on the real game; fix primitives/expectations from MEASURED timing.
3. Persist findings: a WORKLOG entry + (if cross-cutting) an ADR; update the
   game's section if the contract needed a backward-compatible extension.
