# Adapter Contract — plug a new game into the framework

> **What this is**: the exact interfaces a new game must provide so the SAME
> agent layer runs on it unchanged. Distilled from the TWO existing
> implementations — ViZDoom (`env/`, `actions/`) and ToyFPS (`toy_fps/`).
> If your adapter satisfies these, `run_agent_loop` / `run_reflective_agent`,
> the goal parser, the decider, and reflection work on your game with NO edits.

## The three layers

```
agent layer    (DO NOT TOUCH)  goal parser + decider + reflection + graph/loop
adapter layer  (YOU IMPLEMENT) State + Primitives + Perceptor + Action library
content        (YOU WRITE)     goals.feature
```

| Adapter piece | ViZDoom impl | ToyFPS impl |
|---|---|---|
| State | `env.vizdoom_env.DoomState` | `toy_fps.game.ToyState` |
| Primitives | `actions.ActionPrimitives` | `toy_fps.ToyPrimitives` |
| Perceptor | `GroundTruthPerceptor` (or VLM) | `GroundTruthPerceptor` (reused) |
| Action library | `actions.TestActions` | `toy_fps.ToyActions` |
| Goals | `agent/goals.feature` | `toy_fps/goals.feature` |

## 1. State (what your env emits)

The object your env returns from `reset()` / `step()` and that
`primitives.observe()` hands back. Required / optional attributes:

```python
state.game_variables  # dict[str, number]   REQUIRED, e.g. {"ammo": 10, "health": 50}
state.screen          # image array | None   OPTIONAL (ToyFPS has none)
state.done            # bool                  OPTIONAL (episode finished)
```

The metric NAMES in `game_variables` are the metrics you can test
(ammo/health/score/...). That is the only place game-specific metric names enter.

## 2. Primitives (layer 1 — drive the engine)

Stateful; holds the latest state. Required methods the framework calls:

```python
reset()   -> state      # start a new episode (the agent loop calls prim.reset())
observe() -> state      # return the LATEST state WITHOUT advancing the game
```

Plus your own mechanic methods that YOUR composites call, e.g. `fire_once()`,
`wait()`, `heal()`. These are not fixed by the framework — primitives and
composites are designed together inside one adapter.

## 3. Perceptor (layer 2 — state -> GameState)

Satisfy `perception.GameStatePerceptor`:

```python
perceive(screenshot, **kwargs) -> GameState   # screenshot may be None
```

Most games can REUSE `GroundTruthPerceptor`: it reads ammo/health/score from the
`game_variables` you pass and fills a `GameState`. Add a metric? add an
`Optional` field to `GameState` (backward-compatible) and let GT read it. Only
write a custom perceptor if you read from PIXELS (e.g. a `VLMPerceptor`).

## 4. Action library (layer 3 — the agent's tools)

The agent only ever touches these. Required surface:

```python
DESCRIPTIONS: dict[str, str]                 # template -> text shown to the LLM decider
prim                                         # your Primitives instance (loop calls prim.reset())
run(template_name, perceptor) -> result      # execute one template
check_expectation(template_name, result) -> anomaly | None
list_templates() -> list[str]                # the names in DESCRIPTIONS
```

Each composite (template) does: read -> act -> read, and returns a result via
`snapshot_result`:

```python
from actions.result import snapshot_result
def fire_and_check_ammo(self, perceptor):
    before = self._read(perceptor)
    self.prim.fire_once()
    after = self._read(perceptor)
    return snapshot_result({"ammo": before.ammo}, {"ammo": after.ammo})
```

**Result MUST use `snapshot_result`** — it is the single point that enforces the
canonical `<metric>_before` / `<metric>_after` flat naming. Never hand-build
keys like `before_ammo` / `ammoStart` / `ammo_delta`.

`EXPECTATIONS` maps each template to a `describe` string + a predicate built from
the shared helpers (so a violated expectation triggers reflection):

```python
from actions.result import decreased, increased, unchanged
EXPECTATIONS = {
  "fire_and_check_ammo":   {"describe": "...", "check": decreased("ammo", 1)},
  "fire_and_check_score":  {"describe": "...", "check": increased("score", 1)},
  "heal_and_check_health": {"describe": "...", "check": increased("health", 1)},
}
```

`_read` must tolerate a screen-less state (the Observation contract):

```python
state = self.prim.observe()
screen = getattr(state, "screen", None)
gvars = dict(getattr(state, "game_variables", {}) or {})
return perceptor.perceive(screen, game_variables=gvars)
```

## 5. Goals (content you write)

A `goals.feature` of Scenario blocks. `Success:` is a Python expression over the
cumulative dict, which the loop builds for you: per metric the FIRST
`<m>_before` and LATEST `<m>_after`, plus `steps`. Any direction is expressible:

```gherkin
Scenario: Firing consumes ammo
  Goal: Fire and confirm it consumes ammo.
  Available actions: fire_and_check_ammo, heal_and_check_health
  Success: ammo_before - ammo_after >= 1     # decrease
# Success: score_after - score_before >= 1   # increase
# Success: health_after - health_before >= 5 # increase
```

## 6. What you do NOT touch (the agent layer)

`agent/goal.py` (parser), `agent/loop.py` (`run_agent_loop`,
`FunctionCallingDecider`), `agent/graph.py` (`run_reflective_agent`),
`agent/reflection.py` — all metric-agnostic, all reused as-is.

## Checklist: add a new game (mirror `toy_fps/`)

1. A State with `game_variables` (dict); `screen` optional.
2. Primitives: `reset()`, `observe()`, + your mechanic methods.
3. Perceptor: reuse `GroundTruthPerceptor` (add an `Optional` GameState field if
   you have a new metric); custom only if reading pixels.
4. Action library: composites returning `snapshot_result`, plus `DESCRIPTIONS`,
   `EXPECTATIONS` (shared helpers), `run`, `check_expectation`, `list_templates`,
   and a `prim`.
5. `goals.feature` with `Success:` over `<metric>_before/<metric>_after/steps`.
6. Run it: `run_reflective_agent(goal, your_lib, perceptor, decider, reflector)`
   — the agent layer is unchanged. (ToyFPS is the worked example; see
   `tests/test_toy_fps.py`.)
