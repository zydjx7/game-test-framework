# Phase 3.5 — Generalize the result/cumulative/observation schema

> **Status**: DESIGN 2026-06-03. A core refactor to remove the agent layer's
> hidden ammo-only assumption BEFORE ToyGame / new mechanics (Doc/v2-roadmap.md
> Stage 1 step 0). Done under the green ammo test suite (104 passing) — old ammo
> behaviour must stay equivalent.
> **Why first**: ToyGame and a 2nd ViZDoom mechanic both hit the same wall — the
> core only understands ammo. Clearing it once, test-guarded, is cleaner than
> doing it tangled inside ToyGame.

## 1. Where the ammo assumption hides (precise)

| File | What | Problem |
|---|---|---|
| `actions/composites.py` `_delta_result` | returns `{ammo_before, ammo_after, delta}` | ammo-only result shape |
| `actions/composites.py` `EXPECTATIONS` | predicate reads `r.get("delta")` | assumes a single `delta` |
| `actions/composites.py` `_read` | `perceptor.perceive(state.screen, game_variables=...)` | assumes state HAS `.screen` (ToyGame may not) |
| `agent/graph.py` `_apply_result` | hardcodes `result.get("ammo_before")` / `ammo_after` | cumulative only tracks ammo |
| `agent/loop.py` `run_agent_loop` | same hardcoded ammo cumulative | same |
| `agent/reflection.py` `_render_prompt` | renders "ammo delta" | prompt assumes ammo |
| `agent/loop.py` prompts | "Doom-like FPS" + "ammo delta" history | decider assumes ammo/Doom |
| `agent/goals.feature` | `Success: ammo_before - ammo_after >= 1` | (kept — see §3) |

## 2. Target schema

**Design choice vs Codex's nested `{before, after, metrics}`**: use a FLAT,
per-metric naming (`<metric>_before` / `<metric>_after`) instead of nesting.
Reasons: (a) goal success is a sandboxed `eval` over a flat dict, so the
referenced names must be top-level keys; (b) the existing ammo goal expressions
then change ZERO characters. Nesting would force every goal/test to be rewritten.

### 2.1 ActionResult — what a composite returns (flat, per-metric)

```python
# fire_and_check_ammo
{"ammo_before": 26, "ammo_after": 25}
# wait_and_check_health
{"health_before": 100, "health_after": 85}
# a composite may report several metrics:
{"ammo_before": 26, "ammo_after": 25, "score_before": 0, "score_after": 1}
```

The `delta` field is removed; expectations and goals compute from before/after.
A composite reports only the metrics it observes.

### 2.2 cumulative — fed to `goal.is_satisfied` (flat, accumulated)

Generic accumulation over steps, no hardcoded metric:
- for every `<m>_before` seen, keep the FIRST value (across the episode);
- for every `<m>_after`, keep the LATEST value;
- plus `steps` and `last_action`.

```python
{
  "ammo_before": 26,   # first observed this episode
  "ammo_after": 25,    # latest observed
  "health_before": 100, "health_after": 85,
  "steps": 2,
  "last_action": "fire_and_check_ammo",
}
```

Algorithm (replaces the ammo-hardcoded block in graph/loop):
```python
for key, val in result.items():
    if key.endswith("_before"):
        cumulative.setdefault(key, val)        # FIRST wins
    elif key.endswith("_after"):
        cumulative[key] = val                  # LATEST wins
cumulative["steps"] = step
cumulative["last_action"] = action
```

### 2.3 Goal success — expressions stay free-form, ammo ones UNCHANGED

```text
Success: ammo_before - ammo_after >= 1          # existing, 0 changes
Success: ammo_before - ammo_after == 0 and steps >= 1
Success: health_before - health_after >= 10     # new (loss)
Success: score_after - score_before >= 1        # new (gain — author writes direction)
```
No preset metric direction; the goal author writes the arithmetic. before/after
are the raw snapshots, so any direction is expressible.

### 2.4 Observation contract — screen OPTIONAL

A game state must expose `game_variables` (a dict); `screen` is optional (a
ground-truth/ToyGame state has none). `_read` becomes tolerant:
```python
screen = getattr(state, "screen", None)
gvars = dict(getattr(state, "game_variables", {}) or {})
return perceptor.perceive(screen, game_variables=gvars)
```
`GroundTruthPerceptor.perceive` already accepts `screenshot=None`, so a
screen-less ToyGame works. (VLM perceptors still require a real screen — not
used by ToyGame.)

### 2.5 EXPECTATIONS — predicate from before/after

```python
"fire_and_check_ammo": {
  "describe": "firing should decrease ammo by at least 1",
  "check": lambda r: (r.get("ammo_before") or 0) - (r.get("ammo_after") or 0) >= 1,
},
"idle_and_check_ammo": {
  "describe": "idling should leave ammo unchanged",
  "check": lambda r: (r.get("ammo_before") or 0) - (r.get("ammo_after") or 0) == 0,
},
```

### 2.6 Prompts — de-ammo

- `reflection._render_prompt`: render a GENERIC result summary (action +
  the result dict's before/after pairs), not "ammo delta".
- `loop._render_user_prompt` + `SYSTEM_PROMPT`: generic result summary; soften
  "Doom-like FPS" to "an FPS game" so a non-Doom adapter (ToyGame) is not
  contradicted.

## 3. Migration strategy (test-guarded)

The 104-test suite is the safety net; ammo behaviour must stay equivalent.

- **goals.feature / goal tests**: UNCHANGED (ammo_before/ammo_after kept).
- **composites**: `_delta_result` → `_snapshot_result` (no `delta`); EXPECTATIONS
  use before/after; `_read` tolerant. Update the 6 expectation tests in
  `test_actions.py` (they asserted on `delta`).
- **graph/loop cumulative**: generic accumulation. `test_graph.py` /
  `test_agent_loop.py` ScriptedActionLib results already use ammo_before/after;
  drop the now-unused `delta` key from their fixtures.
- **reflection**: generic prompt; `test_reflection.py` unaffected (it asserts on
  classification, not prompt text — keep the history-in-prompt test generic).

## 4. Files to touch

```
actions/composites.py     _snapshot_result, EXPECTATIONS, _read
agent/graph.py            _apply_result -> generic cumulative
agent/loop.py             cumulative + prompts
agent/reflection.py       _render_prompt
tests/test_actions.py     expectation tests (before/after)
tests/test_graph.py       fixtures (drop delta)
tests/test_agent_loop.py  fixtures if needed
agent/goals.feature       UNCHANGED
```

## 5. Verification

- Full suite stays green (104), with the expectation/fixture edits above.
- Live re-smoke: the Phase 2 3-goal demo (`run_reflective_agent`) still 3/3 on
  ViZDoom — proves the generalization didn't change ammo behaviour.
- Then ToyGame (next step) consumes this schema with health/score and the agent
  layer runs unchanged — proves the generalization is real, not ammo-in-disguise.

## 6. Out of scope (do NOT bundle here)

- re_observe / retry separation, recovery_attempts per-anomaly → v2-roadmap
  Stage 1 step 3 (reflection semantics), after ToyGame + mechanics.
- rule-based baseline → thesis track (deferred).
- New ViZDoom mechanics → step 2 (after ToyGame validates the schema).
- This step is ONLY the schema generalization + de-ammo, nothing else.
