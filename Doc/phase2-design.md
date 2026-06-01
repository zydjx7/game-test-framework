# Phase 2 Design Doc — Action Executor + Goal-level Gherkin + Minimal Agent Loop

> **Status**: DRAFT 2026-06-01. Needs Stage 0 verification + user decisions
> (see §3, §10) before locking, exactly like Phase 1's basic→defend_the_center
> correction. Do NOT lock the scenario/action set until Stage 0 is run.
> **Inherits from Phase 1**: `VLMPerceptor` (concrete ammo 100%),
> `GroundTruthPerceptor`, `VizDoomEnv(render_hud=...)`, trajectory infra.

## 1. Goal (what Phase 2 proves)

Upgrade the system from **Observer-only → Observer + Actor**: given a
one-line test *goal*, the agent autonomously chooses which action template to
run and judges success — with NO pre-written per-step function. This is the
research novelty (goal-level testing), not "a better VLM".

Falsifiable v1 target: the agent, choosing among ≥2 meaningful actions, can
achieve ≥3 test goals on a ViZDoom scenario, driven by DeepSeek function
calling, with no hand-written Behave step functions.

## 2. Scope (M1 v1 boundary)

**Doing:**
- 3-layer action library: primitives + 3-5 composites + test templates
- `Goal` dataclass + a ~60-line Gherkin parser (goal-level, not step-level)
- Minimal agent loop: observe → decide → act → check (plain Python `while`)
- `decide` uses **DeepSeek native function calling** (the `tools` API), so
  "Tool Use / Function Calling" is genuinely on the résumé, not prompt-hacked
- Run 3 goals end-to-end

**NOT doing (later phases):**
- Reflection / retry on failure → Phase 3 (Phase 2 just records + reports)
- LangGraph → Phase 3 (when branching/recovery makes a `while` loop messy)
- Planning (plan whole action sequence up front) → keep it reactive for v1
- Multi-scenario / large eval → later

## 3. Stage 0 — verify the environment FIRST (Phase 1 lesson)

Before writing any action code, verify (like we verified ammo range + HUD in
Phase 1):

- Which scenario gives the agent a *meaningful choice* among actions?
  Measured button sets (2026-06-01):
  - `defend_the_center`: TURN_LEFT/RIGHT, ATTACK (3) — proven in Phase 1
  - `deadly_corridor`: + all MOVE directions (7)
  - `deathmatch`: + SELECT_WEAPON1..6, NEXT/PREV_WEAPON (20) — only one with
    weapon switching
- Doom has **no manual reload** (the pistol auto-feeds until AMMO2 hits 0), so
  the design-doc §4 "reload" goal from research-plan.md is NOT directly
  available. Replace it with a feasible goal (see §10).
- For each candidate composite, confirm the effect is observable via the
  existing perceptors (ammo via HUD, enemy via VLM semantic field).

**Output of Stage 0**: locked scenario + the concrete action set + the 3 goals.

## 4. Three-layer Action Library

```
Layer 1  primitives   : fire_once(), turn_left(tics), observe()
                        -> thin wrappers over VizDoomEnv.step()
Layer 2  composites   : fire_and_check_ammo(perceptor) -> {ammo_before, ammo_after, delta}
                        -> one action + a before/after observation
Layer 3  test templates: ⭐ what the AGENT chooses among (the "tools").
                         The LLM never sees raw buttons, only these templates.
```

The research value is Layer 3: the LLM operates on semantic test actions, not
key presses. Swapping games means rewriting the action library, not every
test — that portability is the paper's selling point.

Proposed file layout (per research-plan.md, created only when needed):
- `actions/primitives.py`
- `actions/composites.py`

## 5. Goal-level Gherkin

Paradigm shift (the novelty):

```gherkin
# OLD (undergrad, step-level): every line needs a Python step function
When the player fires the weapon
Then the ammo should decrease by 1

# NEW (master, goal-level): describe the GOAL; the agent decides HOW
Scenario: Verify firing consumes ammo
  Goal: Firing the weapon should decrease ammo.
  Available actions: fire_and_check_ammo, observe
  Success: ammo_before - ammo_after >= 1
```

`Goal` dataclass (per research-plan.md §4.2.3):

```python
@dataclass
class Goal:
    description: str                      # natural-language goal
    available_actions: list[str]          # composite/template names the agent may use
    success_criteria: Callable[[dict], bool]  # compiled from the "Success:" line
    metadata: dict

    def is_satisfied(self, result: dict) -> bool:
        return self.success_criteria(result)
```

A ~60-line parser turns a Gherkin Scenario block into a `Goal`. The
`Success:` line compiles to a lambda over the accumulated result dict. Parser
is deliberately small — the value is the `Goal` abstraction + how the agent
consumes it, not Gherkin syntax richness.

## 6. Agent Loop + Function Calling

```python
def run_agent_loop(goal, env, perceptor, llm, max_steps=20):
    history = []
    for step in range(max_steps):
        state = perceptor.perceive(env.screen(), vizdoom_state=env.state())  # OBSERVE

        # DECIDE: DeepSeek native function calling picks the next template.
        # tools = JSON schema of goal.available_actions; LLM returns tool_calls.
        action_name, args = llm_decide(goal, state, history, tools=available_tools(goal))

        result = run_template(action_name, args, env, perceptor)             # ACT
        history.append((state, action_name, args, result))

        if goal.is_satisfied(result):                                        # CHECK
            return {"status": "success", "steps": step + 1, "history": history}
    return {"status": "max_steps_exceeded", "history": history}
```

- **Perception** uses the VLM (Phase 1, prompt+parse — info extraction).
- **Decision** uses DeepSeek function calling (selecting an action — tool use).
- **No reflection**: a failed/maxed-out run is just reported. Phase 3's job.

Proposed files: `agent/goal.py`, `agent/loop.py`.

## 7. Success Criteria (Phase 2 done = all of)

- Given a one-line goal string, the agent selects among ≥2 actions and reaches
  success WITHOUT a hand-written Behave step function.
- 3 goals pass end-to-end (set fixed in Stage 0).
- `decide` provably uses DeepSeek `tools`/`tool_calls` (not prompt+regex).
- A short demo run is reproducible from a script.

## 8. Implementation stages (filled after Stage 0)

| Stage | Content |
|---|---|
| 0 | verify scenario + action set + lock the 3 goals |
| A | `actions/primitives.py` + `actions/composites.py` + tests |
| B | `agent/goal.py` (Goal + Gherkin parser) + tests |
| C | `agent/loop.py` + DeepSeek function-calling `decide` + tests |
| D | end-to-end demo on the 3 goals + short report |

## 9. Out of scope (confirm)

- ❌ Reflection / retry (Phase 3)
- ❌ LangGraph (Phase 3)
- ❌ Planning / multi-step look-ahead (v1 is reactive)
- ❌ Cross-episode memory (Phase 3)
- ❌ LLM oracle / mutation (Phase 4)

## 10. OPEN decisions (need user input before locking)

1. **Scenario**: (a) stay on `defend_the_center` (proven, simple, but agent
   choice = fire/turn/observe) vs (b) `deathmatch` (weapon switching enables a
   richer "switch_and_check" goal, but a much more complex environment).
2. **The 3 goals**: e.g. {fire consumes ammo; fire×N reduces ammo by ~N;
   turning changes enemy visibility} on defend_the_center — final set depends
   on decision 1.
3. **Decision LLM**: DeepSeek `deepseek-chat` via native function calling
   (recommended, already the project's LLM). Confirm.
