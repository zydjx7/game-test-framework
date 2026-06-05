# Reusability Roadmap — make the framework usable on OTHER FPS games

> **Status**: 2026-06-03. Reusability/portability is now a PRIMARY line of the
> project (not a "future-work footnote"), because the project's value claim is
> "a reusable agent-based FPS testing framework", and that is also what makes it
> a strong internship portfolio. The thesis-grade comparisons come last.
> **Sequence locked (2026-06-03)**: foundation (contract + ToyGame) BEFORE MCP.
> MCP standardises a contract that must first be clear; it is not step one.

## Why this is the main line, not a footnote

Today the AGENT layer (goal-level Gherkin, function-calling decide, reflection,
LangGraph orchestration, eval) is reusable as-is; only the ADAPTER layer
(env + perception + action) is per-game. That is "architecturally portable".
But it is not yet DEMONSTRATED portable, and the adapter contract is not yet
explicit. Raising this is the project's core maturity work.

## Reusability ladder (each rung enables the next)

```
① Adapter contract explicit (ABC + docs)   cheapest; you already have GameStatePerceptor
② Second reference impl (ToyGame adapter)   proves "swap game, agent unchanged" — the killer demo
③ MCP — protocol-ise the contract           swap game = implement an MCP server
④ RAG-B — portable TEST KNOWLEDGE            new game reuses "what to test" (the crown)
   (Phase 4 mutation + rigorous comparison is the thesis track, separate)
```

MCP is rung ③: it standardises a contract that ① must first make clear. RAG-B is
the crown (④) and needs ①②③ as foundation — knowledge-generated goals only land
if there is a clear adapter contract and a rich action library to map onto.

## Stage 1 (June) — Foundation: reusable contract + maturity + packaging

This is three-in-one: maturity AND demonstrated portability AND good packaging.

- **1a. Make the adapter contract explicit.** perception already has the
  `GameStatePerceptor` ABC. Add the missing contracts for the env and action
  layers (an `EnvAdapter`-style interface: `reset()/step()/screen()/game_vars()`,
  and the primitives/composites shape), and document "how to plug in a new game".
- **1b. Extend the action library by 2-3 mechanic dimensions** (health,
  ammo-bounds, death) — maturity, and richer failures for reflection. Verify
  first which game variables the scenario exposes (Phase 1/2 lesson).
- **1c. ToyGame second reference implementation** — a ~100-line pure-Python fake
  FPS (state machine: ammo/health) implementing the SAME contracts. Then
  `run_reflective_agent(goal, toy_lib, toy_perceptor, decider, reflector)` runs
  with the agent/loop, agent/graph, reflection layers UNCHANGED. This turns
  "architecturally portable" into "demonstrated portable", and doubles as a
  fast, ViZDoom-free test fixture. Writing it is also what forces ① to be clean.
- **1d. Packaging.** README (EN or bilingual; FIX the file encoding so GitHub
  renders it — Codex reports it currently shows as garbled), an architecture
  diagram, a "plug in a new game" section, a demo GIF, three run commands.

**Stage 1 deliverable**: explicit contract + TWO game implementations (ViZDoom +
ToyGame) + richer actions + a clean README = genuinely reusable AND presentable.

## Stage 2 (late June / July) — MCP: protocol-ise the interface

Expose the Stage-1 adapter contract as an **MCP server** (Model Context
Protocol; server exposes tools `fire()/observe()/reset()` + resources
`screen`/`game_variables`; the agent is an MCP client). Swap game = implement an
MCP server, no agent change. This is the protocol-isation of rung ①, the
cross-process/standardised upgrade. Résumé value: MCP is a current standard.

## LLM-assisted adapter generation (onboarding cost) — template now, skill later

Lower the cost of plugging in a new game: an LLM generates the adapter DRAFT
(state/primitives/action library/goals/tests) from a game spec + the contract +
ToyFPS. Start with a prompt template (`Doc/adapter-generation-prompt.md`, usable
today); an `adapter-generator` skill is a later packaging. Honest limit: an LLM
cannot infer real engine timing (e.g. `FIRE_TICS=16`) — smoke tests + a human
calibrate. See `Doc/reviews/2026-06-05-llm-adapter-generation-and-agent-collab.md`.

## Stage 3 (July-Aug) — RAG: knowledge that ports

- **RAG-A (smaller)**: case-based reflection — store past FAILURE cases, retrieve
  similar ones during reflection. ReflectionCase is already RAG-ready
  (phase3-design §4.5).
- **RAG-B (the crown)**: knowledge-driven test generation — store FPS general
  TEST KNOWLEDGE, e.g. `{mechanic: "ammo consumption", goal: "firing should
  decrease ammo", common bug: "not decremented / negative", how: "fire and check
  delta"}`. A new game's agent retrieves "what FPS games usually test" →
  auto-generates goals → adapts to its actions/perception. Lifts portability from
  "framework reusable" to "test KNOWLEDGE reusable" — a potential standalone
  thesis contribution. Needs the Stage-1 contract + action library to land.

## Multi-agent (later) — orchestration with LangGraph

LangGraph already runs the single reflective agent; it also orchestrates many:
Coordinator → parallel Test agents → Judge (Phase 4 LLM Oracle = Judge+Reporter)
→ Reporter. Value: scale, Judge independent of Actor. Only when parallelism has
real value — not for its own sake.

## Thesis track (last) — Phase 4 + rigorous comparison

Separate from the reusability line: Phase 4 real mutation bugs + the rigorous
comparisons (vs hardcoded BDD baseline, vs rule-based reflection baseline). These
need the mature action library + richer failure space to be convincing, so they
come AFTER Stage 1 maturity. Do them when the "experiment bench" is mature, not
on the fire/idle prototype.

## Guardrail

Stage 1 is the June focus. Don't jump to MCP/RAG before the contract is explicit
and ToyGame proves portability. Keep interfaces decoupled and reflection cases
RAG-ready (both already true) so nothing here is blocked.
