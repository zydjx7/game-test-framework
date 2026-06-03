# v2 Roadmap — Portability & Extensions (NOT June)

> **Status**: 2026-06-02 captured for direction. **None of this is June work.**
> June = finish Phase 3 (Stage D/E) + packaging. Everything here is July+ /
> project v2 / thesis future work. Recorded so we (a) don't forget, (b) have a
> clean "future work" narrative for the thesis and interviews, (c) keep current
> architecture decisions from blocking these later.

## Goal

Make it easy for OTHER FPS games to use this framework. Today's portability:
the agent layer (goal-level Gherkin, function-calling decide, reflection,
LangGraph orchestration, eval) is reusable as-is; only the adapter layer
(perception + action implementations) is rewritten per game. v2 raises that.

## Portability ladder (the future-work narrative)

```
1. Framework portable   (DONE) agent layer unchanged; swap the adapter layer
2. Interface standardised (MCP) adapter becomes a standard MCP server; swap game = swap server
3. Knowledge portable   (RAG-B) FPS test knowledge as a retrievable library; new game reuses "what to test"
4. Scale                (multi-agent) coordinator + parallel test agents + judge/reporter
```

## 1. MCP — standardise the game interface

Wrap "game interaction" as an **MCP server** (Model Context Protocol, a standard
protocol for agents to call external tools/data):
- tools: `fire()`, `turn(dir)`, `observe()`, `reset()`
- resources: `screen` (frame), `game_variables` (ground truth)

The agent becomes an MCP client and calls abstract tools instead of
`import VizDoomEnv`. **Swap game = write a new game MCP server (e.g. an
UnrealFPS server) exposing the same tools; the agent layer does not change.**
Honest note: the existing GameStatePerceptor / VizDoomEnv abstractions already
decouple the game; MCP upgrades that decoupling to a cross-process standard
protocol (value = standardisation + ecosystem + résumé keyword), it is not the
decoupling from scratch.

## 2. Multi-agent — orchestration with LangGraph

LangGraph (already used for the single reflective agent) also orchestrates
multiple agents as nodes/subgraphs:
- Coordinator agent → assigns a batch of goals to test agents
- Test agents (each = today's reflective graph) → test different mechanics in parallel
- Judge agent → aggregates, decides which are real bugs (Phase 4 LLM Oracle is naturally Judge + Reporter)
- Reporter agent → bug reports

Value: scale (industry-size test suites), Judge independent of Actor (more
trustworthy). Caveat: only when parallelism/separation has real value — avoid
multi-agent for its own sake.

## 3. RAG — two distinct uses

- **RAG-A: case-based reflection** (W4 stretch, simpler): store past FAILURE
  cases; retrieve similar ones during reflection. Cross-episode experience.
  ReflectionCase is already RAG-ready (phase3-design §4.5).
- **RAG-B: knowledge-driven test generation** (bigger, future): store FPS
  general TEST KNOWLEDGE (not failure cases) — e.g. `{mechanic: "ammo
  consumption", goal: "firing should decrease ammo", common bug: "not
  decremented / negative", how: "fire and check delta"}`. A new game's agent
  retrieves "what FPS games usually test" → auto-generates goals → adapts to the
  new game's actions/perception. This lifts portability from "framework
  reusable" to "test KNOWLEDGE reusable" — potentially a standalone thesis
  contribution. Difficulty: knowledge must be seeded/accumulated (cold start),
  and mapping generic knowledge to a specific game's variables still needs
  domain adaptation.

## June guardrail

Do NOT start any of this in June. The only thing to do NOW is keep the current
architecture from blocking it: keep the perception/action interfaces decoupled
(already true) and keep reflection cases RAG-ready (already true). That's it.
