# Review: LLM-Assisted Adapter Generation & Multi-Agent Collaboration

> **Source**: a Codex report (authored 2026-06-05) reviewed by Claude.
> **Purpose**: land the proposals + an explicit verdict so the decision is a
> repo artifact, not chat. This is the first entry under `Doc/reviews/`.

## What Codex proposed

1. **LLM-assisted adapter generation** (§2-4, 8-9): given the adapter contract +
   `toy_fps/` example + a new game's spec, an LLM generates the adapter
   boilerplate (State/Primitives/Action library/DESCRIPTIONS/EXPECTATIONS/
   goals.feature/tests). Honest caveat: an LLM cannot infer real engine timing
   (e.g. ViZDoom `FIRE_TICS=16`); smoke tests + a human must calibrate. Codex
   suggested building an `adapter-generator` **skill**, and ranked it ABOVE MCP.
2. **Collaboration mechanism fixes** (§7): WORKLOG has a duplicate `## Next
   Task`; important decisions are scattered → add `Doc/adr/`; cross-agent
   reviews should live in `Doc/reviews/`; optionally a machine-readable
   `.tasks/current.yaml`.
3. **MCP / skills** (§8): a project MCP server exposing worklog/docs/git +
   validate_adapter; reusable skills. Codex itself said "don't do MCP yet".

## Claude's verdict

| Proposal | Verdict | Action |
|---|---|---|
| WORKLOG duplicate `## Next Task` (§7.1) | **Adopt** | removed the stale block; added a `## Deferred / Later` section |
| `Doc/reviews/` for cross-agent reviews (§7.3) | **Adopt** | this file; rule added to AGENTS.md |
| `Doc/adr/` for long-term decisions (§7.2) | **Adopt** | created `Doc/adr/` + index + ADR-0001/0002/0003 |
| LLM-assisted adapter generation (§2-4) | **Adopt (v2)** | direction recorded in v2-roadmap; honest caveat kept |
| `adapter-generator` skill first (§9) | **Improve** | do the cheapest thing first: an **adapter-generation prompt template** (`Doc/adapter-generation-prompt.md`) usable today; a skill is a later packaging of it |
| `.tasks/current.yaml` task queue (§7.4) | **Decline (now)** | over-engineering for 2 agents; WORKLOG suffices. Codex agreed it's "not必须" |
| MCP server for project state (§8.1) | **Decline (now)** | Codex agrees; stays in v2-roadmap until the contract is stable + 2-3 adapters exist |

## Why "improve" on the skill ordering

A skill is infrastructure (build + maintain). The actual need is "lower the cost
of onboarding a new game **now**". A prompt template delivers that immediately
(paste the contract + ToyFPS example + a game spec → get an adapter draft), and
it doubles as the spec for a future skill. So: **template now, skill later**,
mirroring the project's own "ToyFPS before MCP" lesson — prove the flow before
protocol-ising/packaging it.

## Honest framing kept (do not overstate)

LLM-assisted generation produces a **draft + test skeleton**; smoke tests and a
human must calibrate action timing and perception reliability (the `FIRE_TICS=16`
class of facts). Generation lowers cost from "hand-write everything" to
"spec-driven draft + measured calibration"; it does not remove real-game
verification.
