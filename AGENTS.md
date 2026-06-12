# Multi-Agent Collaboration Protocol

This repository is edited by both Claude Code and Codex. Keep the agents physically
separate, but keep their shared understanding synchronized.

## Orientation (a fresh agent / new chat starts here)

A new Codex session or Claude Code chat has **no memory of prior chats**. The
design and plan live in the repo, not in chat. Before doing anything, read these
in order, then run the Start-of-Task Checklist:

1. **This file (`AGENTS.md`)** — collaboration rules + decision-persistence.
2. **`WORKLOG.md`** — `## Current In Progress` / `## Next Task` / `## Deferred`.
3. **`Doc/project-direction.md`** — the authoritative GO-FORWARD direction
   (GameTest Agent System / Unity track): north star, Gates 0-6 plus
   post-Gate6 Gates 7-9, hard rules,
   reused-vs-new boundary. Read before any v2 decision. After Gate 6, also read
   `Doc/post-gate6-roadmap.md` and the current gate design doc.
4. **`Doc/research-plan.md`** — the ViZDoom-track master plan (Phases 0–3.5,
   **DONE**). Authoritative for the completed v1 design + invariants it records;
   **NOT** the forward direction (that is `project-direction.md`).
5. **`Doc/adr/README.md`** — key long-lived decisions (esp. ADR-0001 result
   schema, ADR-0003 failure-taxonomy boundary).
6. **Task-specific**: `Doc/adapter-contract.md` (adding a game / the runtime
   bridge spec), `Doc/phase{N}-design.md` (a v1 phase),
   `Doc/reviews/` (prior cross-agent reviews).

`CLAUDE.md` (user + project) already routes a new Claude chat here; this list
makes "what to read" explicit so onboarding does not depend on guessing.

## Current Direction & Hard Rules (GameTest Agent System / Unity track)

Forward-authoritative detail: `Doc/project-direction.md`. Summary every agent must
honour on this track:

**Cardinal rule — live-smoke first.** The ViZDoom track stayed honest because every
change was machine-verifiable (`python -m pytest` + a runnable demo). Unity hides
failures in editor/scene/prefab state. So: **no Unity change is "done" without a
command-line PlayMode test or a smoke script that prints PASS/FAIL.** Unity's
`pytest` equivalent:
`Unity -runTests -batchmode -projectPath <proj> -testPlatform PlayMode -testResults results.xml`.

**Build order — Gates (do not start Gate N+1 before Gate N passes).**
0 trivial-mechanic Unity PlayMode smoke (prove the pipeline) → 1 checkpoint-softlock
fixture + injected bug → 2 Python runtime bridge (no-LLM) → 3 Gameplay Agent reports
`progression_softlock` → 4 Spec-to-Test (Test Plan IR + templates) → 5 VLM visual
evidence → 6 bug→regression (generated regression test FAILS on the bug build,
PASSES on the fixed one).
After Gate 6, follow `Doc/post-gate6-roadmap.md`: Gate 7 second bug class ->
Gate 8 artifact/schema/tool contracts -> Gate 9 multi-agent orchestration MVP.

**End state (do not lose sight of it):** a dual-agent loop — Spec-to-Test
(requirement → Test Plan IR → layered tests) + Gameplay QA Agent (goal-level
execution + failure attribution + report) + bug-to-regression feedback. See
`Doc/project-direction.md` § End-state architecture. The Gates are the build order
toward that loop; Gate 0–2 plumbing is not the project itself.

**Hard rules (do not violate):**

1. Never add a feature before the Unity live-smoke passes.
2. Never claim a Unity change works without a PlayMode test / smoke PASS-FAIL.
3. No formal multi-agent orchestration before Gate 7 and Gate 8 are green.
   Before Gate 9, use the existing scripts/tools directly.
4. VLM is visual evidence beside `debug_state`, never the sole oracle.
5. No coverage / mutation infrastructure before Gate 8. Gate toggles are
   controlled bug switches, not a general mutation framework.
6. Keep the ViZDoom v1 project GREEN and untouched — reused Python core + the
   portability proof.
7. The runtime bridge implements `Doc/adapter-contract.md`; editor MCP / automation
   is authoring only, never the runtime oracle.
8. Do not pre-write specs/code for a later Gate (e.g. bridge protocol detail at
   Gate 2, not now) — the same "don't pre-build" discipline that served v1.

## Worktree Layout

- Claude Code works in `.claude/worktrees/<name>/`.
- Codex works in the main worktree at `F:\game-testing-main`.
- Do not edit another agent's active worktree or tool metadata.

## Start-of-Task Checklist

Run this before starting any new task:

```powershell
git fetch origin
git log --oneline HEAD..origin/master
git log --oneline origin/master..HEAD
Get-Content WORKLOG.md | Select-Object -Last 10
```

- If `origin/master` is ahead, rebase before starting work.
- If the current branch has unpushed commits, push or explicitly preserve them
  before starting unrelated work.
- If another agent recently touched a shared file, inspect that change before
  editing the same surface.

## End-of-Task Checklist

After finishing a task:

1. Run the relevant verification commands.
2. Append a concise entry to `WORKLOG.md`.
3. Commit the task.
4. Push immediately.

`WORKLOG.md` is a human-readable summary, while Git history remains the source of
truth for exact commit hashes. For an entry created in the same commit as the work,
the hash may be omitted; do not create a second commit only to backfill that hash.

## Persisting Decisions and Warnings

Important judgments and "do not do X" notes must not live only in chat reports to
the human user — the next agent will never see those chats. Use this three-tier
hierarchy, in order of how tightly the note binds to a specific change:

1. **Commit message body** — judgments tied to the code change itself. Add a
   `Notes for future agents:` paragraph at the end of the body:

   ```
   refactor: consolidate DeepSeek client configuration

   ... [main message body] ...

   Notes for future agents:
   - API_TYPE env var was retired here.
   - Do not reintroduce it unless multi-provider support is redesigned.
   ```

2. **`WORKLOG.md` multi-line entry** — alerts the next agent must see at task
   start (the Start-of-Task Checklist tails WORKLOG). Indent as a sub-bullet
   beneath the main entry:

   ```
   - [Codex] 2026-05-16 refactor: consolidate DeepSeek client -> `c943659`
     - API_TYPE is intentionally retired; do not restore it in docs.
   ```

3. **`AGENTS.md` (this file)** — long-term collaboration rules that span tasks.
   Edit only when the rule itself is being changed.

`HANDOFF.md` is intentionally **not** created. Revisit that idea only if work
starts spanning multiple days / branches and the above three tiers stop scaling.

Default routing: prefer commit body for code-tied decisions, prefer WORKLOG for
cross-task alerts, prefer AGENTS.md for stable rules. When in doubt, write in
the lowest tier that still binds far enough.

### Cross-agent reviews and ADRs

Two more shared surfaces, for things the three tiers above don't fit well:

- **`Doc/reviews/<date>-<topic>.md`** — when one agent reviews another's report
  or code. Land the proposals + an explicit **verdict** (adopt / improve /
  decline, each with a reason) so the review is a repo artifact, not chat. A
  review only "counts" once it is here. (When the human pastes one agent's chat
  report to the other, the receiving agent lands it here.)
- **`Doc/adr/NNNN-title.md`** — long-term, cross-cutting decisions that will be
  referenced repeatedly or could be re-litigated by a future agent (Context /
  Decision / Consequences; append-only, supersede rather than edit). Use an ADR
  only for that class; otherwise the three-tier hierarchy suffices. Index:
  `Doc/adr/README.md`.

## Architecture of Record

**Forward direction (2026-06-10 →): `Doc/project-direction.md`** is authoritative
for where the project is GOING (GameTest Agent System / Unity track): north star,
Gates 0-6 plus post-Gate6 Gates 7-9, hard rules, reused-vs-new boundary. Read it
before any v2 architectural decision. After Gate 6, also read
`Doc/post-gate6-roadmap.md`.

`Doc/research-plan.md` is the ViZDoom-track master plan (Phases 0–3.5, **DONE**). It
remains authoritative for the completed v1 design, module ownership, and invariants
it documents — but it is **not** the forward plan. It defines the v1 phase structure,
module ownership, design invariants, "do not do X" lists, and anticipated defense Q&A.

Rules:

- **For v2 / Unity-track decisions** (anything on the GameTest Agent System track —
  new module, Gate change, scope change, bridge/adapter design): read
  `Doc/project-direction.md` first. If the change modifies the forward plan, update
  `Doc/project-direction.md` (or add an ADR) in the **same commit**, `shared:` prefix.
  **Do NOT update `Doc/research-plan.md` for v2 changes** — it is v1 history.
- **For v1 / ViZDoom-maintenance decisions** (touching the reused Python core,
  adapters, or a v1 invariant): read `Doc/research-plan.md` + the relevant ADRs first.
  If the change conflicts with a v1 invariant, update `Doc/research-plan.md` in the
  same commit (`shared:`) or justify it in the commit body's `Notes for future agents:`.
- Do **not** silently diverge code from whichever plan governs it. Either the plan is
  wrong (update it) or the code is wrong (fix it).
- The user keeps personal Obsidian mirrors under `F:\OBSIDIAN\Obsidian Vault\论文\`
  for nicer reading; those are **not** authoritative and AI agents must not edit them —
  always edit the repo docs (`Doc/*.md`) instead.

## Shared Files

Treat these as shared coordination surfaces:

- `README.md`
- `.gitignore`
- `requirements.txt`
- `pytest.ini`
- `CLAUDE.md`
- `AGENTS.md`
- `WORKLOG.md`

Any commit that changes one of these files must use the prefix `shared:` in the
commit subject and should be pushed immediately after verification.

## Conflict Discipline

- Prefer small, scoped changes to shared files.
- If two agents need the same shared file, the second agent rebases first and
  makes a strictly incremental change.
- Do not reintroduce retired compatibility knobs or old architecture branches
  just because they appear in historical docs.
- When behavior and documentation disagree, inspect the implementation before
  changing docs.

