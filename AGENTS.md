# Multi-Agent Collaboration Protocol

This repository is edited by both Claude Code and Codex. Keep the agents physically
separate, but keep their shared understanding synchronized.

## Orientation (a fresh agent / new chat starts here)

A new Codex session or Claude Code chat has **no memory of prior chats**. The
design and plan live in the repo, not in chat. Before doing anything, read these
in order, then run the Start-of-Task Checklist:

1. **This file (`AGENTS.md`)** — collaboration rules + decision-persistence.
2. **`WORKLOG.md`** — `## Current In Progress` / `## Next Task` / `## Deferred`.
3. **`Doc/research-plan.md`** — the authoritative 5-phase plan + invariants.
4. **`Doc/adr/README.md`** — key long-lived decisions (esp. ADR-0001 result
   schema, ADR-0003 failure-taxonomy boundary).
5. **Task-specific**: `Doc/adapter-contract.md` (adding a game),
   `Doc/phase{N}-design.md` (a phase), `Doc/v2-roadmap.md` (extensions),
   `Doc/reviews/` (prior cross-agent reviews).

`CLAUDE.md` (user + project) already routes a new Claude chat here; this list
makes "what to read" explicit so onboarding does not depend on guessing.

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

`Doc/research-plan.md` is the authoritative 5-Phase research plan for this
project. It defines the phase structure, module ownership, design invariants,
"do not do X" lists, and anticipated defense Q&A.

Rules:

- **Before any architectural decision** (new top-level directory, scope change,
  phase reorder, new module added, baseline-vs-new-line boundary change), read
  the relevant Phase section of `Doc/research-plan.md` first.
- If your proposed change conflicts with the plan, either:
  1. update `Doc/research-plan.md` in the **same commit** as the code change,
     using `shared:` prefix; or
  2. explain in the commit body's `Notes for future agents:` section why the
     deviation is justified.
- Do **not** silently diverge code from the plan. Either the plan is wrong
  (then update it) or the code is wrong (then fix it).
- The user maintains a personal Obsidian mirror at
  `F:\OBSIDIAN\Obsidian Vault\论文\扩展构想-ViZDoom版.md` for nicer reading;
  that file is not authoritative. AI agents must not edit it -- always edit
  `Doc/research-plan.md` instead.

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

