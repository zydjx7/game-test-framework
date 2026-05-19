# WORKLOG

> Collaboration rule: append one concise line for each pushed task.
> Preferred format: `[Agent] YYYY-MM-DD <type>: <summary> -> <commit hash if already known>`.
> If the entry is created in the same commit as the work, the hash may be omitted.
>
> For entries that carry a non-trivial warning or judgment the next agent must see
> (e.g. "this knob was intentionally removed, do not restore"), append indented
> sub-bullets under the main entry:
>
>     - [Agent] 2026-XX-XX shared: ... -> `hash`
>       - Warning / decision / "do not X" note here.
>
> See `AGENTS.md` § "Persisting Decisions and Warnings" for the full three-tier hierarchy.

## 2026-05-15 ~ 16

- [Claude] 2026-05-15 feat: Phase 0.1 perception module -> `989b1ec`
- [Claude] 2026-05-15 shared: README ViZDoom roadmap refresh -> `4b1a8c5`
- [Claude] 2026-05-16 shared: security cleanup for `.backup/` and `.specstory/` -> `7653144`
- [Claude] 2026-05-16 test: CVPerceptor smoke test -> `c0e092d`
- [Codex] 2026-05-16 chore: stop tracking generated artifacts -> `58eecee`
- [Codex] 2026-05-16 refactor: consolidate DeepSeek client -> `c943659`
- [Codex] 2026-05-16 test: isolate legacy RiverGame coverage -> `a07fc68`
- [Codex] 2026-05-16 shared: align README with `deepseek-v4-flash` -> `dd812bb`
- [Codex] 2026-05-16 shared: track `CLAUDE.md`, ignore `.claude/` metadata -> `e096d08`
- [Claude] 2026-05-16 docs: add DeepSeek legacy-model deprecation note -> `0cec04e`
- [Claude] 2026-05-16 test: add DeepSeek API connectivity smoke test -> `51b3596`
- [Codex] 2026-05-16 shared: add repo-level multi-agent protocol and worklog -> `81fe5d8`
- [Claude] 2026-05-16 shared: drop sensei-reporting clauses from project CLAUDE.md -> `bdc394c`
- [Claude] 2026-05-16 shared: codify three-tier decision-persistence hierarchy in AGENTS.md -> `b41671a`
  - When you want to "tell the user something" about another agent's work, ask first
    whether that note should live in commit body / WORKLOG / AGENTS.md instead.
  - User as message forwarder is an anti-pattern; persist decisions where the next
    agent will actually look.
- [Claude] 2026-05-16 shared: clarify architecture progression in project CLAUDE.md
  - "Phase 0 完成定义" no longer says "4 模块目录" (ambiguous: bdd-side vs perception-side).
  - Added explicit per-Phase directory table and 5-point progression principle.
  - Architectural rule: do NOT pre-create empty `actions/` / `agent/` / `oracle/`
    directories; each Phase only creates what it needs. The 6 concept directories
    (perception/env/actions/agent/oracle/experiments) are end-state, not Phase 0.
  - The authoritative 5-Phase plan lives at `F:\OBSIDIAN\Obsidian Vault\论文\扩展构想-ViZDoom版.md`
    (next session will sync it into `Doc/research-plan.md`). 10 inconsistencies were
    fixed in that file in this same chat turn, including: deletion of Phase 0.4
    (folded into Phase 1.3), marking Phase 1.1 / 1.5 as done, removing sensei-reporting
    clauses, removing internship section (TBD), deleting Week-1 section, replacing
    GPT-4o-mini with TBD-Western-baseline placeholder.

## Current In Progress

- None. Phase 0.1 closure, safety cleanup, and repository coordination setup are complete.

## Next Task

- Phase 0.2: move ViZDoom sandbox work into the future `env/` source directory.

