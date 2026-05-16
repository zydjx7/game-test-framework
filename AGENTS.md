# Multi-Agent Collaboration Protocol

This repository is edited by both Claude Code and Codex. Keep the agents physically
separate, but keep their shared understanding synchronized.

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

