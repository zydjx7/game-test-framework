# Phase 3 Reflection Report — Baseline vs Reflective Agent

> **Date**: 2026-06-03
> **Question**: Does 3-type reflection make the agent better under injected
> failures, and how (Doc/phase3-design.md §6)?
> **Answer**: Reflection's value is **diagnosis + detecting non-recoverable
> (logic) faults**, NOT recovering faults that brute-force retry already
> handles. On a persistent (logic-like) fault, the reflective agent reports a
> bug 3/5 of the time while the baseline silently times out 0/5.

## 1. Setup

- Baseline = `run_agent_loop` (Phase 2 while loop, no reflection).
- Proposed = `run_reflective_agent` (LangGraph 3-type reflection).
- Decider fixed (always fire) so reflection, not decision, is under test;
  reflector is the real DeepSeek model. 5 runs per condition, max_steps=6,
  max_recoveries=2. `experiments/eval_reflection.py`.
- Injected failures (`experiments/inject.py`): one-shot perception (mask one
  ammo change), one-shot execution (skip one fire), persistent execution
  (never fires → approximates a logic bug; real logic bugs are Phase 4).

## 2. Results

| Condition | Baseline success | Baseline steps | Proposed | Proposed steps |
|---|---|---|---|---|
| one-shot perception | 5/5 | 2.0 | 5/5 (recovered 5/5) | 2.0 |
| one-shot execution | 5/5 | 2.0 | 5/5 (recovered 5/5) | 2.0 |
| **persistent (logic-like)** | **0/5 (silent timeout)** | 6.0 | **bug_reported 3/5** | 3.6 |

## 3. What this honestly shows

1. **For recoverable (one-shot) faults, reflection adds no success/efficiency.**
   The goal is cumulative (ammo dropped from the first reading), so the baseline
   brute-forces success by simply firing again — both reach success in 2.0
   steps. Reflection still fires once and recovers (5/5 cases recovered), but its
   only *extra* output here is a diagnosis. Honest: do not claim a recovery-rate
   win where brute-force already wins.

2. **For a non-recoverable (persistent / logic-like) fault, reflection is
   decisive.** The baseline cannot tell "stuck" from "the game is broken" — it
   burns all 6 steps and reports nothing (bug MISSED, 0/5). The reflective agent
   escalates to LOGIC and reports a bug (DETECTED, 3/5) in fewer steps (3.6).
   This is a detection-rate proxy for Claim 1: reflection turns a silent timeout
   into a bug report.

3. **3/5, not 5/5 — and that is honest.** Escalation to LOGIC is the LLM's
   judgment within the recovery budget; in 2/5 runs it kept retrying and timed
   out instead of escalating. A larger max_recoveries or a sharper reflection
   prompt would likely raise this; reported as-is rather than tuned to look
   perfect.

4. **perception vs execution is not distinguished — by design.** In the live
   smoke the injected *perception* failure was classified *execution*, yet still
   recovered (same redo). The two are observationally identical (both → delta 0)
   and reflection runs before recovery, so we report only logic-vs-non-logic,
   not a perception/execution accuracy (threats to validity).

## 4. Takeaway (the research point)

Reflection's worth in agent-based testing is **not** "recover everything" — a
cumulative retry already recovers transient faults. Its worth is **diagnosis
and detecting the faults that matter**: turning a non-recoverable failure that
the baseline silently ignores into an explicit bug report. That is exactly what
a testing system is for, and it sets up Phase 4 (real mutation bugs + LLM
oracle) where this detection-rate story is measured on genuine logic bugs.

## 5. Reproduce

```powershell
python experiments/eval_reflection.py --runs 5
```

Outputs `experiments/reflection_results_<ts>.csv` (gitignored) + the summary.
