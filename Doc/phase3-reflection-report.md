# Phase 3 Reflection Report — Baseline vs Reflective Agent

> **Date**: 2026-06-06 (re-run under the diagnostic recovery ladder, ADR-0004;
> supersedes the 2026-06-03 table-based run)
> **Question**: Does 3-type reflection make the agent better under injected
> failures, and how (Doc/phase3-design.md §6)?
> **Answer**: Reflection's value is **diagnosis + detecting non-recoverable
> (logic) faults**, NOT recovering faults that brute-force retry already handles.
> On a persistent (logic-like) fault the reflective agent now reports a bug
> **5/5** (structural escalation on ladder exhaustion) while the baseline
> silently times out 0/5.

## 1. Setup

- Baseline = `run_agent_loop` (Phase 2 while loop, no reflection).
- Proposed = `run_reflective_agent` (LangGraph reflection; diagnostic recovery
  ladder re_observe → retry → report, ADR-0004).
- Decider fixed (always fire) so reflection, not decision, is under test;
  reflector is the real DeepSeek model. 5 runs per condition, max_steps=6,
  **max_reobserves=1, max_retries=1** (the minimal, pre-declared ladder budget).
  `experiments/eval_reflection.py`.
- Injected failures (`experiments/inject.py`): one-shot perception (mask one
  ammo change), one-shot execution (skip one fire), persistent execution
  (never fires → approximates a logic bug; real logic bugs are Phase 4).

## 2. Results

| Condition | Baseline | Baseline steps | Proposed | Proposed steps | Recovery path |
|---|---|---|---|---|---|
| one-shot perception | success 5/5 | 2.0 | success 5/5 (recovered 5/5) | 1.0 | re_observe only |
| one-shot execution | success 5/5 | 2.0 | success 5/5 (recovered 5/5) | 2.0 | re_observe → retry |
| **persistent (logic-like)** | **0/5 (silent timeout)** | 6.0 | **bug_reported 5/5** | 2.0 | re_observe → retry → report |

## 3. What this honestly shows

1. **The recovery ladder disambiguates by OUTCOME, and the path matches the
   fault.** A perception fault recovers at rung 1 (re_observe — a fresh read sees
   the true ammo): 1 step, no second shot. An execution fault survives re_observe
   (re-reading still shows no change) and needs rung 2 (retry): 2 steps. We did
   NOT classify the two up front; the ladder distinguished them by what it took
   to recover (ADR-0004). The perception case is also slightly cheaper/cleaner
   than the baseline (1 step, no wasted shot, vs baseline's 2) — a modest
   appropriateness win. But the headline is still diagnosis, not a recovery-rate
   win on transient faults: the baseline brute-forces those to success too.

2. **For a non-recoverable (persistent / logic-like) fault, reflection is
   decisive — now 5/5.** The baseline cannot tell "stuck" from "the game is
   broken": it burns all 6 steps and reports nothing (bug MISSED, 0/5). The
   reflective agent exhausts the ladder (re_observe fails, retry fails) and
   escalates to SUSPECTED logic, reporting a bug 5/5 in 2.0 steps. This is a
   detection-rate proxy for Claim 1: reflection turns a silent timeout into a
   bug report.

3. **5/5 now, not 3/5 — and the reason is the honest part.** The earlier
   table-based run reported 3/5 because escalation to logic depended on the LLM
   occasionally guessing "logic" within the budget. The ladder removes that
   dependence: a bug is reported on STRUCTURAL evidence (re_observe failed AND
   retry failed), so persistent faults escalate deterministically. The report is
   framed as SUSPECTED logic under this calibration + budget (with
   `recommended_next`: widen the observation window / raise the budget / inspect
   adapter timing), not a verdict that the game is broken.

4. **perception vs execution is still NOT claimed as accurate classification —
   by design.** The LLM's diagnosis is recorded but does not steer routing; in
   this run (and the live smoke) it mislabelled the perception fault as
   "execution", yet the ladder still recovered it at re_observe. We report
   logic-vs-non-logic, not a perception/execution accuracy (ADR-0003; threats to
   validity).

## 4. Takeaway (the research point)

Reflection's worth in agent-based testing is **not** "recover everything" — a
cumulative retry already recovers transient faults. Its worth is **diagnosis and
detecting the faults that matter**: turning a non-recoverable failure the
baseline silently ignores into an explicit, evidence-backed bug report. The
diagnostic ladder makes this STRUCTURAL (no longer hostage to an LLM guess),
which is exactly what a testing system needs, and it sets up Phase 4 (real
mutation bugs + LLM oracle) where the detection-rate story is measured on genuine
logic bugs.

## 5. Reproduce

```powershell
python experiments/eval_reflection.py --runs 5 --max-reobserves 1 --max-retries 1
```

Outputs `experiments/reflection_results_<ts>.csv` (gitignored) + the summary.
The 1/1 budget is the minimal clean ladder; budget-robustness sweeps (2/1, 3/1,
2/2) are a separate experiment, not this baseline refresh.
