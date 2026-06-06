# ADR-0004: Recovery is a diagnostic ladder, not a classification-driven table

**Status**: accepted (2026-06-06, Stage-1 step 3). Implements the future work
named in [ADR-0003](0003-logic-vs-nonlogic-boundary.md).

## Context

ADR-0003 established that perception vs execution share the SAME observable at
the moment of an anomaly, so a single reflection cannot reliably tell them apart;
in v1 both used the same recovery (redo once), so the 3-type classification had
only two distinct consequences (redo vs report). ADR-0003 named "separating
re_observe vs retry so the types have real consequences" as deferred future work.

Naive completion would route by the LLM's label (perception→re_observe,
execution→retry, logic→report). But that re-introduces exactly the unreliable
distinction ADR-0003 warned against: the label is a guess made BEFORE any
recovery, on observationally-identical evidence.

## Decision

Make the three recoveries real, but drive them with a **diagnostic recovery
ladder**, not the LLM's perception/execution label:

1. **re_observe** (rung 1) — re-perceive the current state WITHOUT acting
   (`action_lib.observe(perceptor)`). Rebuilds the result via `snapshot_result`,
   keeping each metric's prior `<metric>_before` and refreshing `<metric>_after`.
   Does NOT advance `steps` and keeps the original `last_action` (it is a
   corrective re-read of the same step, not a new action). Fixes a perception /
   transient-observation fault — the effect happened, we misread it.
2. **retry** (rung 2) — re-run the action. An execution fault (the action did not
   go through) needs this; re_observe alone cannot fix it.
3. **report** (rung 3) — both non-logic recoveries failed → escalate to
   **suspected logic**, reported on STRUCTURAL evidence ("re_observe failed AND
   retry failed"), not on a single LLM guess.

The LLM's 3-type diagnosis is still produced and RECORDED (case library /
taxonomy, and a `diagnosed_logic_by_llm` flag in the report), but it does not
steer the control flow — not even a logic call short-circuits past an unused
recovery budget. Budgets `max_reobserves` / `max_retries` are parameters
(default 1/1 = minimal clean ladder), reset per-anomaly so a recovered anomaly
does not consume a later one's budget.

## Consequences

- The recovery OUTCOME becomes the diagnostic signal: a perception fault recovers
  at rung 1, an execution fault needs rung 2, a real bug survives both. Live
  smoke confirmed this AND that the LLM mislabelled a perception fault as
  "execution" while the ladder still recovered it at rung 1 — direct evidence
  that the ladder does not depend on classification accuracy.
- The honesty boundary of ADR-0003 still holds and is REINFORCED, not weakened:
  we still DO NOT claim accurate perception-vs-execution classification. The
  value of the split is "if the failure is one or the other, the matching
  recovery is more appropriate (re_observe wastes no action / does not perturb
  state)", surfaced empirically — not "we classify them correctly".
- A persistent (logic-like) fault is now REPORTED (suspected logic) on ladder
  exhaustion instead of silently timing out. This is a deliberate improvement,
  not a regression; it strengthens the "baseline silently misses, proposed
  detects" story. `Doc/phase3-reflection-report.md` figures (persistent
  proposed-reported rate) will rise toward 5/5 and must be re-run before they are
  re-cited.
- The report is framed as SUSPECTED logic under the current calibration + budget,
  with `recommended_next` (widen the observation window / raise the recovery
  budget / inspect adapter timing) — not a verdict that the game has a bug.
- New adapter-contract surface: the action library must expose
  `observe(perceptor) -> GameState` (the read half of a composite). Both
  reference adapters implement it as `return self._read(perceptor)`.
- Budget tuning must be a PRE-DECLARED experiment config, never tuned per case
  (clean-result discipline). 1/1 is the default; robustness sweeps (2/1, 3/1,
  2/2) are separate, recorded configs.
