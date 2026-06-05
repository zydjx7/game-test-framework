# ADR-0003: Failure taxonomy's core boundary is logic vs non-logic

**Status**: accepted (2026-06-02, Phase 3)

## Context

Reflection classifies a step anomaly as PERCEPTION / EXECUTION / LOGIC. But a
perception failure and an execution failure produce the SAME observable (the
expected effect didn't happen), and reflection runs BEFORE recovery, so a single
anomaly cannot reliably distinguish them.

## Decision

The valuable, reliable boundary is **logic vs non-logic** (decides whether to
report a bug). Perception-vs-execution is NOT strongly distinguished in v1; both
share the same v1 recovery (redo once), so not separating them does not hurt the
recovery rate. Logic is evidenced by "already retried and still failing".

## Consequences

- DO NOT claim a high perception-vs-execution classification accuracy — that
  would be fake. Report recovery rate + logic-escalation accuracy instead
  (`Doc/phase3-reflection-report.md`).
- Separating re_observe vs retry recovery behaviour (so perception/execution have
  real consequences) is deferred future work, not a v1 claim.
- Stage E (no-reflection vs reflection) is an ABLATION, not the Claim 1 main
  comparison (LLM agent vs hardcoded BDD on mutation bugs); that is Phase 4.
