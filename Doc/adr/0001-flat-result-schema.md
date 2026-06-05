# ADR-0001: Flat `<metric>_before/<metric>_after` result schema

**Status**: accepted (2026-06-03, Phase 3.5)

## Context

The agent core had hardcoded `ammo_before`/`ammo_after`, so it only understood
ammo. A second game (ToyFPS) and new mechanics needed multiple metrics. A nested
shape `{before:{...}, after:{...}, metrics:{...}}` was considered.

## Decision

Composite results use a FLAT, per-metric naming: `<metric>_before` /
`<metric>_after`, built via the single constructor `actions.result.snapshot_result`.
`accumulate` folds them generically (first `<m>_before`, latest `<m>_after`,
plus `steps`). Goal success is a sandboxed `eval` over the flat cumulative dict.

## Consequences

- Existing ammo goals (`ammo_before - ammo_after >= 1`) changed ZERO characters.
- Any metric/direction is expressible (`score_after - score_before >= 1`).
- DO NOT hand-build result keys (`before_ammo`/`ammoStart`/`ammo_delta`) — always
  go through `snapshot_result`, which is the single point enforcing the naming.
- Expectation predicates use shared `decreased/increased/unchanged(metric)`.
