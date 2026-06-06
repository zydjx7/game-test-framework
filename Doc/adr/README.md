# Architecture Decision Records (ADR)

Short, append-only records of decisions that will be referenced repeatedly or
could be re-litigated by a future agent. A decision lands here when it is
cross-cutting and long-lived; otherwise the three-tier hierarchy in `AGENTS.md`
(commit body → WORKLOG → AGENTS) is enough.

Format per ADR: **Context** (why), **Decision** (what), **Consequences**
(what follows / what NOT to do). One file per decision, numbered, never edited
to reverse — supersede with a new ADR instead.

| ADR | Decision |
|---|---|
| [0001](0001-flat-result-schema.md) | Flat `<metric>_before/<metric>_after` result schema (not nested) |
| [0002](0002-foundation-before-mcp.md) | Contract + ToyFPS foundation before MCP/RAG |
| [0003](0003-logic-vs-nonlogic-boundary.md) | Failure taxonomy's core boundary is logic vs non-logic |
| [0004](0004-diagnostic-recovery-ladder.md) | Recovery is a diagnostic ladder (re_observe→retry→report), not a classification-driven table |
