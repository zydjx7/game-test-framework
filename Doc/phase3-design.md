# Phase 3 Design Doc — Reflection (3-type failure classification + recovery)

> **Status**: DRAFT 2026-06-02. Scope locked to a June-deliverable v1.
> **Inherits from Phase 2**: agent loop, `TestActions`, `FunctionCallingDecider`,
> `GroundTruthPerceptor`/`VLMPerceptor`.
> **Authoritative parent**: research-plan.md §5 (full Phase 3). This doc is the
> simplified, time-boxed v1 for the June internship-portfolio deadline.

## 0. Why this is the highest-ROI next step (job-hunt context)

The two JD keywords the project is missing — **Reflection** and **Agent
evaluation (recovery rate / false-positive / 误判分析)** — are BOTH delivered
here. Reflection is also novelty point #2 of the thesis. So Phase 3 serves the
paper and the résumé at once.

## 1. Goal

When an action does not produce its expected effect, the agent should
**classify the failure into one of three types and attempt recovery**, instead
of blindly continuing. The 3-type classification is the thesis novelty and must
be kept; only the recovery strategies are simplified for v1.

```python
class FailureType(Enum):
    PERCEPTION = "perception"   # the perceptor misread the state (VLM/CV wrong)
    EXECUTION  = "execution"    # the action did not go through
    LOGIC      = "logic"        # the game itself is wrong (a real bug)
```

## 2. Scope (v1, June)

**Doing:**
- Per-action *expected effect* so the loop can detect an anomaly (a "failure").
- `agent/reflection.py`: an LLM reflection step that classifies the anomaly
  into PERCEPTION / EXECUTION / LOGIC and proposes a recovery.
- v1 recovery (one round, no learning):
  - PERCEPTION → re-observe (and, if VLM, re-query / majority vote) once
  - EXECUTION → retry the action once
  - LOGIC → stop, mark as a suspected bug, emit a report; do NOT keep retrying
- Failure-injection harness to create PERCEPTION and EXECUTION failures on
  demand (LOGIC failures come from Phase 4 mutations; for v1 a single hand-made
  logic stub is enough for one case study).
- `experiments/eval_reflection.py`: baseline (no reflection) vs proposed,
  reporting the metrics in §6.

- **LangGraph for the reflection control flow** (2026-06-02 decision, moved IN
  from stretch): Phase 3 introduces branching (check -> success/continue/anomaly;
  reflect -> perception/execution/logic), which is exactly what a StateGraph
  models cleanly. Implement the reflective agent as a LangGraph graph
  (`agent/graph.py`) whose NODES reuse existing code (FunctionCallingDecider,
  TestActions, reflection.py). Keep the Phase 2 `run_agent_loop` while-version
  UNCHANGED as the no-reflection baseline — the two versions are the natural
  baseline-vs-proposed pair for §6 eval. This is genuine LangGraph use (real
  branching), not a forced rewrite.

**NOT doing (later / stretch):**
- Cross-episode / long-term memory (research-plan says skip for M1).
- RAG / case-based reflection → W4 stretch (route-2 bonus).
- Real mutation-injected LOGIC bugs → Phase 4 (July).

## 2.5 Reflective agent as a LangGraph StateGraph

```
decide -> act -> check
                  |- success  -> END
                  |- continue -> decide
                  '- anomaly   -> reflect
                                    |- perception -> re_observe -> decide
                                    |- execution  -> retry      -> check
                                    '- logic      -> report     -> END
```

- Nodes: decide / act / reflect / re_observe / retry / report. Each is a plain
  Python function over a shared state dict (goal, history, cumulative,
  last_result, failure_type, status, steps).
- Conditional edges: after check (3-way) and after reflect (by failure_type).
- LangGraph orchestrates control flow ONLY; the DeepSeek calls stay in the
  reused decider/reflection code (LangGraph is used independently of LangChain).
- Risk control: Stage 0 verifies `pip install langgraph` + a 3-node toy graph
  runs and can be used without LangChain, BEFORE wiring business logic. Fallback
  is a while+if/else reflection loop (same behaviour, minus the LangGraph word).

## 3. Detecting a failure (key design point for implementation)

The Phase 2 loop only checks the *cumulative* goal. Reflection needs a
*per-step* expectation. Plan: each composite declares its expected effect, e.g.
`fire_and_check_ammo` expects `delta >= 1`. The loop compares the actual step
result to that expectation:

- expectation met → continue as today
- expectation violated → invoke reflection (classify + recover)

This is the cleanest way to generate reflection material without changing the
goal-level Gherkin. Exact mechanism (where the expectation lives — composite
attribute vs goal metadata) is the first implementation decision.

## 4. Reflection prompt (per research-plan §5.2, kept)

Input: recent history + the violated expectation (expected vs actual).
Output JSON: `{"failure_type": "...", "recovery_action": "...", "confidence": 0-1}`.
The LLM is asked to weigh perception vs execution vs logic and justify a logic
call with evidence (so it does not cry "bug" too easily — false positives are
the metric we care about).

Implementation note: reflection produces structured output via DeepSeek
function calling (a single `classify_failure` tool with `failure_type` /
`confidence` / `reasoning`), consistent with the decider. The recovery action
is NOT chosen by the LLM — it is looked up from a fixed table by failure type
(research-plan §5.3), keeping v1 deterministic.

## 4.5 Memory layers (short-term now, RAG-ready for W4)

The agent has two memory layers; only the first is built in v1.

| | short-term (working) | long-term (episodic) |
|---|---|---|
| What | the current-episode `history` list | a library of past reflection cases |
| Scope | one goal/episode (cleared on reset) | across all episodes |
| How accessed | passed directly into the reflection prompt | embed + vector retrieve (RAG) |
| Phase | v1 (already in run_agent_loop) | W4 stretch (case-based reflection) |

So today's `history` answers "what did I just do this episode"; it does NOT
remember past episodes. That cross-episode gap is exactly what the W4 RAG layer
fills: retrieve the top-k most similar past failures and add them to the
reflection prompt ("a similar anomaly last time was PERCEPTION, re-observe
recovered it").

**RAG-ready case structure (build it now, retrieve in W4).** Stage B's
reflection returns a structured `ReflectionCase`:

```python
{
  "anomaly": {...},          # from TestActions.check_expectation
  "failure_type": "perception|execution|logic",
  "recovery_action": "re_observe|retry|report",
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "recovered": None,         # filled after recovery runs (Stage C)
}
```

v1 just logs these cases (no vector store, no retrieval). W4 adds embedding +
a lightweight store (Chroma / FAISS / numpy cosine) + top-k retrieval into the
prompt — **adding retrieval, not re-architecting**, because the case shape is
already right. The same case log is reused three ways: RAG source, the Stage E
eval dataset, and thesis case studies.

## 5. Failure injection harness

To exercise reflection deterministically:
- **PERCEPTION**: wrap a perceptor so ammo is perturbed (±2) or the screen is
  noised/cropped → the perceived delta is wrong though the game is fine.
- **EXECUTION**: wrap primitives so with probability p (~30%) `fire_once`
  does NOT actually advance ATTACK → ammo does not drop though the agent "fired".
- **LOGIC**: deferred to Phase 4 mutations; one hand-made stub for a single
  case study is acceptable in v1.

Injection lives in test/experiment wrappers, NOT in production classes.

## 6. Evaluation — HONEST metrics (2026-06-02 revised)

**Key design realisation**: a PERCEPTION failure and an EXECUTION failure
produce the SAME observable (fire → delta=0). Reflection classifies BEFORE
recovery, so from a single anomaly it cannot reliably tell "misread" from
"didn't execute". Therefore we do NOT claim a high perception-vs-execution
classification accuracy (that would be fake). Two facts save the design:
1. their recovery strategy is the same in v1 (redo once), so not distinguishing
   them does not hurt the recovery rate;
2. the reliably-distinguishable and most valuable boundary is **logic vs
   non-logic** (logic is evidenced by "already retried and still failing" in the
   history — confirmed by the Stage B live smoke: 2 failed retries → logic 0.95).

So `experiments/eval_reflection.py`, over N injected-failure runs, reports:

| Metric | Meaning |
|---|---|
| **Goal success rate** (baseline vs proposed) | no-reflection while loop vs reflective graph, under injected failure — the core "reflection helps" delta |
| **Recovery rate** | injected one-shot perception/execution failure: fraction the reflect→redo round recovers |
| **Logic-escalation accuracy / 误判率** | injected PERSISTENT failure: fraction reflection correctly escalates to logic (stops retrying, does not miss the "bug"); and the inverse (one-shot failure wrongly called logic = false alarm) |
| avg recovery rounds | cost of reflection |

**Honest scope** (threats to validity, write it in the paper):
- perception vs execution NOT strongly distinguished (same observable + same
  recovery); distinguishing them needs a re-observe-then-compare signal → future work.
- real logic bugs (mutation) deferred to Phase 4; v1 approximates with a
  persistent injected failure ("retries keep failing → should be logic").

Targets: proposed goal success >> baseline; perception/execution recovery
≥ 70-80%; logic-escalation accuracy high (≥ ~80%) with low false alarms.

## 7. Success criteria (Phase 3 v1 done)

- Loop detects per-step anomalies and routes them to reflection.
- Reflection classifies into the 3 types and triggers the matching recovery.
- eval_reflection produces the §6 table (baseline vs proposed).
- 1–2 written case studies (one per failure type) for the thesis / README.

## 8. Implementation stages

| Stage | Content |
|---|---|
| 0 | verify langgraph installs + a 3-node toy StateGraph runs (no LangChain) |
| A | per-step expectation + anomaly detection (works for both loop versions) |
| B | `agent/reflection.py` (classify + recovery dispatch) + tests (fake LLM) |
| C | `agent/graph.py` — LangGraph StateGraph wiring decide/act/reflect/recover nodes (reuses B); keep Phase 2 while-loop as the no-reflection baseline |
| D | failure-injection wrappers (perception, execution) + tests |
| E | `experiments/eval_reflection.py` (baseline while-loop vs proposed graph) + metrics table + case studies |

## 9. June packaging note (the actual deliverable)

Per the 2026-06 priority decision (complete story + packaging): after Phase 3
works, W3 is a packaging pass (README + architecture diagram + demo GIF +
Phase 1/3 metric tables + repo cleanup). If Phase 3 slips past ~June 14,
FREEZE it and package Phase 0–2 — never end June with undocumented code.
