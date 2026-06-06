"""Reflective agent as a LangGraph StateGraph (Phase 3 Stage C; Stage-1 step 3).

Nodes are plain functions that reuse the already-built pieces; LangGraph just
orchestrates the branching control flow that a while loop would express messily.

    decide -> act -> [route_anomaly]
                       success  -> mark_success -> END
                       maxsteps -> mark_maxsteps -> END
                       report   -> report -> END        (anomaly survives, ladder exhausted)
                       reflect  -> reflect -> [route_recovery]
                                                re_observe -> re_observe -> [route_anomaly]
                                                retry      -> retry      -> [route_anomaly]
                       continue -> decide

DIAGNOSTIC RECOVERY LADDER (step 3). Perception and execution failures share the
SAME observable at the moment of the anomaly (ADR-0003), so we do NOT trust the
LLM's perception-vs-execution label to pick a recovery. Instead we disambiguate
EMPIRICALLY by escalating through increasing-side-effect recoveries:

    1. re_observe  -- re-perceive the CURRENT state without acting. A perception
                      / transient-observation fault clears here (the effect did
                      happen, we just misread it last time). Zero side effects;
                      does NOT advance `steps`.
    2. retry       -- re-run the action. An execution fault (the action did not
                      go through) needs this; re_observe alone could not fix it.
    3. report      -- both non-logic recoveries failed -> escalate to SUSPECTED
                      logic (route_anomaly routes here once the ladder is spent).

So the recovery OUTCOME is itself the diagnostic signal: a perception fault
recovers at rung 1, an execution fault needs rung 2, a real (logic) bug survives
both and is reported. We do NOT claim to classify perception vs execution up
front -- the ladder distinguishes them by what it takes to recover. The LLM's
3-type diagnosis is RECORDED (taxonomy / case library) and noted in the report as
corroborating evidence, but it does not by itself drive the control flow: a bug
is only reported on STRUCTURAL evidence (re_observe failed AND retry failed), not
on a single LLM guess made before any recovery was tried.

Budgets `max_reobserves` / `max_retries` are PARAMETERS (default 1/1 = the
minimal, cleanest ladder), recorded in the state and bug report so an experiment
can vary them as a separate, pre-declared config rather than tuning per case.
They reset to 0 once an anomaly is resolved, so the budget is PER-ANOMALY: a
recovered anomaly does not exhaust the budget for a later one.

Nodes reuse: FunctionCallingDecider (decide), TestActions.run / .observe /
check_expectation (act / retry / re_observe), Reflector (reflect). The Phase 2
while-loop run_agent_loop is left untouched as the no-reflection baseline; this
graph is the proposed (with-reflection) version for the Stage E comparison.

Goal success is judged on the CUMULATIVE result (first observed <m> .. latest),
exactly like the while loop, so the two are comparable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from actions.result import accumulate, snapshot_result


class AgentState(TypedDict, total=False):
    history: List[Dict[str, Any]]
    cumulative: Dict[str, Any]
    last_action: str
    last_result: Dict[str, Any]
    anomaly: Optional[Dict[str, Any]]
    last_failure_type: Optional[str]
    cases: List[Dict[str, Any]]
    status: str
    step: int
    reobserve_attempts: int
    retry_attempts: int
    bug_report: Optional[Dict[str, Any]]


def _tools_spec(goal: Any, action_lib: Any) -> List[Dict[str, str]]:
    descriptions = getattr(action_lib, "DESCRIPTIONS", {})
    return [
        {"name": name, "description": descriptions.get(name, name)}
        for name in goal.available_actions
    ]


_BEFORE_SUFFIX = "_before"


def _metrics_of(result: Dict[str, Any]) -> List[str]:
    """Metric names a result observed, inferred from its `<metric>_before` keys."""

    return sorted({k[: -len(_BEFORE_SUFFIX)] for k in result if k.endswith(_BEFORE_SUFFIX)})


def _apply_result(
    state: AgentState,
    action: str,
    result: Dict[str, Any],
    action_lib: Any,
    recovery: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared state update for `act` and the `retry` recovery (both ADVANCE a
    step because they run the action). `recovery` tags the history entry."""

    step = state.get("step", 0) + 1
    cumulative = accumulate(state.get("cumulative", {}), result, step, action)
    entry: Dict[str, Any] = {"step": state.get("step", 0), "action": action, "result": result}
    if recovery:
        entry["recovery"] = recovery
    history = state.get("history", []) + [entry]
    anomaly = action_lib.check_expectation(action, result)
    return {
        "last_result": result,
        "history": history,
        "cumulative": cumulative,
        "anomaly": anomaly,
        "step": step,
    }


def _mark_case_and_budget(
    state: AgentState, update: Dict[str, Any], anomaly: Optional[Dict[str, Any]]
) -> None:
    """After a recovery: mark the most recent case recovered iff the anomaly is
    gone, and (if recovered) reset the ladder budget so a FUTURE anomaly gets a
    fresh budget (per-anomaly, not global)."""

    cases = list(state.get("cases", []))
    if cases:
        cases[-1] = {**cases[-1], "recovered": anomaly is None}
        update["cases"] = cases
    if anomaly is None:
        update["reobserve_attempts"] = 0
        update["retry_attempts"] = 0


def build_reflective_app(
    goal: Any,
    action_lib: Any,
    perceptor: Any,
    decider: Any,
    reflector: Any,
    max_steps: int = 8,
    max_reobserves: int = 1,
    max_retries: int = 1,
):
    """Compile the reflective StateGraph. Dependencies are captured by closure;
    AgentState carries only data."""

    tools_spec = _tools_spec(goal, action_lib)

    # -- nodes -----------------------------------------------------------
    def decide(state: AgentState) -> Dict[str, Any]:
        choice = decider.decide(goal.description, state.get("history", []), tools_spec)
        return {"last_action": choice}

    def act(state: AgentState) -> Dict[str, Any]:
        result = action_lib.run(state["last_action"], perceptor)
        return _apply_result(state, state["last_action"], result, action_lib)

    def reflect(state: AgentState) -> Dict[str, Any]:
        case = reflector.reflect(state["anomaly"], state.get("history", []))
        cases = state.get("cases", []) + [case.to_dict()]
        # The LLM's 3-type diagnosis is RECORDED (taxonomy / case library), but
        # only its logic-vs-non-logic call drives control flow (route_recovery);
        # perception-vs-execution is resolved by the ladder, not this label.
        return {"cases": cases, "last_failure_type": case.failure_type.value}

    def re_observe(state: AgentState) -> Dict[str, Any]:
        """Recovery rung 1: re-perceive the CURRENT state WITHOUT acting.

        Keeps each metric's prior `<metric>_before`, refreshes `<metric>_after`
        from a fresh observation, and rebuilds the result via snapshot_result so
        the canonical shape is preserved. Does NOT advance `step` (no action was
        taken) and keeps the ORIGINAL `last_action` in the cumulative/history so
        goal/history semantics are not polluted by the re-read.
        """

        prior = state.get("last_result", {})
        metrics = _metrics_of(prior)
        perceived = action_lib.observe(perceptor)
        before = {m: prior.get(f"{m}{_BEFORE_SUFFIX}") for m in metrics}
        after = {m: getattr(perceived, m, None) for m in metrics}
        result = snapshot_result(before, after)

        action = state["last_action"]
        step = state.get("step", 0)  # NOT incremented: re_observe is not an action
        cumulative = accumulate(state.get("cumulative", {}), result, step, action)
        anomaly = action_lib.check_expectation(action, result)
        history = state.get("history", []) + [
            {"step": step, "action": action, "result": result, "recovery": "re_observe"}
        ]
        update: Dict[str, Any] = {
            "last_result": result,
            "cumulative": cumulative,
            "anomaly": anomaly,
            "history": history,
            "reobserve_attempts": state.get("reobserve_attempts", 0) + 1,
        }
        _mark_case_and_budget(state, update, anomaly)
        return update

    def retry(state: AgentState) -> Dict[str, Any]:
        """Recovery rung 2: re-run the SAME action (an execution fault needs the
        action to actually go through; re_observe alone could not fix it)."""

        result = action_lib.run(state["last_action"], perceptor)
        update = _apply_result(state, state["last_action"], result, action_lib, recovery="retry")
        update["retry_attempts"] = state.get("retry_attempts", 0) + 1
        _mark_case_and_budget(state, update, update["anomaly"])
        return update

    def report(state: AgentState) -> Dict[str, Any]:
        reob = state.get("reobserve_attempts", 0)
        ret = state.get("retry_attempts", 0)
        budgets_exhausted = reob >= max_reobserves and ret >= max_retries
        # Honest framing: this is SUSPECTED logic under the current calibration +
        # recovery budget, not a verdict that the game definitely has a bug.
        return {
            "status": "bug_reported",
            "bug_report": {
                "goal": goal.description,
                "anomaly": state.get("anomaly"),
                "suspected_logic": True,
                "evidence": {
                    "reobserve_attempts": reob,
                    "retry_attempts": ret,
                    "budgets_exhausted": budgets_exhausted,
                    "diagnosed_logic_by_llm": state.get("last_failure_type") == "logic",
                },
                "recommended_next": [
                    "increase the observation window (adapter timing calibration)",
                    "rerun with a larger recovery budget (max_reobserves / max_retries)",
                    "inspect the adapter timing / mechanic for this action",
                ],
                "history": state.get("history", []),
                "cases": state.get("cases", []),
            },
        }

    def mark_success(state: AgentState) -> Dict[str, Any]:
        return {"status": "success"}

    def mark_maxsteps(state: AgentState) -> Dict[str, Any]:
        return {"status": "max_steps_exceeded"}

    # -- routers ---------------------------------------------------------
    def _ladder_budget_remaining(state: AgentState) -> bool:
        return (
            state.get("reobserve_attempts", 0) < max_reobserves
            or state.get("retry_attempts", 0) < max_retries
        )

    def route_anomaly(state: AgentState) -> str:
        """After act / re_observe / retry: decide where to go next."""

        if goal.is_satisfied(state.get("cumulative", {})):
            return "success"
        if state.get("anomaly"):
            if _ladder_budget_remaining(state):
                return "reflect"
            return "report"  # ladder exhausted, anomaly persists -> suspected logic
        if state.get("step", 0) >= max_steps:
            return "maxsteps"
        return "continue"

    def route_recovery(state: AgentState) -> str:
        """After reflect: pick the next ladder rung (re_observe before retry).

        The LLM's diagnosis (perception / execution / LOGIC) is RECORDED but does
        NOT drive routing -- not even a logic call short-circuits here. We never
        report a suspected bug without first trying the cheap recoveries; logic is
        reported STRUCTURALLY (route_anomaly -> report) only once the ladder is
        exhausted and the anomaly survives. This keeps the bug signal evidence-
        based ("re_observe failed AND retry failed") rather than a single LLM
        guess made before any recovery was attempted. `report` here is only a
        safety fallback (route_anomaly gates entry to reflect on budget)."""

        if state.get("reobserve_attempts", 0) < max_reobserves:
            return "re_observe"
        if state.get("retry_attempts", 0) < max_retries:
            return "retry"
        return "report"

    # -- wiring ----------------------------------------------------------
    g = StateGraph(AgentState)
    for name, fn in [
        ("decide", decide), ("act", act), ("reflect", reflect),
        ("re_observe", re_observe), ("retry", retry), ("report", report),
        ("mark_success", mark_success), ("mark_maxsteps", mark_maxsteps),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "decide")
    g.add_edge("decide", "act")
    anomaly_targets = {
        "success": "mark_success",
        "maxsteps": "mark_maxsteps",
        "reflect": "reflect",
        "report": "report",
        "continue": "decide",
    }
    g.add_conditional_edges("act", route_anomaly, anomaly_targets)
    g.add_conditional_edges(
        "reflect", route_recovery,
        {"re_observe": "re_observe", "retry": "retry", "report": "report"},
    )
    g.add_conditional_edges("re_observe", route_anomaly, anomaly_targets)
    g.add_conditional_edges("retry", route_anomaly, anomaly_targets)
    g.add_edge("report", END)
    g.add_edge("mark_success", END)
    g.add_edge("mark_maxsteps", END)
    return g.compile()


def run_reflective_agent(
    goal: Any,
    action_lib: Any,
    perceptor: Any,
    decider: Any,
    reflector: Any,
    max_steps: int = 8,
    max_reobserves: int = 1,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Run one goal through the reflective graph; return a summary dict
    shaped like run_agent_loop's output (plus cases / bug_report)."""

    app = build_reflective_app(
        goal, action_lib, perceptor, decider, reflector,
        max_steps, max_reobserves, max_retries,
    )
    action_lib.prim.reset()
    initial: AgentState = {
        "history": [], "cumulative": {},
        "cases": [], "status": "running", "step": 0,
        "reobserve_attempts": 0, "retry_attempts": 0,
        "anomaly": None, "bug_report": None,
    }
    final = app.invoke(initial, config={"recursion_limit": 50})
    return {
        "status": final.get("status", "running"),
        "steps": final.get("step", 0),
        "cumulative": final.get("cumulative", {}),
        "history": final.get("history", []),
        "cases": final.get("cases", []),
        "bug_report": final.get("bug_report"),
        "goal": goal.description,
    }
