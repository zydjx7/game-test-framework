"""Reflective agent as a LangGraph StateGraph (Phase 3 Stage C).

This is the real version of scripts/smoke_langgraph.py: nodes are plain
functions that reuse the already-built pieces, LangGraph just orchestrates the
branching control flow that a while loop would express messily.

    decide -> act -> [route_after_act]
                       success  -> mark_success -> END
                       maxsteps -> mark_maxsteps -> END
                       reflect  -> reflect -> [route_recovery]
                                                re_observe -> redo -> [route_after_act]
                                                retry      -> redo -> [route_after_act]
                                                report     -> END
                       continue -> decide

Nodes reuse: FunctionCallingDecider (decide), TestActions.run +
check_expectation (act / redo), Reflector (reflect). The Phase 2 while-loop
run_agent_loop is left untouched as the no-reflection baseline; this graph is
the proposed (with-reflection) version for the Stage E comparison.

Goal success is judged on the CUMULATIVE result (first observed ammo .. latest),
exactly like the while loop, so the two are comparable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict, total=False):
    history: List[Dict[str, Any]]
    cumulative: Dict[str, Any]
    first_before: Optional[int]
    last_action: str
    last_result: Dict[str, Any]
    anomaly: Optional[Dict[str, Any]]
    last_recovery: Optional[str]
    cases: List[Dict[str, Any]]
    status: str
    step: int
    recovery_attempts: int
    bug_report: Optional[Dict[str, Any]]


def _tools_spec(goal: Any, action_lib: Any) -> List[Dict[str, str]]:
    descriptions = getattr(action_lib, "DESCRIPTIONS", {})
    return [
        {"name": name, "description": descriptions.get(name, name)}
        for name in goal.available_actions
    ]


def _apply_result(state: AgentState, action: str, result: Dict[str, Any], action_lib: Any) -> Dict[str, Any]:
    """Shared state update for both `act` and recovery `redo`."""

    first_before = state.get("first_before")
    if first_before is None:
        first_before = result.get("ammo_before")
    ammo_after = result.get("ammo_after")
    step = state.get("step", 0) + 1
    cumulative = {
        "ammo_before": first_before,
        "ammo_after": ammo_after,
        "delta": (first_before or 0) - (ammo_after or 0),
        "steps": step,
        "last_action": action,
        "last_delta": result.get("delta"),
    }
    history = state.get("history", []) + [
        {"step": state.get("step", 0), "action": action, "result": result}
    ]
    anomaly = action_lib.check_expectation(action, result)
    return {
        "first_before": first_before,
        "last_result": result,
        "history": history,
        "cumulative": cumulative,
        "anomaly": anomaly,
        "step": step,
    }


def build_reflective_app(
    goal: Any,
    action_lib: Any,
    perceptor: Any,
    decider: Any,
    reflector: Any,
    max_steps: int = 8,
    max_recoveries: int = 1,
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
        return {"cases": cases, "last_recovery": case.recovery_action}

    def redo(state: AgentState) -> Dict[str, Any]:
        # one recovery round: re-run the same action (re-observe / retry share
        # this in v1; the distinction is semantic + a future extension point).
        result = action_lib.run(state["last_action"], perceptor)
        update = _apply_result(state, state["last_action"], result, action_lib)
        update["recovery_attempts"] = state.get("recovery_attempts", 0) + 1
        # mark the most recent case recovered iff the anomaly is now gone
        cases = list(state.get("cases", []))
        if cases:
            cases[-1] = {**cases[-1], "recovered": update["anomaly"] is None}
            update["cases"] = cases
        return update

    def report(state: AgentState) -> Dict[str, Any]:
        return {
            "status": "bug_reported",
            "bug_report": {
                "goal": goal.description,
                "anomaly": state.get("anomaly"),
                "history": state.get("history", []),
                "cases": state.get("cases", []),
            },
        }

    def mark_success(state: AgentState) -> Dict[str, Any]:
        return {"status": "success"}

    def mark_maxsteps(state: AgentState) -> Dict[str, Any]:
        return {"status": "max_steps_exceeded"}

    # -- routers ---------------------------------------------------------
    def route_after_act(state: AgentState) -> str:
        if goal.is_satisfied(state.get("cumulative", {})):
            return "success"
        if state.get("anomaly") and state.get("recovery_attempts", 0) < max_recoveries:
            return "reflect"
        if state.get("step", 0) >= max_steps:
            return "maxsteps"
        return "continue"

    def route_recovery(state: AgentState) -> str:
        recovery = state.get("last_recovery")
        if recovery == "report":
            return "report"
        return "redo"  # re_observe and retry both redo the action in v1

    # -- wiring ----------------------------------------------------------
    g = StateGraph(AgentState)
    for name, fn in [
        ("decide", decide), ("act", act), ("reflect", reflect),
        ("redo", redo), ("report", report),
        ("mark_success", mark_success), ("mark_maxsteps", mark_maxsteps),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "decide")
    g.add_edge("decide", "act")
    act_targets = {
        "success": "mark_success",
        "maxsteps": "mark_maxsteps",
        "reflect": "reflect",
        "continue": "decide",
    }
    g.add_conditional_edges("act", route_after_act, act_targets)
    g.add_conditional_edges("reflect", route_recovery, {"redo": "redo", "report": "report"})
    g.add_conditional_edges("redo", route_after_act, act_targets)
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
    max_recoveries: int = 1,
) -> Dict[str, Any]:
    """Run one goal through the reflective graph; return a summary dict
    shaped like run_agent_loop's output (plus cases / bug_report)."""

    app = build_reflective_app(
        goal, action_lib, perceptor, decider, reflector, max_steps, max_recoveries
    )
    action_lib.prim.reset()
    initial: AgentState = {
        "history": [], "cumulative": {}, "first_before": None,
        "cases": [], "status": "running", "step": 0,
        "recovery_attempts": 0, "anomaly": None, "bug_report": None,
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
