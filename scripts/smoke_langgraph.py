"""Phase 3 Stage 0 smoke: prove LangGraph runs with a conditional edge.

This is the de-risking step before agent/graph.py. It proves:
  (a) langgraph imports and a StateGraph compiles + runs,
  (b) a CONDITIONAL edge routes state down different branches (the whole point
      for Phase 3: check -> success/continue/anomaly, reflect -> 3 failure types),
  (c) nodes are plain Python functions -- NO LangChain LLM wrappers needed
      (the real graph's nodes will call our own DeepSeek client / TestActions).

Run:
    python scripts/smoke_langgraph.py
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: int
    path: str


def start_node(state: State) -> dict:
    # plain function: receives state, returns a partial update
    return {"value": state["value"]}


def branch_positive(state: State) -> dict:
    return {"path": f"positive({state['value']})"}


def branch_nonpositive(state: State) -> dict:
    return {"path": f"nonpositive({state['value']})"}


def router(state: State) -> str:
    # this is the conditional edge: pick the next node by inspecting state
    return "pos" if state["value"] > 0 else "nonpos"


def build_app():
    graph = StateGraph(State)
    graph.add_node("start", start_node)
    graph.add_node("branch_positive", branch_positive)
    graph.add_node("branch_nonpositive", branch_nonpositive)

    graph.add_edge(START, "start")
    graph.add_conditional_edges(
        "start",
        router,
        {"pos": "branch_positive", "nonpos": "branch_nonpositive"},
    )
    graph.add_edge("branch_positive", END)
    graph.add_edge("branch_nonpositive", END)
    return graph.compile()


def main() -> None:
    app = build_app()
    for v in (5, -3):
        result = app.invoke({"value": v, "path": ""})
        print(f"input value={v:>3}  ->  path={result['path']}")
    print("OK: conditional-edge StateGraph runs; no LangChain import used.")


if __name__ == "__main__":
    main()
