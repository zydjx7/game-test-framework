"""Minimal agent loop + DeepSeek native function-calling decider (Phase 2).

The loop is reactive (observe -> decide -> act -> check), no reflection
(Phase 3). The decider uses DeepSeek's native `tools` / `tool_calls` API so
"Tool Use / Function Calling" is genuinely exercised, not prompt-hacked.

Goal success is judged on a CUMULATIVE result across steps (generic over any
metric, see actions/result.accumulate): for every observed metric the first
`<m>_before` and latest `<m>_after` are kept, plus `steps`. This lets one goal
("fire reduces ammo by >= 3") span multiple steps while a single-step goal
("fire reduces ammo by >= 1") still resolves immediately.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from actions.result import accumulate, summarize_result

SYSTEM_PROMPT = (
    "You are an automated game-testing agent for an FPS game. Given a test "
    "goal and the result history so far, call EXACTLY ONE of the available "
    "action tools that best progresses the goal. Match the tool's semantics to "
    "the goal: if the goal is about NOT firing / staying idle, do not fire."
)


def _tools_spec(goal: Any, action_lib: Any) -> List[Dict[str, str]]:
    descriptions = getattr(action_lib, "DESCRIPTIONS", {})
    return [
        {"name": name, "description": descriptions.get(name, name)}
        for name in goal.available_actions
    ]


def run_agent_loop(
    goal: Any,
    action_lib: Any,
    perceptor: Any,
    decider: Any,
    max_steps: int = 8,
) -> Dict[str, Any]:
    """Drive one goal to success or until max_steps."""

    action_lib.prim.reset()
    tools_spec = _tools_spec(goal, action_lib)

    history: List[Dict[str, Any]] = []
    cumulative: Dict[str, Any] = {}

    for step in range(max_steps):
        choice = decider.decide(goal.description, history, tools_spec)
        result = action_lib.run(choice, perceptor)

        cumulative = accumulate(cumulative, result, step + 1, choice)
        history.append({"step": step, "action": choice, "result": result})

        if goal.is_satisfied(cumulative):
            return {
                "status": "success",
                "steps": step + 1,
                "cumulative": cumulative,
                "history": history,
                "goal": goal.description,
            }

    return {
        "status": "max_steps_exceeded",
        "steps": max_steps,
        "history": history,
        "goal": goal.description,
    }


class FunctionCallingDecider:
    """Picks the next action via DeepSeek native function calling.

    Inject a `client` (OpenAI-compatible) for tests; if omitted, one is built
    from the project's DeepSeek config in `src.llm.client_helpers`.
    """

    def __init__(self, client: Any = None, model: Optional[str] = None) -> None:
        if client is None:
            from openai import OpenAI
            from src.llm.client_helpers import (
                create_openai_client,
                load_deepseek_config,
            )

            cfg = load_deepseek_config()
            client = create_openai_client(OpenAI, cfg.api_key, cfg.base_url)
            model = model or cfg.model
        self.client = client
        self.model = model or "deepseek-chat"

    def decide(
        self,
        goal_description: str,
        history: List[Dict[str, Any]],
        tools_spec: List[Dict[str, str]],
    ) -> str:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for t in tools_spec
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _render_user_prompt(goal_description, history)},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return tool_calls[0].function.name

        raise RuntimeError(
            f"decider got no tool_call; model content={getattr(message, 'content', None)!r}"
        )


def _render_user_prompt(goal_description: str, history: List[Dict[str, Any]]) -> str:
    lines = [f"Test goal: {goal_description}", ""]
    if history:
        lines.append("Actions taken so far (action -> result):")
        for h in history:
            lines.append(f"  step {h['step']}: {h['action']} -> {summarize_result(h['result'])}")
    else:
        lines.append("No actions taken yet.")
    lines.append("")
    lines.append("Call exactly one action tool to best progress the goal.")
    return "\n".join(lines)
