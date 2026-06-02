"""Reflection: classify an anomaly into 3 failure types + pick a recovery.

When a step violates its expected effect (TestActions.check_expectation returns
an anomaly), the reflector asks the LLM to diagnose WHY:

    PERCEPTION  the state was misread (VLM/CV wrong) -- the game is fine
    EXECUTION   the action did not go through
    LOGIC       the game itself is wrong (a real bug)

The LLM only CLASSIFIES (+ confidence + reasoning); the recovery action is
looked up from a fixed table (research-plan.md §5.3), keeping v1 deterministic.
Diagnosis uses DeepSeek native function calling for structured output, mirroring
the decider. The returned ReflectionCase is RAG-ready (Doc/phase3-design.md
§4.5): v1 just logs it; W4 will embed + retrieve similar past cases.

The reflector does NOT execute recovery -- that is the graph's job (Stage C).
It is the "brain" (diagnose); the graph is the "hands" (act on the diagnosis).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureType(str, Enum):
    PERCEPTION = "perception"
    EXECUTION = "execution"
    LOGIC = "logic"


# research-plan.md §5.3: one recovery strategy per failure type (v1, fixed).
RECOVERY: Dict[FailureType, str] = {
    FailureType.PERCEPTION: "re_observe",  # look again / re-query / vote
    FailureType.EXECUTION: "retry",        # redo the action
    FailureType.LOGIC: "report",           # mark suspected bug, stop retrying
}


@dataclass
class ReflectionCase:
    """RAG-ready record of one reflection (Doc/phase3-design.md §4.5)."""

    anomaly: Dict[str, Any]
    failure_type: FailureType
    recovery_action: str
    confidence: float
    reasoning: str
    recovered: Optional[bool] = None  # filled after recovery runs (Stage C)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly": self.anomaly,
            "failure_type": self.failure_type.value,
            "recovery_action": self.recovery_action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recovered": self.recovered,
        }


_SYSTEM_PROMPT = (
    "You diagnose why an automated game-testing action did not have its expected "
    "effect. Classify the cause as exactly one of: perception (the state was "
    "misread, but the game is fine), execution (the action did not go through), "
    "or logic (the game itself is wrong -- a real bug). Be CONSERVATIVE about "
    "'logic': prefer perception or execution unless there is evidence, such as "
    "the same action already retried and still failing in the recent steps."
)

_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_failure",
        "description": "Report the diagnosed failure type with confidence and reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "failure_type": {
                    "type": "string",
                    "enum": ["perception", "execution", "logic"],
                },
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["failure_type", "confidence", "reasoning"],
        },
    },
}


class Reflector:
    """Diagnoses anomalies via DeepSeek function calling.

    Inject a `client` (OpenAI-compatible) for tests; otherwise one is built from
    the project's DeepSeek config.
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

    def reflect(
        self, anomaly: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> ReflectionCase:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _render_prompt(anomaly, history)},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[_CLASSIFY_TOOL],
            # NOTE: deepseek-v4-flash runs in "thinking mode", which rejects a
            # forced tool_choice ({"type":"function",...}) with a 400. Use "auto"
            # (proven by the decider) plus a single tool + an explicit "call
            # classify_failure" instruction so the model reliably calls it.
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            raise RuntimeError(
                f"reflector got no tool_call; content={getattr(message, 'content', None)!r}"
            )

        args = json.loads(tool_calls[0].function.arguments)
        failure_type = FailureType(args["failure_type"])
        return ReflectionCase(
            anomaly=anomaly,
            failure_type=failure_type,
            recovery_action=RECOVERY[failure_type],
            confidence=float(args.get("confidence", 0.0)),
            reasoning=str(args.get("reasoning", "")),
        )


def _render_prompt(anomaly: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    lines = [
        "An action did not have its expected effect.",
        f"Action: {anomaly.get('action')}",
        f"Expected: {anomaly.get('expected')}",
        f"Observed result: {anomaly.get('result')}",
        "",
    ]
    if history:
        lines.append("Recent steps this episode (action -> ammo delta):")
        for h in history:
            result = h.get("result", {})
            lines.append(f"  step {h.get('step')}: {h.get('action')} -> delta {result.get('delta')}")
    else:
        lines.append("No prior steps this episode.")
    lines.append("")
    lines.append("Call classify_failure with your diagnosis.")
    return "\n".join(lines)
