"""Canonical result + cumulative schema (Phase 3.5 generalization).

The agent core must not hardcode any specific metric (it used to assume ammo).
Composites report results through `snapshot_result`, and the agent loop / graph
fold them with `accumulate`. Both rely on ONE naming convention:

    <metric>_before / <metric>_after   (flat, top-level keys)

`snapshot_result` is the single construction point, so the convention cannot
drift into before_ammo / ammoStart / ammo_delta etc. Goal success is a sandboxed
eval over the (flat) cumulative dict, e.g. `ammo_before - ammo_after >= 1` or
`score_after - score_before >= 1`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def snapshot_result(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a result dict in canonical <metric>_before/<metric>_after form.

    `before` / `after` are {metric_name: value} over the metrics this composite
    observed (e.g. {"ammo": 26}). This is the ONLY place result keys are coined,
    which is what keeps the flat naming consistent.
    """

    result: Dict[str, Any] = {}
    for metric, value in before.items():
        result[f"{metric}_before"] = value
    for metric, value in after.items():
        result[f"{metric}_after"] = value
    return result


def accumulate(
    cumulative: Mapping[str, Any],
    result: Mapping[str, Any],
    step: int,
    action: str,
) -> Dict[str, Any]:
    """Fold one step's result into the cumulative, generically (no metric is
    hardcoded): the FIRST `<m>_before` seen this episode is kept, the LATEST
    `<m>_after` wins, plus `steps` and `last_action`.
    """

    cum: Dict[str, Any] = dict(cumulative)
    for key, value in result.items():
        if key.endswith("_before"):
            cum.setdefault(key, value)  # first observation wins
        elif key.endswith("_after"):
            cum[key] = value  # latest observation wins
    cum["steps"] = step
    cum["last_action"] = action
    return cum


def summarize_result(result: Mapping[str, Any]) -> str:
    """Human/LLM-readable one-line summary, generic over metrics.

    e.g. {"ammo_before": 26, "ammo_after": 25} -> "ammo 26->25".
    """

    metrics = sorted({k[:-7] for k in result if k.endswith("_before")})
    parts = [f"{m} {result.get(f'{m}_before')}->{result.get(f'{m}_after')}" for m in metrics]
    return ", ".join(parts) if parts else "no change"
