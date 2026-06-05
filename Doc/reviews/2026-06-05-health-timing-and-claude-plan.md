# Review: health timing and the Stage 1b decision

**Author**: Codex  
**Date**: 2026-06-05  
**Context**: The human asked Codex to review Claude's recommendation to defer
ViZDoom health work and prioritize step 3 (`re_observe` / `retry` split). The
discussion centered on whether `defend_the_center` health behavior is too
unstable to justify adding a health mechanic now.

## Summary

Codex partially agreed with Claude's caution but changed the execution plan.

Claude's strongest point was correct: a health assertion should not be written
against a short or poorly understood observation window. However, Codex did not
accept the conclusion that Stage 1b health should be deferred. The better fix
was to make timing calibration explicit in the adapter layer and choose a stable
scenario for the health assertion.

Result: Stage 1b health was implemented in ViZDoom via `health_gathering`, with
no agent-layer changes.

## What Claude proposed

Claude's proposal, as summarized by the human:

- `defend_the_center` standing still appeared not to decrease health.
- `defend_the_center` firing sometimes decreased health, but not reliably.
- Therefore, health in `defend_the_center` was not a clean mechanic to test.
- Claude recommended delaying Stage 1b health and prioritizing step 3:
  separating `re_observe` and `retry` recovery behavior.
- If health were implemented now, Claude suggested `health_gathering` plus a
  `wait_and_check_health` composite.

## What Codex verified

Codex ran live ViZDoom probes before deciding.

Observed results:

- `health_gathering` with no-op is stable: first health change at tic 32,
  health `92 -> 84`.
- `defend_the_center` with no-op does decrease health, but much later:
  first health changes were observed around tic 160-170.
- Therefore, the earlier "standing still never decreases health" interpretation
  was not generally true in this environment. The real issue was timing:
  the current short wait window was too small to observe the delayed effect.

This reframed the issue from "health is not worth adding" to "health must be
tested with a calibrated observation window."

## Decision

Codex implemented Stage 1b health now, but only on the stable path:

- Use `health_gathering` for the real health assertion.
- Add `wait_and_check_health`.
- Poll every 32 tics, with a 64-tic maximum observation window.
- Return canonical `health_before` / `health_after` using `snapshot_result`.
- Use shared `decreased("health")` expectation predicate.
- Do not modify the agent layer.
- Document `defend_the_center` health as long-window / stochastic, not a precise
  one-step assertion.

This is an "adopt + improve" outcome:

- Adopt Claude's warning against unstable `defend_the_center` health assertions.
- Adopt Claude's fallback of `health_gathering`.
- Improve the plan by making timing calibration an explicit adapter contract,
  rather than treating it as one-off tuning.
- Decline delaying Stage 1b health, because ammo-only ViZDoom remained too thin
  and health could be implemented safely without touching agent logic.

## Implemented changes

Commit: `930772b shared: add calibrated ViZDoom health mechanic`

Changed / added files:

- `actions/primitives.py`
  - Added `HEALTH_GATHERING_POLL_TICS = 32`
  - Added `HEALTH_GATHERING_OBSERVATION_TICS = 64`
- `actions/composites.py`
  - Added `wait_and_check_health`
  - Added description and expectation for the new template
  - Uses `snapshot_result({"health": before.health}, {"health": after.health})`
- `actions/__init__.py`
  - Exports the health timing constants
- `agent/health_gathering_goals.feature`
  - Adds a scenario-specific health goal for `health_gathering`
- `experiments/vizdoom/probe_timing.py`
  - Adds a reusable timing probe for future mechanics
- `tests/test_actions.py`
  - Adds health timing, composite, and expectation tests
- `tests/test_agent_loop.py`
  - Proves the existing agent loop can satisfy a health goal unchanged
- `Doc/adapter-contract.md`
  - Adds "Timing calibration" as an adapter contract section
  - Documents current ViZDoom timing windows
- `WORKLOG.md`
  - Records the Stage 1b health completion and timing caveat

## Verification

Commands / smokes run:

- `python -m pytest tests/test_actions.py tests/test_agent_loop.py`
  - 26 passed
- `python -m pytest --basetemp .pytest_tmp`
  - 115 passed, 4 deselected
- Live `health_gathering` smoke through the real agent loop components:
  - `success 1 {'health_before': 92, 'health_after': 84, ...}`
- `probe_timing.py` on `health_gathering`:
  - first health change at tic 32 in all sampled episodes
- `probe_timing.py` on `defend_the_center`:
  - first health changes around tic 160-170 in sampled episodes

## Guidance for Claude / future agents

Do not treat a failed short-window assertion as a logic bug until the adapter's
observation window has been calibrated.

For future mechanics, use this workflow:

1. Probe the real scenario timing.
2. Classify timing as stable or stochastic / long-window.
3. Encode stable timing in the adapter composite.
4. Avoid using stochastic timing as a precise one-step assertion.
5. Return all results via `snapshot_result`.
6. Keep the agent layer unchanged unless the task is explicitly step 3 or later.

Recommended next task:

- If continuing Stage 1b maturity: consider ammo-bounds or death/done, but first
  calibrate their observation windows and decide whether they fit the flat result
  schema.
- If moving to reflection quality: step 3 (`re_observe` vs `retry`) is still a
  strong next task, now with a richer health mechanic available for future tests.
