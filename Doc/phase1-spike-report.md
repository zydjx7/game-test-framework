# Phase 1 Spike Report — VLM Ammo Perception on ViZDoom

> **Date**: 2026-06-01
> **Question (design doc §1)**: Can Qwen3-VL-Flash read the ammo HUD on
> defend_the_center end-to-end, and roughly how accurate is it?
> **Answer**: Yes. Concrete ammo reading is **100% (104/104)** on settled
> frames. The pipeline works; proceed to Phase 2.

## 1. Headline numbers

| Metric | Result |
|---|---|
| Frames evaluated | 104 (5 episodes × ~21 ammo-change keyframes) |
| **Concrete accuracy** (exact ammo match) | **104/104 = 100.0%** |
| Abstract accuracy (VLM self-reported high/med/low) | 83/104 = 79.8% |
| Failures (null / malformed JSON / backend error) | 0/104 |
| Mean latency | ~4.9 s / frame |
| Tokens | 54,600 in / 1,541 out |
| Est. cost | ~¥0.011 (whole run; verify in 阿里云百炼 console) |

Backend: `qwen3-vl-flash` via DashScope OpenAI-compatible endpoint.
Scenario: `defend_the_center`, constant-ATTACK policy, render_hud=True.

## 2. The result that almost fooled us (key methodology finding)

The FIRST eval reported **concrete 4.8% (5/104)** with a systematic
`VLM = GT + 1` pattern. That number is a trap: it looks like "the VLM reads
ammo but is imprecise," which would wrongly kill the whole VLM approach.

Root cause was **not** the VLM. At the tic the ammo game-variable decrements,
the rendered HUD still shows the previous digit for ~1 tic (`screen_buffer`
lags `game_variables`). The sampler was picking the ammo-change *transition*
frame — exactly the desynced edge — so the VLM read the old HUD digit = GT+1.

Diagnostic (100% reproducible): on a fixed ammo plateau, the transition frame
gives `VLM=GT+1` (MISS) while middle/last frames give `VLM=GT` (OK).

Fix: sample the **middle frame of each constant-ammo run** (HUD settled).
Concrete accuracy jumped 4.8% → 100%.

> **Paper material (threats to validity)**: a rendered frame and the state
> vector used as ground truth must correspond. A 1-tic render/state desync
> silently corrupts perception accuracy. This belongs in Section IV.A
> methodology. It was caught because the errors were *systematically* +1 —
> random VLM noise would not be.

## 3. Concrete is perfect, abstract is not — and why

Concrete reading is 100%, yet the VLM's **self-reported** level
(high/med/low) disagrees with our boundary function on 21/104 frames. In
**every** one of those 21, `vlm_ammo == gt_ammo` (the digit was read
correctly) — the mismatch is purely the VLM's categorical judgment:

| VLM said | Our definition | Examples (ammo) |
|---|---|---|
| "high" | medium | 16 |
| "low" | medium | 9, 10, 11, 12, 13 |

The prompt explicitly stated the thresholds (`medium = 9–17`), yet the VLM
applies its own prior intuition near the boundaries instead of the arithmetic.

**Lesson**: for clear HUD digits, do **not** trust the VLM's self-reported
abstract level. Derive the level in code from the VLM's concrete number using
the shared `ammo_level()` — that yields 100% abstract accuracy for free
(since concrete is 100%). The TITAN "abstract-then-concrete" step earns its
keep on *harder* perception (blur, motion, partial occlusion, semantic fields
like `enemy_visible`), not on legible digits.

## 4. Decision

- ✅ **Perception link works; concrete ammo reading is essentially solved on
  clear HUD digits.** Proceed to Phase 2 (action + agent loop).
- 🔧 **Prompt/scoring v2 (small, deferred)**: either drop the abstract level
  from the VLM output entirely and compute it from the number, or keep it only
  as a diagnostic. Do this when perception is reused in Phase 2, not now.
- ⚠️ **Latency ~5 s/frame** is fine for offline eval but will pace the Phase 2
  online agent loop. Note for later: batch, cache, or a local model.

## 5. What this spike de-risked (3 blocking issues caught early)

1. `basic.wad` ammo never varies ([46,50]) → switched to `defend_the_center`.
2. `render_hud = false` by default → VLM had no digits to read → added
   `VizDoomEnv(render_hud=True)`.
3. HUD/state 1-tic desync → 4.8% artifact → middle-frame sampling.

Each would have wasted days (or produced a wrong conclusion) if we had jumped
straight to a full multi-backend evaluation. The spike did its job.

## 6. Reproduce

```powershell
python scripts/record_spike_trajectories.py --episodes 5 --max-tics 300
python experiments/eval_perception_spike.py
```

Outputs `experiments/spike_results_<ts>.csv` (gitignored) + the summary above.
