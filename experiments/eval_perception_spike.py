"""Phase 1 spike evaluation: VLM vs ground truth on ammo reading.

Pipeline (design doc §1.4 / §5):
  recorded trajectories -> ammo-change keyframes -> VLM perceive
  -> compare against GroundTruthPerceptor -> CSV + summary.

Usage (from project root, with .venv active and DASHSCOPE_API_KEY in .env):

    python experiments/eval_perception_spike.py
    python experiments/eval_perception_spike.py --glob "data/trajectories/defend_the_center_*.pkl"
    python experiments/eval_perception_spike.py --limit 30   # cap total VLM calls

Output: experiments/spike_results_<timestamp>.csv (gitignored) + a printed
summary with concrete/abstract accuracy, latency, token usage, est. cost.
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.trajectory_recorder import load_trajectory  # noqa: E402
from experiments.sampling import sample_ammo_change_keyframes  # noqa: E402
from perception import GroundTruthPerceptor, VLMPerceptor, ammo_level  # noqa: E402
from perception.backends.qwen3_vl_flash import Qwen3VLFlashBackend  # noqa: E402

# Rough Qwen3-VL-Flash pricing (CNY per 1M tokens). VERIFY against the
# 阿里云百炼 console before quoting in the paper; providers change prices often.
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 1.50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="data/trajectories/defend_the_center_*.pkl")
    parser.add_argument("--limit", type=int, default=0, help="Cap total VLM calls (0 = no cap).")
    parser.add_argument("--out-dir", default="experiments")
    args = parser.parse_args()

    paths = sorted(globmod.glob(str(PROJECT_ROOT / args.glob)))
    if not paths:
        print(f"No trajectories matched {args.glob}. Record some first:")
        print("  python scripts/record_spike_trajectories.py")
        return

    gt = GroundTruthPerceptor()
    vlm = VLMPerceptor(Qwen3VLFlashBackend())

    rows = []
    n_concrete_ok = n_abstract_ok = n_fail = 0
    tok_in = tok_out = 0
    total_latency = 0.0

    for path in paths:
        traj = load_trajectory(path)
        episode = Path(path).stem
        for frame in sample_ammo_change_keyframes(traj):
            if args.limit and len(rows) >= args.limit:
                break

            g = gt.perceive(frame.screen, game_variables=frame.game_variables)
            v = vlm.perceive(frame.screen)
            meta = v.raw_response

            gt_level = ammo_level(g.ammo)
            vlm_level = meta.get("vlm_level")
            failed = "error" in meta or v.ammo is None
            concrete_ok = (not failed) and (v.ammo == g.ammo)
            abstract_ok = (not failed) and (vlm_level == gt_level)

            n_concrete_ok += int(concrete_ok)
            n_abstract_ok += int(abstract_ok)
            n_fail += int(failed)
            tok_in += int(meta.get("prompt_tokens", 0) or 0)
            tok_out += int(meta.get("completion_tokens", 0) or 0)
            total_latency += float(meta.get("latency_ms", 0) or 0)

            rows.append({
                "episode": episode,
                "tick": frame.tick,
                "gt_ammo": g.ammo,
                "gt_level": gt_level,
                "vlm_ammo": v.ammo,
                "vlm_level": vlm_level,
                "concrete_ok": int(concrete_ok),
                "abstract_ok": int(abstract_ok),
                "failed": int(failed),
                "latency_ms": round(float(meta.get("latency_ms", 0) or 0), 1),
                "error": meta.get("error", ""),
            })
            print(
                f"{episode} t{frame.tick}: GT={g.ammo}({gt_level}) "
                f"VLM={v.ammo}({vlm_level}) "
                f"{'OK' if concrete_ok else ('FAIL' if failed else 'MISS')}"
            )
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("No keyframes sampled.")
        return

    n = len(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / args.out_dir / f"spike_results_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    est_cost = tok_in / 1e6 * PRICE_IN_PER_M + tok_out / 1e6 * PRICE_OUT_PER_M

    print("\n===== Phase 1 Spike Summary =====")
    print(f"frames evaluated : {n}")
    print(f"concrete accuracy: {n_concrete_ok}/{n} = {n_concrete_ok / n:.1%}  (exact ammo match)")
    print(f"abstract accuracy: {n_abstract_ok}/{n} = {n_abstract_ok / n:.1%}  (high/med/low)")
    print(f"failures         : {n_fail}/{n}  (null / malformed / backend error)")
    print(f"mean latency     : {total_latency / n:.0f} ms/frame")
    print(f"tokens           : {tok_in} in / {tok_out} out")
    print(f"est. cost        : ~CNY {est_cost:.4f}  (rate {PRICE_IN_PER_M}/{PRICE_OUT_PER_M} per 1M; VERIFY in console)")
    print(f"CSV              : {out_path}")


if __name__ == "__main__":
    main()
