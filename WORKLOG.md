# WORKLOG

> Collaboration rule: append one concise line for each pushed task.
> Preferred format: `[Agent] YYYY-MM-DD <type>: <summary> -> <commit hash if already known>`.
> If the entry is created in the same commit as the work, the hash may be omitted.
>
> For entries that carry a non-trivial warning or judgment the next agent must see
> (e.g. "this knob was intentionally removed, do not restore"), append indented
> sub-bullets under the main entry:
>
>     - [Agent] 2026-XX-XX shared: ... -> `hash`
>       - Warning / decision / "do not X" note here.
>
> See `AGENTS.md` § "Persisting Decisions and Warnings" for the full three-tier hierarchy.

## 2026-05-15 ~ 16

- [Claude] 2026-05-15 feat: Phase 0.1 perception module -> `989b1ec`
- [Claude] 2026-05-15 shared: README ViZDoom roadmap refresh -> `4b1a8c5`
- [Claude] 2026-05-16 shared: security cleanup for `.backup/` and `.specstory/` -> `7653144`
- [Claude] 2026-05-16 test: CVPerceptor smoke test -> `c0e092d`
- [Codex] 2026-05-16 chore: stop tracking generated artifacts -> `58eecee`
- [Codex] 2026-05-16 refactor: consolidate DeepSeek client -> `c943659`
- [Codex] 2026-05-16 test: isolate legacy RiverGame coverage -> `a07fc68`
- [Codex] 2026-05-16 shared: align README with `deepseek-v4-flash` -> `dd812bb`
- [Codex] 2026-05-16 shared: track `CLAUDE.md`, ignore `.claude/` metadata -> `e096d08`
- [Claude] 2026-05-16 docs: add DeepSeek legacy-model deprecation note -> `0cec04e`
- [Claude] 2026-05-16 test: add DeepSeek API connectivity smoke test -> `51b3596`
- [Codex] 2026-05-16 shared: add repo-level multi-agent protocol and worklog -> `81fe5d8`
- [Claude] 2026-05-16 shared: drop sensei-reporting clauses from project CLAUDE.md -> `bdc394c`
- [Claude] 2026-05-16 shared: codify three-tier decision-persistence hierarchy in AGENTS.md -> `b41671a`
  - When you want to "tell the user something" about another agent's work, ask first
    whether that note should live in commit body / WORKLOG / AGENTS.md instead.
  - User as message forwarder is an anti-pattern; persist decisions where the next
    agent will actually look.
- [Claude] 2026-05-16 shared: clarify architecture progression in project CLAUDE.md -> `42d58ea`
  - "Phase 0 完成定义" no longer says "4 模块目录" (ambiguous: bdd-side vs perception-side).
  - Added explicit per-Phase directory table and 5-point progression principle.
  - Architectural rule: do NOT pre-create empty `actions/` / `agent/` / `oracle/`
    directories; each Phase only creates what it needs. The 6 concept directories
    (perception/env/actions/agent/oracle/experiments) are end-state, not Phase 0.
- [Claude] 2026-05-16 shared: sync research plan into repo as Doc/research-plan.md
  - Doc/research-plan.md is now the authoritative 5-Phase master plan.
  - Obsidian mirror at F:\OBSIDIAN\Obsidian Vault\论文\扩展构想-ViZDoom版.md is kept
    for the user's reading comfort but is **not authoritative**. AI agents must NOT
    edit the Obsidian file; edit Doc/research-plan.md instead, then user manually
    syncs repo → Obsidian.
  - AGENTS.md adds an "Architecture of Record" section requiring agents to consult
    Doc/research-plan.md before any architectural decision.
  - CLAUDE.md planning-docs table updated: Doc/research-plan.md is now first
    (marked ⭐ 权威版), Obsidian entry demoted to "mirror, AI do not edit".

- [Codex] 2026-05-19 shared: migrate Phase 0.2 ViZDoom sandbox into main repo
  - `env/` is project source, not a virtualenv; use `.venv/` for local Python environments.
  - Do not commit `_vizdoom.ini`, screenshots, or trajectory outputs.
  - `experiments/vizdoom/hello_doom.py` is the real ViZDoom wrapper smoke command.

- [Claude] 2026-05-20 shared: lock Phase 1 VLM backend selection and data pipeline design
  - Phase 1.3 backend table finalized to 4: Gemini 2.5 Flash + Qwen3-VL-Plus + Qwen3-VL-Flash + local Qwen2.5-VL 7B (INT4).
  - Removed candidates (do NOT restore without re-discussion): DeepSeek-VL2 (weak vision), GPT-4o / Claude Sonnet (cost), Qwen-VL-Max (superseded by Qwen3-VL-Plus).
  - Phase 1.4 rewritten as 3-stage pipeline: trajectory recorder → keyframe sampling (event_driven / uniform / stratified) → eval script. Do NOT skip the recorder and call VLM live during episode — separating record and eval lets us replay sampling strategies cheaply.
  - Sampling sensitivity analysis is REQUIRED in the output table; single-sampling results are not acceptable for Section IV.A.
  - New success criterion: per-eval cost ≤ ¥100, tracked in `experiments/cost_tracking.md`.

- [Claude] 2026-05-20 shared: add Research Claim section and downscale Phase 1 to spike
  - §0 adds explicit "Research Claim" with claim / metric / baseline triplet. Plan was previously a feature list; now anchored on a falsifiable empirical claim.
  - Main claim: detection_rate(LLM Agent + Goal-level BDD) > detection_rate(hardcoded BDD) on mutation-injected bugs, with failures classifiable into perception / execution / logic.
  - §0 adds "M1 simplification priority": each module has v1 boundary; do NOT chase industrial completeness in any single module.
  - Phase 1.3 adds explicit "Phase usage" table: Phase 1 uses ONE backend (Qwen3-VL-Flash) on ONE scenario as a spike. Full 4-backend comparison moves to Phase 4 to share infrastructure with mutation testing.
  - Phase 1 output / success criteria downscaled: spike target is "link works + first accuracy number", NOT "90% accuracy across 4 backends × 3 scenarios". Per-spike budget ≤ ¥5.
  - Removed from Phase 1 scope (moved to Phase 4): 4-backend comparison, multi-scenario eval, sampling sensitivity, 50-episode large-scale data.

- [Claude] 2026-05-20 shared: Phase 1 Stage 0 + A complete; design doc landed
  - Stage 0 verified by user: `experiments/vizdoom/hello_doom.py` runs successfully on 4090 Laptop, game window visible.
  - Stage A locked in `Doc/phase1-design.md` (TITAN abstract-then-concrete prompt, exact-match metric, no threshold gate).
  - Stage B order: trajectory_recorder -> record_script -> ground_truth -> prompt -> backend -> vlm_perceptor -> eval.

- [Claude] 2026-06-01 feat: Stage B Step 1 (trajectory recorder) + scenario switch to defend_the_center
  - env/trajectory_recorder.py + scripts/record_spike_trajectories.py + tests/test_trajectory_recorder.py landed. Full suite 33 passed, 4 legacy deselected.
  - Real-ViZDoom smoke INVALIDATED the basic.wad assumption: basic ammo starts at 50 (not 26) and only varies in [46,50] because the pistol fires every ~14 tics and the episode ends on the single kill. basic cannot exercise medium/low ammo.
  - DECISION: spike scenario switched basic -> defend_the_center. Constant-ATTACK drives ammo 26->5/7 (high/med/low all covered) + health 100->24. ATTACK is button index 2 in both, so policy [0,0,1] unchanged.
  - Level boundaries updated to ammo[0,26]: high>=18, medium 9-17, low<9.
  - Keyframe sampling changed equidistant -> event-driven (sample on ammo-value change), because ammo only changes every ~14 tics; equidistant would oversample identical ammo values.
  - VLM backend decision: NO dashscope SDK. Use existing openai SDK against DashScope OpenAI-compatible endpoint (base_url https://dashscope.aliyuncs.com/compatible-mode/v1, model qwen3-vl-flash, base64 data-URI images). Reads DASHSCOPE_API_KEY from .env.

- [Claude] 2026-06-01 feat: enable HUD rendering for VLM perception capture
  - Reading the scenario .cfg files revealed render_hud = false on ALL bundled scenarios, so the bottom status bar (ammo/health digits) is not drawn by default. The first batch of recorded trajectories had NO HUD digits -> VLM would have nothing to read. Spike-blocking.
  - Fix: VizDoomEnv gains render_hud: bool = False (default preserves Phase 0.2 RL behaviour). record_spike_trajectories.py passes render_hud=True. Verified: re-recorded frames show the classic Doom status bar; red ammo digit matches game_variables ammo exactly.
  - tests/test_vizdoom_env.py FakeDoomGame gained set_render_hud; added test_render_hud_opt_in_is_forwarded and a default-off assertion. Full suite 34 passed, 4 legacy deselected.
  - Do NOT default render_hud to True: pixel HUD would pollute RL-style observations that read state from game_variables. Perception capture opts in explicitly.

## Current In Progress

- Phase 1 Stage B: Steps 1-2 done (recorder + record script + HUD fix). Next is Step 3 ground_truth.py.

## Next Task

- Phase 1 Stage B Step 3: implement `perception/ground_truth.py` (GroundTruthPerceptor reading ammo/health from game_variables + shared ammo_level boundary function). No API key needed.
- User action: add `DASHSCOPE_API_KEY=sk-...` line to `.env` (real-name account; AI cannot do this). Needed before Step 5 (backend) / Step 8 (eval).

