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

- [Claude] 2026-06-01 feat: Stage B Step 3 ground-truth perceptor
  - perception/ground_truth.py: GroundTruthPerceptor reads ammo/health from a recorded game_variables dict (oracle for VLM accuracy). Shared ammo_level() bucketing (high>=18 / medium 9-17 / low<9 / unknown) lives here so GT and VLM scoring use identical boundaries.
  - tests/test_ground_truth.py (10 tests) covers boundaries + dict/live-state extraction + no-raise on missing fields. Full suite 44 passed, 4 legacy deselected.
  - Verified on real recorded data: ammo 26(high)->17(medium)->7(low), health 100->24.

- [Claude] 2026-06-01 feat: Stage B Steps 4-6 VLM perception path + LIVE smoke passed
  - prompts/vizdoom_ammo_v1.txt: TITAN abstract-then-concrete prompt; describes the real Doom status-bar layout (AMMO = far-left red digits, HEALTH = next red number with %).
  - perception/backends/qwen3_vl_flash.py: Qwen3VLFlashBackend over DashScope OpenAI-compatible endpoint using the existing openai SDK (NO dashscope SDK). BackendResponse carries text + latency + token usage. Reuses src.llm.client_helpers (load_project_dotenv, openai_client_kwargs).
  - perception/vlm_perceptor.py: VLMPerceptor (backend-agnostic) owns prompt + image encode (RGB->BGR->PNG->base64 data URI) + tolerant JSON parse. Honours no-raise contract; failures land in raw_response.
  - tests/test_vlm_perceptor.py (10 tests, FakeBackend, no API). Full suite 54 passed, 4 legacy deselected.
  - LIVE smoke (real DashScope call, 3 real frames): VLM read ammo 26/17/7 EXACT-MATCH with GT, levels high/medium/low all correct. ~4-6s/call, ~525 in / ~15 out tokens. Pipeline works end-to-end; accuracy looks high (full number pending Step 8).

- [Claude] 2026-06-01 feat: Stage B Steps 7-8 keyframe sampling + spike eval harness
  - experiments/sampling/ammo_change_keyframes.py: event-driven sampler, one frame per distinct ammo value (avoids oversampling the ~14-tic constant-ammo stretches). 4 unit tests.
  - experiments/eval_perception_spike.py: loads recorded trajectories -> samples keyframes -> VLM vs GroundTruthPerceptor -> CSV (experiments/spike_results_<ts>.csv, gitignored) + printed summary (concrete/abstract accuracy, failures, latency, tokens, est. cost).
  - Full suite 58 passed, 4 legacy deselected. Recorded 5 defend_the_center episodes; full spike eval (~100 VLM calls) was run to produce the first accuracy number (see next entry once written).

- [Claude] 2026-06-01 fix: spike sampler now takes settled (middle) frame per ammo run
  - First spike eval gave concrete accuracy 5/104 = 4.8% but abstract 78.8% with a SYSTEMATIC VLM=GT+1 pattern. Root cause is NOT the VLM: at the tic ammo decrements, the rendered HUD lags game_variables by ~1 tic, so the transition frame shows the old digit. Sampling that edge frame guaranteed an off-by-one.
  - Diagnostic confirmed 100%: same ammo plateau, transition frame VLM=GT+1 (MISS), middle/last frame VLM=GT (OK).
  - Fix: sample_ammo_change_keyframes now returns the middle frame of each constant-ammo run (HUD settled), not the first. Tests updated. Re-running the spike eval to get the corrected number.
  - This is an observation-timing finding (rendered frame must match the state vector); logged in design doc §0 as Section IV.A methodology / threats-to-validity material.

- [Claude] 2026-06-01 docs: Phase 1 spike report — concrete ammo accuracy 100%
  - Corrected eval (middle-frame sampling): concrete 104/104 = 100.0%, 0 failures, ~¥0.01, ~4.9s/frame. VLM reads clear HUD ammo digits essentially perfectly.
  - Abstract (VLM self-reported level) only 79.8%: in all 21 misses vlm_ammo == gt_ammo; the VLM applies its own high/med/low intuition near boundaries even though the prompt states the thresholds. Lesson: derive level in code from the concrete number, do not trust VLM self-classification for clear digits.
  - Doc/phase1-spike-report.md written. DECISION: perception link works; proceed to Phase 2. Prompt/scoring v2 (drop VLM self-level) deferred to when perception is reused in Phase 2.
  - Phase 1 spike is COMPLETE.

- [Claude] 2026-06-01 docs: Phase 2 design doc DRAFT
  - Doc/phase2-design.md drafted: Observer->Observer+Actor, 3-layer action library, goal-level Gherkin + Goal dataclass, minimal while-loop agent with DeepSeek native function calling for the decide step (Tool Use / Function Calling genuinely used, not prompt-hacked). No reflection / LangGraph / planning in v1.
  - Stage 0 (verify env first, Phase 1 lesson) probed button sets: defend_the_center=3 (turn/attack), deadly_corridor=7 (move-rich), deathmatch=20 (only one with SELECT_WEAPON* weapon switching). Doom has NO manual reload, so the research-plan "reload" goal is replaced.
  - OPEN decisions pending user: (1) scenario defend_the_center vs deathmatch; (2) the 3 goals; (3) confirm deepseek-chat function calling for decide.

- [Claude] 2026-06-01 docs: Phase 2 decisions LOCKED + Stage 0 fire mechanics
  - Scenario locked = defend_the_center for v1 (reuse all Phase 1 infra; deathmatch weapon-switching deferred). Decide LLM = deepseek-chat native function calling. v1 perception = ammo only.
  - 3 goals locked: (1) firing consumes ammo; (2) idle does NOT consume ammo (control / the meaningful decision test); (3) repeated firing reduces ammo by ~N (multi-step). Enemy-visibility goal deferred until a VLM enemy_visible field exists.
  - Stage 0 fire-timing measured: a fire composite must advance ~16 tics of ATTACK for exactly 1 ammo decrement (4/8 tics -> delta 0; 14/16/20 -> delta 1). Cause: episode_start_time=10 gun-raise + 4-tic PISTOL1. A naive 4-tic fire would read ammo-unchanged and be MIS-CLASSIFIED as a logic bug -- composite must use 16 tics + settle tics.

- [Claude] 2026-06-01 feat: Phase 2 Stage A action library (primitives + composites)
  - actions/primitives.py: ActionPrimitives over VizDoomEnv. fire_once() holds ATTACK for FIRE_TICS=16 (Stage 0: episode_start_time 10 + PISTOL1 4 => ~14 to first shot; 16 = margin for exactly 1 decrement) then SETTLE_TICS=2 so a pixel perceptor reads the updated HUD. wait()/observe() too.
  - actions/composites.py: TestActions (Layer-3 templates the agent chooses among). fire_and_check_ammo / idle_and_check_ammo return {ammo_before, ammo_after, delta}. Perceptor is INJECTED (GroundTruthPerceptor for deterministic v1, VLMPerceptor optional). run()/list_templates() for the agent loop. __test__=False to avoid pytest collecting the Test*-named production class.
  - tests/test_actions.py (9 tests, FakeActionEnv modeling the 14-tic cadence, no ViZDoom). Full suite 67 passed, 4 legacy deselected.
  - Real-ViZDoom smoke: fire delta=1 (26->25), idle delta=0, fire x3 -> 26->23. All 3 goal mechanics confirmed on the real game.

- [Claude] 2026-06-01 feat: Phase 2 Stage B+C+D — goal-level Gherkin + agent loop + live demo
  - agent/goal.py: Goal dataclass + parse_goals (goal-level Gherkin). Success: line compiles to a sandboxed-eval predicate over the loop's cumulative result (no builtins; Gherkin is trusted input). agent/goals.feature holds the 3 locked goals.
  - agent/loop.py: run_agent_loop (reactive observe-decide-act-check, no reflection). Goal judged on CUMULATIVE result (first ammo_before .. latest ammo_after) so a single goal can span multiple steps. FunctionCallingDecider uses DeepSeek NATIVE tools/tool_calls (injectable client for tests).
  - actions/composites.py: added DESCRIPTIONS (per-template text the LLM sees to choose).
  - tests/test_goal.py (8) + tests/test_agent_loop.py (6, FakeDecider + FakeClient, no API). Full suite 80 passed, 4 legacy deselected.
  - LIVE Stage D demo (experiments/phase2_agent_demo.py, real DeepSeek FC + real ViZDoom): 3/3 goals achieved. Agent chose fire for the fire-goal, IDLE for the idle-goal (the meaningful decision test -- it did NOT fire), and fire x3 for the reduce-by-3 goal. NO hand-written step functions.

- [Claude] 2026-06-02 docs: June roadmap + Phase 3 design draft (internship-portfolio priority)
  - Career-route decision: target Route 3 (AI Testing) primarily, Route 2 (Agent Engineer) as the overlapping bonus; Route 1 (RL Bot) dropped (no RL in this project). The two missing high-frequency JD keywords (Reflection, Agent evaluation) are BOTH delivered by Phase 3 -> Phase 3 is the highest-ROI next step, NOT polishing Phase 2.
  - June priority (user-chosen): COMPLETE STORY + PACKAGING. Phase 4 (mutation/oracle) deferred to July.
  - June plan: W1-2 Phase 3 reflection v1 + eval; W3 packaging (README + architecture diagram + demo GIF + metric tables) = THE job-hunt deliverable; W4 buffer + stretch (LangGraph rewrite, or RAG case-reflection, or a 2-bug mutation teaser).
  - SAFETY NET: if Phase 3 slips past ~June 14, freeze it and package Phase 0-2 (already a complete story). Never end June with undocumented code.
  - Doc/phase3-design.md drafted: keep the 3-type failure classification (PERCEPTION/EXECUTION/LOGIC, thesis novelty); simplify recovery to one round. Key impl design point: add a PER-STEP expected effect so the loop can detect an anomaly and route it to reflection (Phase 2 only checks the cumulative goal). Failure injection (perception perturb / execution skip) in test wrappers only; LOGIC bugs come from Phase 4.
  - LangGraph stays deferred until reflection adds real branching (then it is a genuine resume item, not forced).

- [Claude] 2026-06-02 docs: move LangGraph INTO Phase 3 (reflection control flow)
  - User wants to actually apply LangGraph. Correct call: Phase 3 introduces real branching (check -> success/continue/anomaly; reflect -> perception/execution/logic), which is exactly what a StateGraph models cleanly. Phase 2 was linear so a while loop was right; Phase 3 is not.
  - Decision: implement the reflective agent as a LangGraph StateGraph (agent/graph.py) whose nodes REUSE existing code (FunctionCallingDecider, TestActions, reflection.py). LangGraph orchestrates control flow only; DeepSeek calls stay in reused code; used independently of LangChain.
  - Keep Phase 2 run_agent_loop (while, no reflection) UNCHANGED -> it is the natural no-reflection baseline for eval (baseline while vs proposed graph). No duplicate work, no forced rewrite.
  - Risk control: Phase 3 Stage 0 verifies langgraph installs + a 3-node toy graph runs without LangChain BEFORE business logic; fallback is while+if/else (same behaviour, minus the LangGraph keyword). June safety net unchanged (freeze + package if it slips past ~June 14).

- [Claude] 2026-06-02 note: dual-route job strategy (apply to BOTH Route 2 and Route 3)
  - User prefers Route 2 (Agent Engineer) but the project is rooted in Route 3 (AI Testing); will apply to both. Feasible because the skills overlap heavily -> ONE project, two framings (not two projects). Testing is the application domain; the Agent architecture (Tool Use / Reflection / LangGraph / eval) is the transferable skill.
  - Consequence for the plan: W4 stretch PRIORITY = RAG (case-based reflection: store past failure cases in a vector store, retrieve similar ones during reflection). It fills Route 2's biggest gaps (RAG + Memory) in one move and is a natural extension of reflection, not a forced add. The 2-bug mutation teaser drops below it.
  - README must be route-neutral-but-complete: architecture diagram labels BOTH the Agent layer (Planning/Tool Use/Reflection/LangGraph) AND the testing capabilities (BDD/VLM/bug detection) so both kinds of interviewer find their keywords.
  - Do NOT build extra modules to please both routes; the skeleton already spans both. Core remains: finish Phase 3 + package well.

- [Claude] 2026-06-02 chore: Phase 3 Stage 0 — LangGraph verified
  - langgraph 1.2.2 installed (pulled langchain-core 1.4.0 transitively; we do NOT import langchain directly). requirements.txt adds `langgraph>=1.2,<2`.
  - scripts/smoke_langgraph.py: minimal 3-node StateGraph with a CONDITIONAL edge (router sends value>0 to one branch, else the other). Confirms the exact API agent/graph.py will use in Stage C: `from langgraph.graph import StateGraph, START, END`, add_node(plain fn), add_conditional_edges, compile, invoke. Nodes are plain functions -- no LangChain LLM wrappers.
  - Verified: smoke prints positive(5)/nonpositive(-3); full suite still 80 passed, 4 legacy deselected (new dep does not break collection).

- [Claude] 2026-06-02 feat: Phase 3 Stage A — per-step expected effect + anomaly detection
  - actions/composites.py: TestActions.EXPECTATIONS maps each template to {describe, check}: fire expects delta>=1, idle expects delta==0. TestActions.check_expectation(name, result) returns None if met (or no expectation) else an anomaly dict {action, expected, result} for the reflection layer to consume.
  - Does NOT touch run_agent_loop (Phase 2 while-loop stays the no-reflection baseline). The graph version (Stage C) will call check_expectation in its "check" node to route violations to reflect.
  - tests/test_actions.py +6 (met/violated for both templates, unknown template -> no expectation, delta=None -> anomaly without crash). Full suite 86 passed, 4 legacy deselected.

- [Claude] 2026-06-02 feat: Phase 3 Stage B — reflection (3-type classify + recovery table)
  - agent/reflection.py: FailureType (PERCEPTION/EXECUTION/LOGIC), RECOVERY table (perception->re_observe, execution->retry, logic->report), ReflectionCase (RAG-ready dataclass with to_dict + recovered slot), Reflector.reflect(anomaly, history) -> ReflectionCase via DeepSeek function calling. LLM only CLASSIFIES (+confidence+reasoning); recovery is table lookup (research-plan §5.3). Reflector diagnoses only; executing recovery is Stage C's graph.
  - Doc/phase3-design.md §4.5 added: short-term (history) vs long-term (RAG case library) memory; ReflectionCase is built RAG-ready now so W4 adds retrieval, not a rewrite. Same case log reused as RAG source + eval dataset + thesis case studies.
  - REAL-API FINDING: deepseek-v4-flash runs in "thinking mode" which 400s on a forced tool_choice ({"type":"function",...}). Switched to tool_choice="auto" (proven by the decider) + single tool + explicit instruction. Do NOT use forced tool_choice with this model.
  - tests/test_reflection.py (7, FakeClient, no API). Full suite 93 passed, 4 legacy deselected.
  - LIVE smoke (real DeepSeek): first failure no-history -> EXECUTION/retry (conf 0.6); same anomaly after 2 failed retries in history -> LOGIC/report (conf 0.95). Reflection genuinely uses history and is conservative about "logic" -- direct evidence for the low-false-positive design.

- [Claude] 2026-06-02 feat: Phase 3 Stage C — reflective agent as a LangGraph StateGraph
  - agent/graph.py: build_reflective_app / run_reflective_agent. Nodes (decide/act/reflect/redo/report/mark_success/mark_maxsteps) are plain functions reusing FunctionCallingDecider + TestActions.run/check_expectation + Reflector; deps captured by closure, AgentState carries only data. Conditional edges: route_after_act (success/maxsteps/reflect/continue) and route_recovery (redo for re_observe+retry, report for logic). Goal judged on the cumulative result, same as the while loop, so the two are comparable. recovery_attempts capped by max_recoveries (no infinite loop). Phase 2 run_agent_loop left untouched as the no-reflection baseline.
  - tests/test_graph.py (5, ScriptedActionLib + FakeDecider + FakeReflector, no API/ViZDoom): clean fire/idle -> success (0 cases); execution failure -> reflect -> redo -> recovered=True -> success; logic failure -> bug_reported (not retried); recovery capped -> max_steps. Full suite 98 passed, 4 legacy deselected.
  - LIVE smoke (real ViZDoom + real DeepSeek decider+reflector): 3/3 goals via the reflective graph, 0 cases (clean runs, no anomalies) -- proves the LangGraph wiring runs end-to-end with all real components. Reflection branches are unit-tested; Stage D will trigger them live via injected failures.

- [Claude] 2026-06-02 shared: precision pass on claim + Phase 3 metrics + v2 roadmap
  - research-plan §0 Research Claim PRECISED: (1) mutation testing is an EVALUATION METHOD (controlled ground-truth bugs), NOT the contribution -- contribution is the agent test system; do not say "we do mutation testing research". (2) the 3 failure types are LAYERED: logic-vs-non-logic is the core boundary (decides whether to report a bug); perception-vs-execution is future work because they share an observable + recovery strategy.
  - research-plan §5 + phase3-design §6 metrics made HONEST: report Goal success (baseline while vs proposed graph) + recovery rate + logic-escalation accuracy/误判率; explicitly DO NOT claim high perception-vs-execution classification accuracy (same observable, classified before recovery). Real logic bugs deferred to Phase 4; v1 approximates with a persistent injected failure.
  - Doc/v2-roadmap.md NEW: portability ladder (framework -> MCP interface -> RAG-B knowledge -> multi-agent scale). MCP = wrap the game as an MCP server so swapping games = swapping servers. RAG-B = knowledge-driven test generation (store FPS test knowledge, not just failure cases). ALL July+ / future work; June guardrail = only keep interfaces decoupled + cases RAG-ready (both already true).

## Current In Progress

- Phase 3 Stages A+B+C done. Docs precised (claim + honest metrics + v2 roadmap). Resuming Stage D next.

## Next Task

- Phase 3 Stage D: failure-injection wrappers (perception perturb / execution skip) in experiments/inject.py (test/experiment only, NOT production), to trigger reflection deterministically on the real stack.
- Then Stage E: experiments/eval_reflection.py comparing baseline (while) vs proposed (graph) -> the honest metrics table (phase3-design §6).

## Next Task

- Phase 3 Stage 0: pip install langgraph; confirm a 3-node StateGraph runs and is usable without LangChain. (requirements.txt update will be a shared: commit.)
- Phase 3 Stage A: add a per-step expected effect (e.g. fire_and_check_ammo expects delta>=1) + anomaly detection, usable by both the while baseline and the graph version.

