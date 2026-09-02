# Brief prod-fix-1khz          to: experiment-engineer   from: lab-director   written 2026-09-02
Goal: implement lab/specs/prod-v1-recipe-fix.md exactly — the peaking_eq primitive + one chain step, fidelity rerun, D3'' — and land the mission in one of its pre-registered branches. Kevin authorized this frozen-value correction and the fallback; NO further check-ins are needed in any branch.
Inputs: lab/specs/prod-v1-recipe-fix.md (the contract — follow its numbers, D3'', resize formula and fallback verbatim); lab/reports/prod-fid-rerun.md + .audit.md (measurement procedure to reproduce: real-cohort LTAS vs runs/calib_v2/clips, matched KID vs the fixed 997-clip seed-0 reference, trim −35 dB, −26 dB RMS both sides); .github/skills/generator-config/SKILL.md; .github/skills/gpu-jobs/SKILL.md; .github/skills/paired-analysis/SKILL.md.
Deliverable: lab/reports/prod-fix-1khz.md. Director summary: the primitive/step as landed, attempt-1 LTAS (all seven bands vs real cohort) and KID ± SE, D3'' verdict; if attempt 2 ran, its resize Δ and results; the final branch taken (PASS / resized-PASS / fallback-revert) and explicit confirmation the config is in its final render state.
Steps:
1. Implement `peaking_eq` as a new chain primitive (inert unless named), thin code + a unit test per repo conventions. Full test suite must pass.
2. Add the spec's chain step to configs/mode2_v1.yaml after the LP; config-load smoke.
3. Fidelity render (job prod-fix-1khz via jobs.py --gpu; if > 10 min, launch and return "launched, watch pending" — the director watches). Same 150-clip seed-0 procedure as prod-fid-rerun.
4. LTAS + matched KID per the spec's D3'' (including the KID ≤ 0.005728 guardrail). Apply D3''.
5. Branches (pre-authorized): PASS → done. FAIL → attempt 2 with the spec's resize formula (skip if 2 kHz alone binds), one more render prod-fix-1khz-a2, re-gate. Still FAIL → revert the peaking_eq step from the config (keep LP), confirm config-load, document the deficit — fallback branch.
6. In every branch: commit `prod-v1: fix-1khz <PASS|PASS-a2|fallback>` (code, config, report). Do NOT start §3 — the director briefs the renders separately.
Budget: 90 min own activity per invocation (GPU time excluded). Kill: crash twice on the same step → stop and report (only genuine stop left).
Dataset-read guardrail (standing): no listing/reading inside dataset dirs; aggregate commands only; 5-min cap on ad-hoc reads only.
Do not: touch data/real/kixd/kixd_locked_day.csv; change anything beyond the spec's primitive + step (and its revert).
If your reply is lost: the report file is the result.
