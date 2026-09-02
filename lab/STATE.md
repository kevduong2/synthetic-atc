# Lab state            (updated 2026-09-02 15:18Z by experiment-engineer)
Mission: lab/missions/prod-v1.md — V1 production corpus from the FROZEN config (runbook docs/runbook-v1-3080.md)
Clock: started 2026-09-01; overnight autonomous run; per-phase kill rules in the mission file
Standing rule (Kevin, 2026-09-01): never enumerate/read files inside dataset dirs (~209k clips); use one aggregate shell command for counts/sizes; 5-min timeout on ad-hoc data reads ONLY — runbook compute jobs (fit/train/render) run to expected duration under a watcher. Every brief must carry this.
Standing rule (Kevin, 2026-09-02): commit at every phase boundary — the engineer commits its deliverables (report, configs, code) with message `prod-v1: <phase> <verdict>` before its brief is closed. Watchers use chunked `jobs.py watch --max-wait 900` (15 min) calls in a loop, never one unbounded blocking call; lab-assistant watcher runs on Luna Max (Kevin, 2026-09-02).
Autonomy (Kevin, 2026-09-02): overnight run is fully pre-authorized through prod-close per the mission's branches B1–B3 and kill rules; do not wait for Kevin unless a STOP rule fires.
B3 delegation (Kevin, 2026-09-02 morning): Kevin defers the band-edge decision to the director's best judgment — pick from the B3 packet, land it as one chain-step in configs/mode2_v1.yaml, rerun fidelity, then proceed to §3–§4 and close. The lab, not Kevin, now owns this call.
Recipe-correction authorization (Kevin, 2026-09-02 afternoon): "just fix it, then run the clips" — Kevin authorizes ONE pre-registered frozen-value correction targeting the 1 kHz LTAS deficit (spec lab/specs/prod-v1-recipe-fix.md), fidelity re-gate, then §3–§4 and close WITHOUT further check-ins. Pre-registered fallback: if the fix fails D3'' twice, render as-is with the deficit documented (Kevin wants the corpus either way). Fix work delegated on Opus 5.
## Phases
- P0 setup/bench      DONE (PASS: CUDA ok, 780 tests, all six stations present; report lab/reports/prod-p0-setup.md)
- P1 calibration      DONE (D1 PASS: KEUG 85, KOJC 143, S50 144, KSLE 145, KIXD 150, KSDL 142; mode2_v1.yaml loads; report lab/reports/prod-p1-calib.md)
- P2 residual         DONE (D2 PASS: G_selected.pt step 3500, kid 0.0058±0.0004; report lab/reports/prod-p2-resid.md)
- FID fidelity        DONE (D3'' PASS after the peaking_eq fix: real-cohort in-band max 1.43 dB, matched WavLM KID 0.004331+/-0.000938 <= 0.005728 guardrail; report lab/reports/prod-fix-1khz.md. Superseded D3' FAIL-STOP: lab/reports/prod-fid-rerun.md)
- P3 render s1..s4+noise  NOT RUN (awaiting director brief)
- P4 gate+export      NOT RUN
- CLOSE               PENDING
## Decisions
- D1 (calibration balanced, 6 stations n≥30): PASS 2026-09-02 (min n=85, station_mix exact six)
- D2 (residual selected): PASS 2026-09-02 (selected, step 3500, all 10 evals gates_ok)
- D3 (fidelity: in-band LTAS ≤2 dB, matched KID w/ SE): FAIL 2026-09-02 — in-band max 2.8 dB (audit-confirmed 2.77); KID 0.003443±0.000731 confirmed. Audit notes: off-vs-on KID claim struck (unpaired 73–77-clip cohorts); reference curve itself sits 7.4 dB off the real cohort in band — vs real, edges are +10.8/+10.3 dB not +16.3/+16.4.
- D3' (real-cohort LTAS ≤2.0 dB at 1/2/3 kHz + matched KID with SE): FAIL-STOP 2026-09-02 — max gap 4.69 dB at 1 kHz (audit-exact); KID 0.004134±0.000797 valid. Audit PASS-with-notes: deficit pre-existing, masked by defective hardcoded reference (+2.77 − 7.37 = −4.60 on the first render too); LP blameless in-band, fixed 4 kHz (−13 dB paired).
- D3'' (LTAS ≤2.0 dB at 1/2/3 kHz + matched KID with SE + KID ≤0.005728 guardrail): PASS 2026-09-02 on attempt 1 — peaking_eq f0 1100 / +7.0 dB / Q 1.7 after the LP; gaps −0.53/−0.34/−1.43 dB (in-band max 1.43); WavLM KID 0.004331±0.000938 (150/997), CLAP 0.001063±0.000133. Attempt 2 and the fallback revert not used; configs/mode2_v1.yaml is in its final render state. Report not yet audited.
## Running job
none (GPU lock free)
## Next action
director: audit lab/reports/prod-fix-1khz.md, then brief runbook §3 renders (s1..s4 + noise) from the corrected configs/mode2_v1.yaml
## Status log (newest first, one line each, written by watchers)
- 2026-09-02 16:30Z prod-fix-1khz: D3'' PASS; in-band max 1.43 dB; matched WavLM KID 0.004331+/-0.000938; peaking_eq step stays in configs/mode2_v1.yaml; §3 not started
- 2026-09-02 16:16Z prod-fix-1khz-kid: launched --gpu, finished exit 0 in 137 s (wavlm 55.1 s, clap 78.1 s)
- 2026-09-02 16:13Z prod-fix-1khz: launched --gpu (150-clip seed-0 render, runs/prod_fid_d3pp_a1), finished exit 0 in 61 s, QC 150/150 kept
- 2026-09-02 15:18Z prod-fid-rerun: D3' FAIL-STOP; real-cohort max 1/2/3 kHz gap 4.69 dB; matched WavLM KID 0.004134+/-0.000797
- 2026-09-02 15:11Z prod-fid-rerun: finished, exit code 0 after 150 clips rendered
- 2026-09-02 15:07Z prod-fid-rerun: running, child PID 53520; preflight 782 passed, 3 skipped; watch pending
- 2026-09-02 05:50Z prod-fid audit: PASS-with-notes; KID/LTAS/D2 confirmed; off-vs-on claim struck; real-cohort edge note added
- 2026-09-02 05:28Z prod-fid: B3 packet complete; D3 FAIL, STOP for Kevin; no Section 3 render started
- 2026-09-02 05:16Z prod-fid-kid rows: off .003928±.000737 / on .003043±.000525 / on+LP .003364±.000630 / on+LP+HP .003599±.000715; v1 matched .003443±.000731
- 2026-09-02 05:04Z prod-fid render: finished exit 0, 150/150
- 2026-09-02 04:18Z prod-p1-fit: finished exit 0; prod-p2-resid finished exit 0 (04:52Z)
- 2026-09-01 P0 PASS: torch 2.14.0+cu126, CUDA True, 780 tests, bench ok, 209,259 wavs, six stations present
