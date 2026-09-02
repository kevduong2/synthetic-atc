# Lab state            (updated 2026-09-02 05:28Z by experiment-engineer)
Mission: lab/missions/prod-v1.md — V1 production corpus from the FROZEN config (runbook docs/runbook-v1-3080.md)
Clock: started 2026-09-01; overnight autonomous run; per-phase kill rules in the mission file
Standing rule (Kevin, 2026-09-01): never enumerate/read files inside dataset dirs (~209k clips); use one aggregate shell command for counts/sizes; 5-min timeout on ad-hoc data reads ONLY — runbook compute jobs (fit/train/render) run to expected duration under a watcher. Every brief must carry this.
Standing rule (Kevin, 2026-09-02): commit at every phase boundary — the engineer commits its deliverables (report, configs, code) with message `prod-v1: <phase> <verdict>` before its brief is closed. Watchers use chunked `jobs.py watch --max-wait 900` (15 min) calls in a loop, never one unbounded blocking call; lab-assistant watcher runs on Luna Max (Kevin, 2026-09-02).
Autonomy (Kevin, 2026-09-02): overnight run is fully pre-authorized through prod-close per the mission's branches B1–B3 and kill rules; do not wait for Kevin unless a STOP rule fires.
## Phases
- P0 setup/bench      DONE (PASS: CUDA ok, 780 tests, all six stations present; report lab/reports/prod-p0-setup.md)
- P1 calibration      DONE (D1 PASS: KEUG 85, KOJC 143, S50 144, KSLE 145, KIXD 150, KSDL 142; mode2_v1.yaml loads; report lab/reports/prod-p1-calib.md)
- P2 residual         DONE (D2 PASS: G_selected.pt step 3500, kid 0.0058±0.0004; report lab/reports/prod-p2-resid.md)
- FID fidelity        STOP B3 COMPLETE (D3 FAIL: packet in lab/reports/prod-fid.md; audit prod-fid.audit.md PASS-with-notes; Kevin owns band-edge decision)
- P3 render s1..s4+noise  PENDING (blocked on D3)
- P4 gate+export      PENDING (blocked on P3)
- CLOSE               PENDING
## Decisions
- D1 (calibration balanced, 6 stations n≥30): PASS 2026-09-02 (min n=85, station_mix exact six)
- D2 (residual selected): PASS 2026-09-02 (selected, step 3500, all 10 evals gates_ok)
- D3 (fidelity: in-band LTAS ≤2 dB, matched KID w/ SE): FAIL 2026-09-02 — in-band max 2.8 dB (audit-confirmed 2.77); KID 0.003443±0.000731 confirmed. Audit notes: off-vs-on KID claim struck (unpaired 73–77-clip cohorts); reference curve itself sits 7.4 dB off the real cohort in band — vs real, edges are +10.8/+10.3 dB not +16.3/+16.4.
- D4 (corpus complete: 4×38,944 + 4,800 rows, 3 files): pending
## Running job
none (GPU lock free)
## Next action
KEVIN: B3 band-edge decision. Packet: lab/reports/prod-fid.md (4-row filter table, KID±SE, 7-band LTAS); audit: lab/reports/prod-fid.audit.md. LP fixes the 4 kHz edge but every row still exceeds the 2 dB in-band limit (max ~2.8 dB at 1 kHz). Auditor flag worth weighing: the hardcoded LTAS reference is itself 7.4 dB off the real cohort in band — the miss may be the reference, not the render. Any approved change = one chain-step in configs/mode2_v1.yaml, rerun fidelity, then §3. All work committed (182ab36); GPU idle.
## Status log (newest first, one line each, written by watchers)
- 2026-09-02 05:50Z prod-fid audit: PASS-with-notes; KID/LTAS/D2 confirmed; off-vs-on claim struck; real-cohort edge note added
- 2026-09-02 05:28Z prod-fid: B3 packet complete; D3 FAIL, STOP for Kevin; no Section 3 render started
- 2026-09-02 05:16Z prod-fid-kid rows: off .003928±.000737 / on .003043±.000525 / on+LP .003364±.000630 / on+LP+HP .003599±.000715; v1 matched .003443±.000731
- 2026-09-02 05:04Z prod-fid render: finished exit 0, 150/150
- 2026-09-02 04:18Z prod-p1-fit: finished exit 0; prod-p2-resid finished exit 0 (04:52Z)
- 2026-09-01 P0 PASS: torch 2.14.0+cu126, CUDA True, 780 tests, bench ok, 209,259 wavs, six stations present
