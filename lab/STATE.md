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
- FID fidelity        STOP B3 COMPLETE (D3 FAIL: packet in lab/reports/prod-fid.md; Kevin owns band-edge decision)
- P3 render s1..s4+noise  PENDING (blocked on D3)
- P4 gate+export      PENDING (blocked on P3)
- CLOSE               PENDING
## Decisions
- D1 (calibration balanced, 6 stations n≥30): PASS 2026-09-02 (min n=85, station_mix exact six)
- D2 (residual selected): PASS 2026-09-02 (selected, step 3500, all 10 evals gates_ok)
- D3 (fidelity: in-band LTAS ≤2 dB, matched KID w/ SE): FAIL 2026-09-02 (in-band max 2.8 dB; B3 packet required)
- D4 (corpus complete: 4×38,944 + 4,800 rows, 3 files): pending
## Running job
none (GPU lock free)
## Next action
Kevin: choose whether a band edge changes; Section 3 remains blocked until fidelity reruns and D3 passes
## Status log (newest first, one line each, written by watchers)
- 2026-09-02 05:28Z prod-fid: B3 packet complete; D3 FAIL, STOP for Kevin; no Section 3 render started
- 2026-09-02 05:27Z prod-fid-kid-on-lp-hp: finished exit 0, WavLM KID 0.003599 +/- 0.000715
- 2026-09-02 05:26Z prod-fid-kid-on-lp-hp: launched and verified running, child PID 40324; expected under 2 min
- 2026-09-02 05:25Z prod-fid-kid-on-lp: finished exit 0, WavLM KID 0.003364 +/- 0.000630
- 2026-09-02 05:24Z prod-fid-kid-on-lp: launched and verified running, child PID 54328; expected under 2 min
- 2026-09-02 05:23Z prod-fid-kid-on: finished exit 0, WavLM KID 0.003043 +/- 0.000525
- 2026-09-02 05:21Z prod-fid-kid-on: launched and verified running, child PID 29164; expected under 2 min
- 2026-09-02 05:21Z prod-fid-kid-off: finished exit 0, WavLM KID 0.003928 +/- 0.000737
- 2026-09-02 05:20Z prod-fid-kid-off: launched and verified running, child PID 39184; expected under 2 min
- 2026-09-02 05:16Z prod-fid-kid-v1: finished, matched KID metrics written to runs\prod_fid\kid\kid_v1_matched.json
- 2026-09-02 05:11Z prod-fid-kid-v1: launched and verified running, child PID 18084; B3 LTAS packet ready, watch pending
- 2026-09-02 05:10Z prod-fid: D3 FAIL, in-band max 2.8 dB and 4 kHz excess +16.4 dB; B3 branch active
- 2026-09-02 05:04Z prod-fid: finished, exit code 0 with calibrated generation at 150/150 and manifest/stats written
- 2026-09-02 05:00Z prod-fid: launched and verified running, child PID 42924; watch pending
- 2026-09-02 04:51Z prod-p2-resid: stalled, no log output for 23 min with GPU 1 at 93%
- 2026-09-02 04:46Z prod-p2-resid: still running, progress 100%|
- 2026-09-02 04:18Z prod-p1-fit: finished, exit code 0
- 2026-09-02 03:57Z prod-p1-fit: still running, progress 1140/1302
- 2026-09-02 03:35Z prod-p1-fit healthy: 929/1302 presets, ~5950 s elapsed, ETA ~40 min
- 2026-09-02 01:55Z prod-p1-fit: relaunched exact command after prior 90-min kill; clarified ad-hoc-read cap does not apply; expect ~04:20Z
- 2026-09-01 P0 PASS: torch 2.14.0+cu126, CUDA True, 780 tests, bench ok, 209,259 wavs, six stations present
