# Lab state            (updated 2026-09-03 11:22Z by experiment-engineer)
Mission: lab/missions/prod-v1.md — V1 production corpus from the FROZEN config (runbook docs/runbook-v1-3080.md)
Clock: started 2026-09-01; overnight autonomous run; per-phase kill rules in the mission file
Standing rule (Kevin, 2026-09-01): never enumerate/read files inside dataset dirs (~209k clips); use one aggregate shell command for counts/sizes; 5-min timeout on ad-hoc data reads ONLY — runbook compute jobs (fit/train/render) run to expected duration under a watcher. Every brief must carry this.
Standing rule (Kevin, 2026-09-02): commit at every phase boundary — the engineer commits its deliverables (report, configs, code) with message `prod-v1: <phase> <verdict>` before its brief is closed. Watch cadence: 15-minute checks — loop `watch --interval 900 --max-wait 900` calls (these survive a session; the old 240 s guidance is obsolete), one STATE line per watch assignment; lab-assistant watcher runs on Luna (Kevin, 2026-09-02).
Autonomy (Kevin, 2026-09-02): overnight run is fully pre-authorized through prod-close per the mission's branches B1–B3 and kill rules; do not wait for Kevin unless a STOP rule fires.
B3 delegation (Kevin, 2026-09-02 morning): Kevin defers the band-edge decision to the director's best judgment — pick from the B3 packet, land it as one chain-step in configs/mode2_v1.yaml, rerun fidelity, then proceed to §3–§4 and close. The lab, not Kevin, now owns this call.
Recipe-correction authorization (Kevin, 2026-09-02 afternoon): "just fix it, then run the clips" — Kevin authorizes ONE pre-registered frozen-value correction targeting the 1 kHz LTAS deficit (spec lab/specs/prod-v1-recipe-fix.md), fidelity re-gate, then §3–§4 and close WITHOUT further check-ins. Pre-registered fallback: if the fix fails D3'' twice, render as-is with the deficit documented (Kevin wants the corpus either way). Fix work delegated on Opus 5.
## Phases
- P0 setup/bench      DONE (PASS: CUDA ok, 780 tests, all six stations present; report lab/reports/prod-p0-setup.md)
- P1 calibration      DONE (D1 PASS: KEUG 85, KOJC 143, S50 144, KSLE 145, KIXD 150, KSDL 142; mode2_v1.yaml loads; report lab/reports/prod-p1-calib.md)
- P2 residual         DONE (D2 PASS: G_selected.pt step 3500, kid 0.0058±0.0004; report lab/reports/prod-p2-resid.md)
- FID fidelity        DONE (D3'' PASS after the peaking_eq fix: real-cohort in-band max 1.43 dB, matched WavLM KID 0.004331+/-0.000938 <= 0.005728 guardrail; report lab/reports/prod-fix-1khz.md. Superseded D3' FAIL-STOP: lab/reports/prod-fid-rerun.md)
- P3 render s1..s4+noise  DONE (all five outputs exact aggregate counts + stats.json; report lab/reports/prod-p3-render.md)
- P4 gate+export      DONE (D4 PASS: all four shard gates exit 0; export wrote 157,462 train + 3,114 test rows and manifest.json; report lab/reports/prod-p4.md)
- CLOSE               PENDING
## Decisions
- D1 (calibration balanced, 6 stations n≥30): PASS 2026-09-02 (min n=85, station_mix exact six)
- D2 (residual selected): PASS 2026-09-02 (selected, step 3500, all 10 evals gates_ok)
- D3 (fidelity: in-band LTAS ≤2 dB, matched KID w/ SE): FAIL 2026-09-02 — in-band max 2.8 dB (audit-confirmed 2.77); KID 0.003443±0.000731 confirmed. Audit notes: off-vs-on KID claim struck (unpaired 73–77-clip cohorts); reference curve itself sits 7.4 dB off the real cohort in band — vs real, edges are +10.8/+10.3 dB not +16.3/+16.4.
- D3' (real-cohort LTAS ≤2.0 dB at 1/2/3 kHz + matched KID with SE): FAIL-STOP 2026-09-02 — max gap 4.69 dB at 1 kHz (audit-exact); KID 0.004134±0.000797 valid. Audit PASS-with-notes: deficit pre-existing, masked by defective hardcoded reference (+2.77 − 7.37 = −4.60 on the first render too); LP blameless in-band, fixed 4 kHz (−13 dB paired).
- D3'' (LTAS ≤2.0 dB at 1/2/3 kHz + matched KID with SE + KID ≤0.005728 guardrail): PASS 2026-09-02 on attempt 1 — peaking_eq f0 1100 / +7.0 dB / Q 1.7 after the LP; gaps −0.53/−0.34/−1.43 dB (in-band max 1.43); WavLM KID 0.004331±0.000938 (150/997), CLAP 0.001063±0.000133. Attempt 2 and the fallback revert not used; configs/mode2_v1.yaml is in its final render state. Audit PASS-with-notes (lab/reports/prod-fix-1khz.audit.md) — render cleared.
## Running job
none
## Next action
director: start prod-close transcription, audit, and sign-off
## Status log (newest first, one line each, written by watchers)
- 2026-09-03 11:22Z prod-p4: D4 PASS; shard 4 finished exit 0 (gold 4,965, silver 7,542, adversarial 10,796, rejected 15,641); export wrote 157,462 train + 3,114 test rows and manifest.json, hashes verified
- 2026-09-03 10:05Z prod-p4-gate-s4: launched, running under GPU lock, child PID 64388; shard 3 finished exit 0 with 38,944 clips gated; watch pending
- 2026-09-03 10:01Z prod-p4-gate-s3: PASS; finished exit 0 after 11,110 seconds; gold 4,964, silver 7,443, adversarial 10,929, rejected 15,608; noise set goes direct to export per runbook §4
- 2026-09-03 06:56Z prod-p4-gate-s3: launched, running under GPU lock, child PID 34824; shard 2 finished exit 0 with 38,944 clips gated; watch pending
- 2026-09-03 06:43Z prod-p4-gate-s2: PASS; finished exit 0 after 4,843 seconds; gold 4,935, silver 7,422, adversarial 10,934, rejected 15,653
- 2026-09-03 05:22Z prod-p4-gate-s2: launched, running under GPU lock, child PID 66072; shard 1 finished exit 0 with 38,944 clips gated; watch pending
- 2026-09-03 04:05Z prod-p4-gate-s1: launched, running under GPU lock, child PID 50432; gating 0/4,868 batches; watch pending
- 2026-09-03 03:53Z prod-p4: preflight 785 passed, 3 skipped; pre-gate manifests total 160,576 rows; GPU lock free; launching prod-p4-gate-s1
- 2026-09-03 03:49Z prod-p3-noise: PASS; finished exit 0 after 158 seconds; aggregate output count 4,800 WAV files; stats.json present
- 2026-09-03 03:46Z prod-p3-noise: launched, running under GPU lock, child PID 18480; watch pending
- 2026-09-03 03:44Z prod-p3-s4: PASS; finished exit 0 after 9,976 seconds; aggregate output count 38,944 WAV files; stats.json present
- 2026-09-03 00:54Z prod-p3-s4: launched, running under GPU lock, child PID 62020; watch pending
- 2026-09-02 00:53Z prod-p3-s3: finished, generating (calibrated) completed at 100% (38944/38944), exit code 0
- 2026-09-03 00:05Z prod-p3-s3: still running, generating (calibrated) at 71% (27577/38944)
- 2026-09-02 23:15Z prod-p3-s3: still running, generating (calibrated) reached 39% (15203/38944)
- 2026-09-02 22:22Z prod-p3-s3: still running, last observed generating (calibrated) at 0% (0/38944)
- 2026-09-02 22:14Z prod-p3-s3: launched, running under GPU lock, child PID 13408; watch pending
- 2026-09-02 22:11Z prod-p3-s2: finished, exit 0
- 2026-09-02 21:47Z prod-p3-s2: timeout, still running at 87% (33931/38944)
- 2026-09-02 20:17Z prod-p3-s2: still running, generating (calibrated) reached 37% (14322/38944)
- 2026-09-02 19:15Z prod-p3-s2: launched, running under GPU lock, child PID 21864; watch pending
- 2026-09-02 19:11Z prod-p3-s1: finished, generating (calibrated) completed at 100% (38944/38944); exit code 0
- 2026-09-02 18:18Z prod-p3-s1: still running, generating (calibrated) reached 67% (26142/38944)
- 2026-09-02 17:17Z prod-p3-s1: still running, generating (calibrated) 11067/38944 (28%)
- 2026-09-02 16:56Z prod-p3-s1: still running, generating (calibrated) 5626/38944 (14%)
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
