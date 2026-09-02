# Brief prod-fid-rerun          to: experiment-engineer   from: lab-director   written 2026-09-02
Goal: land the pre-registered band-edge decision and rerun the fidelity check; apply D3'.
Authority: Kevin delegated the B3 decision to the lab (STATE.md header note, 2026-09-02). The decision is fixed in lab/specs/prod-fid-bandedge.md — follow it exactly; no other frozen value changes.
Inputs: lab/specs/prod-fid-bandedge.md (the decision + D3' + kill rule); lab/reports/prod-fid.md (B3 packet, artifact paths); lab/reports/prod-fid.audit.md (real-cohort LTAS facts); docs/runbook-v1-3080.md §5; .github/skills/paired-analysis/SKILL.md; .github/skills/gpu-jobs/SKILL.md.
Deliverable: lab/reports/prod-fid-rerun.md. Director summary: the chain-step as landed (exact YAML lines), matched KID ± SE, the 7-band LTAS table vs the REAL-cohort reference, in-band (per spec) max gap, D3' PASS/FAIL. Report-not-gate: 100/200/400 Hz and 4 kHz gaps vs both references.
Steps:
1. Add the one chain-step from the spec to configs/mode2_v1.yaml (LP per spec). Config-load smoke.
2. Rerun the §5 fidelity render with the updated config (GPU via jobs.py, job id prod-fid-rerun; if > 10 min, launch and return "launched, watch pending").
3. Matched KID + LTAS per the spec: LTAS gate computed against the real-cohort reference (measurement fix; 2.0 dB limit unchanged), per-spec in-band bands gated, everything else reported.
4. Apply D3' exactly as worded in the spec. PASS → report + commit `prod-v1: fid-rerun PASS`, do not start §3 (director briefs it). FAIL → the spec's kill rule: STOP the mission, commit `prod-v1: fid-rerun FAIL-STOP`, no alternates.
Budget: 60 min own activity per invocation. Kill: crash twice on same step → stop and report.
Dataset-read guardrail (standing): no listing/reading inside dataset dirs; aggregate commands only; 5-min cap on ad-hoc reads only.
Do not: touch data/real/kixd/kixd_locked_day.csv; alter any frozen value beyond the spec's chain-step; start §3.
If your reply is lost: the report file is the result.
