# Brief prod-p2-resid          to: experiment-engineer   from: lab-director   written 2026-09-02
Goal: complete runbook §2 (train + validate the FastCUT residual on the recalibrated presets) and report the validation_report.json selection block.
Inputs: docs/runbook-v1-3080.md §2 (follow exactly); lab/missions/prod-v1.md (D2, branch B2); configs/mode2_v1.yaml (from P1, D1 PASS); .github/skills/gpu-jobs/SKILL.md; lab/reports/prod-p1-calib.md (facts only).
Deliverable: lab/reports/prod-p2-resid.md using the report template. Director summary must state: selection.status, the selected checkpoint path, key selection metrics (kid_mean etc.) from validation_report.json, and D2 PASS/FAIL.
Steps: runbook §2 exactly. Launch the training via scripts/lab/jobs.py launch --gpu as job id prod-p2-resid (one GPU stream). This is a long runbook compute job — it runs to its expected duration.
Watching: after launching, hand off — do NOT block on the job yourself. Verify the job started (one status call), write a launch note in the report file (partial report is fine), and return with the Director summary stating "launched, watch pending". The director runs the watcher separately. When re-invoked after the job finishes, complete the analysis and the final report.
Budget: 60 min wall clock of your own activity (the GPU job itself is not counted). Exceeded → stop and report.
Pre-authorized decisions (branch B2, from the mission — follow without asking):
- selection.status != "selected": diagnose from evaluations[]. Every kid_mean null → eval never scored, a setup fault: fix and rerun the SAME seed. gates_ok false throughout → genuine failure: ONE retry as prod-p2-resid-s1 with --seed 1 --out runs/fastcut_v1_s1. A second failure → STOP; never fall back to G_ema.pt or residual.enabled: false.
Kill criteria: crash twice on the same step → stop and report.
Dataset-read guardrail (standing rule): never list/glob/read individual files inside dataset directories (~209k clips); aggregate shell commands only; 5-min cap on ad-hoc reads ONLY — never kill the training job for duration.
Commit rule (standing): at phase close, commit deliverables (report, any config/code) with message `prod-v1: p2-resid <verdict>`.
Do not: start fidelity/§3; touch data/real/kixd/kixd_locked_day.csv; change frozen values.
If your reply is lost: the report file is the result; the director reads it, not the chat.
