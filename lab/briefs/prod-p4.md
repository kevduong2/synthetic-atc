# Brief prod-p4          to: experiment-engineer   from: lab-director   written 2026-09-03
Goal: runbook §4 — gate the five render dirs and export data/corpus/V1.0.0/ (corpus_train.csv + corpus_test.csv + manifest.json). Apply D4.
Inputs: docs/runbook-v1-3080.md §4 (exact commands); lab/missions/prod-v1.md (D4, P4 row); .github/skills/gpu-jobs/SKILL.md; lab/reports/prod-p3-render.md (shard verifications).
First: close P3 — verify prod-p3-noise output (aggregate count 4,800 + stats.json), note PASS in lab/reports/prod-p3-render.md, commit `prod-v1: p3 renders complete`.
Deliverable: lab/reports/prod-p4.md. Director summary: gate tier table per shard (kept/total by tier), corpus row counts per file, manifest.json present, D4 PASS/FAIL (pre-gate rows = 4×38,944 + 4,800; export writes all three files).
Steps: runbook §4 exactly. Gating jobs via jobs.py launch --gpu (job ids prod-p4-gate-*); if any job > 10 min, launch and return "launched, watch pending" — the director watches. Export is CPU and runs inline.
Pre-authorized (mission): export refuses → report, do NOT hand-edit CSVs; that is the only stop.
Budget: 45 min own activity per invocation. Kill: crash twice on the same step → stop.
Dataset-read guardrail (standing): aggregate commands only; never list/read individual files inside render/output dirs; 5-min cap on ad-hoc reads only.
Commit rule: on D4, commit `prod-v1: p4 <verdict>` (report + any state; corpus CSVs/manifest only if the runbook says they are tracked).
Do not: touch data/real/kixd/kixd_locked_day.csv; modify configs; re-render anything.
If your reply is lost: the report file is the result.
