# Brief prod-p3-render          to: experiment-engineer   from: lab-director   written 2026-09-02
Goal: runbook §3 — render the four production shards + the noise set from the final configs/mode2_v1.yaml (frozen recipe + LP@3.8k + peaking_eq, D3''-cleared).
Inputs: docs/runbook-v1-3080.md §3 (exact commands, seeds, --out paths); lab/missions/prod-v1.md (shard rules); .github/skills/gpu-jobs/SKILL.md.
Deliverable per invocation: launch note appended to lab/reports/prod-p3-render.md. Final deliverable (after noise): Director summary with, per shard/noise dir: clip count, stats.json present, QC kept/total, elapsed.
Protocol (one shard per invocation): launch the NEXT pending job in order s1 → s2 → s3 → s4 → noise via jobs.py launch --gpu (job ids prod-p3-s1..s4, prod-p3-noise), verify started with one status call, append the launch note, return "launched: <id>, watch pending". The director watches; you are re-invoked when it ends.
On a finished job (when re-invoked): verify its output dir (aggregate count + stats.json only — no file listing), note PASS in the report, then launch the next.
Pre-authorized (mission): a failed shard is re-rendered ONCE with the same seed and --out (job id <id>-r1). A second failure of the same shard → STOP the mission and report.
Resumability: per-script resumability applies — a re-render resumes, never restarts from zero; do not delete partial outputs.
Budget: 20 min own activity per invocation. Kill: launch fails twice → stop.
Dataset-read guardrail (standing): aggregate commands only; no listing inside output/dataset dirs; 5-min cap on ad-hoc reads only — render jobs run to duration.
Commit rule: after noise verifies, commit report + any state with `prod-v1: p3 renders complete`.
Do not: touch data/real/kixd/kixd_locked_day.csv; modify configs; start §4.
If your reply is lost: the report file is the result.
