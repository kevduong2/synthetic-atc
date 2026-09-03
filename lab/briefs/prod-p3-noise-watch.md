# Watch prod-p3-noise

Job: prod-p3-noise   log: lab/jobs/prod-p3-noise/log.txt   expected end: ~2026-09-03 04:30Z   expected progress: generating 4,800 noise-only clips
Command: `uv run python scripts/lab/jobs.py watch prod-p3-noise --interval 900 --max-wait 900`
On finished: append one status line to `lab/STATE.md`, then return.
On failed/error_pattern: kill if still running (`uv run python scripts/lab/jobs.py kill prod-p3-noise`), paste the last 40 log lines into `lab/jobs/prod-p3-noise/status.md`, then return.
On stalled: run `uv run python scripts/lab/jobs.py status prod-p3-noise --gpu`; if GPU utilization is below 5% for two checks, kill and return; otherwise keep watching.
On timeout: append one status line to `lab/STATE.md`, then return for reassignment.
Never relaunch or change flags.