# Watch brief prod-p1-fit

Job: prod-p1-fit   log: lab/jobs/prod-p1-fit/log.txt   expected end: 2026-09-02 04:20Z   expected progress: fitter output or log activity within 20 min
Command: uv run python scripts/lab/jobs.py watch prod-p1-fit --interval 300 --max-wait 10800
On finished: report the finished event; the experiment-engineer will run the aggregate preset-stat checks and create `configs/mode2_v1.yaml`.
On failed/error_pattern: kill if still running (`uv run python scripts/lab/jobs.py kill prod-p1-fit`), retain at most the last 40 log lines in `lab/jobs/prod-p1-fit/status.md`, and return.
On stalled: run `uv run python scripts/lab/jobs.py status prod-p1-fit --gpu`; if GPU utilization is below 5% for two checks, kill and return; otherwise keep watching.
On timeout: append one status line to `lab/STATE.md` and return.
Never relaunch or change flags. The brief's two-crash stop rule remains with the experiment-engineer.
Never kill this runbook compute job for exceeding the five-minute ad-hoc dataset-read cap; duration alone is not a stall.