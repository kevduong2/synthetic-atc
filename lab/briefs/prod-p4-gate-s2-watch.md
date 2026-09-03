# Watch prod-p4-gate-s2

Job: prod-p4-gate-s2   log: lab/jobs/prod-p4-gate-s2/log.txt   expected end: full-scale gate, more than 10 minutes   expected progress: gate tqdm percentage
Command: `uv run python scripts/lab/jobs.py watch prod-p4-gate-s2 --interval 900 --max-wait 900`
On finished: append one status line to `lab/STATE.md`, then return; the director re-invokes the experiment-engineer to record shard-2 tier counts and launch shard 3.
On failed/error_pattern: kill if still running (`uv run python scripts/lab/jobs.py kill prod-p4-gate-s2`), paste the last 40 log lines into `lab/jobs/prod-p4-gate-s2/status.md`, then return.
On stalled: run `uv run python scripts/lab/jobs.py status prod-p4-gate-s2 --gpu`; if GPU utilization is below 5% for two checks, kill and return; otherwise keep watching.
On timeout: append one status line to `lab/STATE.md`, then return for reassignment.
Never relaunch or change flags.