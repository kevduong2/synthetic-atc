# Watch brief prod-fid

Job: prod-fid   log: lab/jobs/prod-fid/log.txt   expected end: ~05:20Z   expected progress: render percentage or completed-clip output
Command: `uv run python scripts/lab/jobs.py watch prod-fid --interval 900 --max-wait 900`
On finished: write the status line, then return; the director re-invokes the experiment-engineer for matched-set preparation.
On failed/error_pattern: kill if still running, paste the last 40 log lines into `lab/jobs/prod-fid/status.md`, then return. Do not relaunch.
On stalled: run `uv run python scripts/lab/jobs.py status prod-fid --gpu`; if GPU utilization is below 5% for two consecutive checks, kill and return; otherwise keep watching.
On timeout: append one status line to `lab/STATE.md` and return so the director can re-invoke the watcher.
Never relaunch, change flags, start an evaluation, or start another GPU job.