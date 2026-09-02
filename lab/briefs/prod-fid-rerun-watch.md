# Watch brief prod-fid-rerun

Job: prod-fid-rerun   log: lab/jobs/prod-fid-rerun/log.txt   expected end: 2026-09-02 15:55Z   expected progress: render progress through 150 clips
Command: uv run python scripts/lab/jobs.py watch prod-fid-rerun --interval 300 --max-wait 900
On finished: write status line only; experiment-engineer runs matched-set, KID, LTAS, and D3' analysis, then return.
On failed/error_pattern: kill if still running (`uv run python scripts/lab/jobs.py kill prod-fid-rerun`), paste the last 40 log lines into lab/jobs/prod-fid-rerun/status.md, then return.
On stalled: `uv run python scripts/lab/jobs.py status prod-fid-rerun --gpu`; if GPU util < 5% for two checks, kill and return; otherwise keep watching in another chunk.
On timeout: append one status line to lab/STATE.md and return; the caller re-invokes the watcher.
Never relaunch or change flags. Do not run analysis or start Section 3.
