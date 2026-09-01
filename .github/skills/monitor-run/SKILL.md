---
name: monitor-run
description: Watching a running atc-gan job as the lab assistant. The single jobs.py watch command that blocks until an event, what each event means (finished, failed, error_pattern, stalled, timeout, lost, killed), the only actions a watcher may take, the ten-line status format, and when to escalate. Use when briefed to monitor a job.
---

# Monitor a run

You watch by running one blocking command, not by polling in chat turns:

```
uv run python scripts/lab/jobs.py watch <id> --interval 300 --max-wait 1500
```

It checks the job every `--interval` seconds and returns ONE JSON event with
`event`, `state`, `exit_code`, `elapsed_s`, `log_idle_s`, `last_progress`,
`matched` (error lines), `tail` (last 20 log lines), and `gpu_snapshot`
(nvidia-smi utilisation and memory). Keep `--max-wait` at or below the shell
tool's timeout; on `timeout` you return to the caller, who re-invokes you.
Per-job progress patterns can be passed with `--progress-regex`; the default
catches `step k/N`, `trial k/N`, `cell k/N`, `epoch k/N` and tqdm percentages.

## Events and the only allowed responses

| event | meaning | you do |
|---|---|---|
| `finished` | exit 0 | run the post-step named in the watch brief (if any), write status, return |
| `failed` | non-zero exit | paste the last 40 log lines into `lab/jobs/<id>/status.md`, write status, return |
| `error_pattern` | still running but the log shows Traceback / OOM / nan / Killed | `jobs.py kill <id>` only if the brief pre-authorizes it, paste matched lines, return |
| `stalled` | log silent for `--stall-min` (default 20) while running | `jobs.py status <id> --gpu`; if GPU util < 5% on two consecutive checks and the brief allows, kill; else watch again |
| `timeout` | `--max-wait` elapsed, job healthy | append one line to `lab/STATE.md` status log, return |
| `lost` | wrapper and child vanished without an exit code | treat as failed; note "lost" |
| `killed` | someone killed it | write status, return |

Exit codes: 0 finished, 1 failed/error/lost/killed, 3 stalled, 4 timeout.

## Status format (`lab/jobs/<id>/status.md` gets the auto-line; you add ≤10 lines)

```
<UTC> <id>: <event> | progress <last_progress> | elapsed <m> min | gpu <util>% <mem> MB
outcome: <one line>
next: <what the caller should do, or "nothing">
<up to 5 log lines if they explain the outcome>
```

Then append the first line to the `## Status log` in `lab/STATE.md` (newest
first) and return the same lines as your reply.

## Never

- Relaunch, change flags, edit configs or code, or start another GPU job.
- Decide anything the watch brief did not pre-authorize. If in doubt write
  `ESCALATE: <one line>` at the top of `status.md` and return.
- Read the whole log. `jobs.py tail <id> -n 40` is the most you need.
