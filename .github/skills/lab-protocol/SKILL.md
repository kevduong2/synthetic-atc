---
name: lab-protocol
description: How the atc-gan agent lab coordinates. The lab/ file layout (STATE.md board, missions, briefs, specs, reports, jobs), the brief, spec and report templates every agent writes, the experiment-discipline rules (power gate before any search, at least 4 paired seeds, bounded WER, single-question fixed arms, fresh --out on any metric change, locked data read once, matched KID only) and claim-discipline wording. Read at the start of a lab session and before writing any brief, spec or report.
---

# Lab protocol

## 1. Files, not messages

A chat reply can be lost; a file cannot (one overnight session lost 5 of 5
approvals to a one-way message failure; agents that wrote to agreed paths lost
nothing). Every brief, spec, result and decision is a file under `lab/` before
it is a message. Briefs pre-authorize decisions so a lost reply cannot stall
the lab. Any factual claim that gates a decision is verified by the
results-auditor before it is acted on.

## 2. Layout

```
lab/
  STATE.md            the board: mission, phase, running job, next action (≤60 lines; director owns it)
  missions/<m>.md     goal, constraints, budget, phases with clocks, decision rules, kill rules
  briefs/<id>.md      one task for one agent (§4); <id>-watch.md for a monitor brief (§7)
  specs/<id>.md       pre-registered experiment design (§5)
  reports/<id>.md     results with a Director summary on top (§6); <id>.audit.md from the auditor
  jobs/<id>/          gitignored: cmd.txt, log.txt, status.json, status.md   (scripts/lab/jobs.py)
  GPU_LOCK            gitignored: holder of the single GPU stream
```

Ids are `<mission>-<phase>-<short>`, e.g. `win2-p1-gate`; one id is reused by
the brief, the job, the spec and the report.

## 3. STATE.md

```
# Lab state            (updated <UTC time> by <agent>)
Mission: lab/missions/<m>.md — <one-line goal>
Clock: started <UTC>; hard stop <UTC>
## Phases
- P0 bench            DONE  C = 4.7 min/cell  (lab/reports/win2-p0-bench.md)
- P1 gate             RUNNING job win2-p1-gate, expect done ~14:10Z, watcher: lab-assistant
- P2A channel arms    BLOCKED on D1
## Decisions
- D1 (channel visible?): pending
## Running job
win2-p1-gate | uv run python scripts/rl_power_check.py --out runs/win2_gate ... | budget 6C
## Next action
director: read lab/reports/win2-p1-gate.md summary when the watcher posts; decide D1
## Status log (newest first, one line each, written by watchers)
- 13:42Z win2-p1-gate: running, cell 3/6, log growing, gpu 97%
```

## 4. Brief template (`lab/briefs/<id>.md`)

```
# Brief <id>                 to: <agent>   from: <agent>   written <UTC>
Goal: <one sentence; the question or deliverable>
Inputs: <files to read, nothing else>
Deliverable: lab/reports/<id>.md using the report template; also <artifacts>
Budget: <C units or minutes>; hard stop <UTC>. When the clock crosses it, stop and report what completed.
Pre-authorized decisions: <e.g. "if a cell OOMs: halve --ft-batch once, then stop"; "if D1 fails: do not start P2A">
Kill criteria: <what ends the task early>
Do not: <frozen values, locked data, second GPU job, ...>
If your reply is lost: the report file is the result; the director reads it, not the chat.
```

## 5. Spec template (`lab/specs/<id>.md`, written BEFORE anything runs)

```
# Spec <id>
Question: <one sentence>
Hypothesis: if <X>, then <arm> beats base by ≥ <effect> on the paired statistic, ≥3/4 seeds agreeing.
Decision this changes: <what happens on pass / on fail>
Arms: base (control) + one single-knob arm per question; each arm's config diff in one line.
Seeds: <n ≥ 4 paired for a claim; 2 only for a gate>
Budget: <cells × C>; clocks; checkpoint at which the run stops regardless.
Decision rule: |paired mean| ≥ 2 × paired SE AND ≥ <k>/<n> seeds one direction [AND fidelity guard: matched KID not worse than base by > 1 SE].
Kill criterion: <e.g. first cell > 12 min → drop arm X>
Artifacts that prove it ran: runs/<id>/trials/*/dev_rows.jsonl, config.resolved.yaml per trial, the dev-composition line from the log.
Ideas (unscheduled): <one line each, why parked>
```

## 6. Report template (`lab/reports/<id>.md`)

```
# Report <id>                (spec: lab/specs/<id>.md; brief: lab/briefs/<id>.md)
## Director summary (≤150 words)
<what ran, C measured, the decision-rule outcome stated as PASS/FAIL against the pre-registered numbers,
 what did not run and why, the single recommended next action>
## Results
<paired table: arm | n | mean | SE | t | df | direction | per-seed diffs>   (from scripts/analysis/paired_report.py)
<fidelity table if any: matched KID ± SE, LTAS at 100/200/400/1k/2k/3k/4k>
## Decision rules
D<k>: rule as pre-registered → observed numbers → PASS/FAIL
## Interpretation            (researcher; claim-discipline wording, §9)
## Artifacts and exact commands
## Not done / deviations
```

## 7. Watch brief (`lab/briefs/<id>-watch.md`, for the lab-assistant)

```
Job: <id>   log: lab/jobs/<id>/log.txt   expected end: <UTC>   expected progress: "cell k/6" every ~C min
Command: uv run python scripts/lab/jobs.py watch <id> --interval 300 --max-wait 1500
On finished: run <post-command or "write status line only">, then return.
On failed/error_pattern: kill if still running (`jobs.py kill <id>`), paste last 40 log lines into lab/jobs/<id>/status.md, return.
On stalled: `jobs.py status <id> --gpu`; if GPU util < 5% for two checks, kill and return; else keep watching.
On timeout: append one status line to lab/STATE.md and return (the caller re-invokes you).
Never relaunch, never change flags.
```

## 8. Discipline (every rule was bought with wasted GPU hours)

1. **Gate before you search.** No optimization until a power check proves the
   reward can see the thing: known-good arm, known-bad arm, base, ≥2 paired
   seeds; known-bad must separate from base by ≥2× the paired SE. A channel
   wrecked to 0–6 dB SNR with 40% dropouts was invisible to the reward (t=0.10).
2. **Paired seeds or nothing.** A seed fixes generation draw and fine-tune
   order, so nuisance cancels in paired differences. Report per-seed diffs,
   direction count, paired t. ≥4 seeds for a claim (2-seed "3.5×" became 1.5×
   at 4); 8–10 to resolve ~1 WER point. Never quote unpaired separation.
3. **Bounded WER is the decision metric** (per-row errors capped at reference
   length; whisper-tiny loops). Raw counts stay in `dev_rows.jsonl`. Changing
   a metric, model id or dev slice mid-run needs a fresh `--out`.
4. **Fixed interpretable arms beat CEM at ≤30 trials.** One question per arm,
   base as shared control, check additivity when splitting an effect.
5. **Locked data is read once, ever.** `data/real/kixd/kixd_locked_day.csv`
   (day 2025-08-08, 337 rows) is for the final trained model only. Dev is
   `kixd_dev.csv` (200 rows, day 2025-08-07). EU rows are monitor-only.
6. **KID only on matched audio** (energy-trim + RMS-normalize both sides,
   fixed 1,000-clip reference subset); raw KID is 35–40% padding/level.
7. **Timebox and pre-register.** Budget, decision rule and kill criterion are
   written before the run. At a checkpoint follow the written branch; never
   extend because it feels close. Write results as each stage lands.
8. **One GPU stream.** Two concurrent GPU jobs slowed both by 40%.
9. **Frozen means frozen.** V1 values in `docs/runbook-v1-3080.md` §5 are not
   changed inside a mission; a win becomes a V1.1 spec, not a diff.

## 9. Claim-discipline wording

Development evidence: "supports", "consistent with", "at this budget". Never
"proves" or "confirms" from dev slices. A null on a blind reward says the
reward cannot select that knob, not that the knob does not matter. Confirmatory
language waits for the single locked-day read of the final model. A clean null
on every phase is a valid, reportable outcome.

## 10. Known traps

- `--dev-indices` defaults to `0:200`; on a mixed dev file that selects one
  source. Every run prints its dev composition; read it.
- `runs/power_check_kixd/summary.json` is mixed-metric; recompute from rows.
- `export_corpus_csv.py --version` is strictly `V<int>.<int>.<int>`.
- `build_paired_views.py` writes 24 kHz audio; `channel_fit` needs 16 kHz.
- Preset `passband_hz` looks degenerate; the real EQ is `band_edges_hz` /
  `band_gains_db`; rendered audio is full-band.
- `sequential:` text sources refuse `--n-samples` beyond file length.
- Voice/speed per clip live under the manifest row's nested `gen` key.
- Baseline caches key off corpus; verify they key off the model too before
  sharing `--out` across models.
- `rl_power_check.py`'s `degraded` arm edits `channel.chain`, so it only
  resolves on a Mode 1 (procedural) base config.
