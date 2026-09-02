# Brief prod-p1-calib          to: experiment-engineer   from: lab-director   written 2026-09-01
Goal: complete runbook §1 (recalibrate presets on the full multi-airport clip set; create configs/mode2_v1.yaml per §1c) and report the per-station presets_stats table.
Inputs: docs/runbook-v1-3080.md §1 (follow exactly); lab/missions/prod-v1.md (constraints, D1, branch B1); .github/skills/generator-config/SKILL.md; .github/skills/gpu-jobs/SKILL.md; lab/reports/prod-p0-setup.md (environment facts only).
Deliverable: lab/reports/prod-p1-calib.md using the report template. Director summary must state: per-station kept-preset counts from presets_stats.json for KEUG, KOJC, S50, KSLE, KIXD, KSDL (plus any non-deployed receivers, unmarked), whether configs/mode2_v1.yaml loads (a config-load smoke, e.g. resolved-config dump), and D1 PASS/FAIL/B1 per the mission rule: all six deployed stations n ≥ 30 and station_mix lists exactly those six.
Steps: runbook §1a–§1c exactly. Any GPU step goes through scripts/lab/jobs.py launch --gpu (one stream).
Budget: 90 min wall clock. When exceeded, stop and report what completed.
Pre-authorized decisions:
- Branch B1 (a deployed station with < 30 kept presets): keep it in station_mix, proceed, flag "thin calibration: <station> n=<k>" prominently in the Director summary. Do not drop the station.
- Frozen values (runbook §5) do not change; mode2_v1.yaml only adds calibration paths, explicit station_mix, and the residual checkpoint path per §1c.
Kill criteria: calibration crashes twice on the same step → stop and report; a station absent from the clip set entirely → stop (that contradicts P0 and needs the director).
Dataset-read guardrail (standing lab rule — scope clarified 2026-09-01 after first attempt):
- NEVER list, glob, or read individual files inside dataset directories (reference-data-for-v1-run/airport_clips_v2 holds ~209k files). No list_dir/file_search/read_file on dataset trees.
- All counting/sizing goes through one aggregate shell command; check scale before touching any data directory; kill an *ad-hoc read/enumeration command* that runs > 5 min and note it in the report.
- The 5-min cap applies ONLY to ad-hoc reads/listings. It does NOT apply to runbook compute steps (calibration fit, training, rendering) launched via scripts/lab/jobs.py — those run to their runbook-expected durations and are watched, never killed for taking > 5 min. The first attempt killed the fit job under this misreading; do not repeat that.
Resume note: runs/calib_v2 (corpus_stats.json), runs/channel_data_v2 (split_stats.json) and both probe dirs already exist from the first attempt; per-script resumability applies — do not redo completed §1 steps, continue from the preset fit.
Resume note 2 (2026-09-02): job prod-p1-fit FINISHED, exit 0 (fit on 1302 presets complete). Do NOT relaunch the fit. Remaining work: verify/emit presets_stats.json, build the per-station table, create configs/mode2_v1.yaml per §1c, config-load smoke, write the report, then commit deliverables with message `prod-v1: p1-calib <verdict>` (standing rule: commit at every phase boundary).
Do not: start §2+; touch data/real/kixd/kixd_locked_day.csv; edit any config other than creating configs/mode2_v1.yaml.
If your reply is lost: the report file is the result; the director reads it, not the chat.
