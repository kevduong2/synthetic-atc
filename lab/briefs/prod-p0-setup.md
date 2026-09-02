# Brief prod-p0-setup          to: experiment-engineer   from: lab-director   written 2026-09-01
Goal: complete runbook §0 (environment, tests, bench, clip extraction) and report CUDA status, test count, bench numbers, per-station clip counts.
Inputs: docs/runbook-v1-3080.md §0; lab/missions/prod-v1.md (constraints only); .github/skills/gpu-jobs/SKILL.md.
Deliverable: lab/reports/prod-p0-setup.md using the report template. Director summary must state: torch version + cuda.is_available(), pytest pass/fail count, bench s/render and s/step for tts/gan/sft sections, per-station wav counts for KEUG, KOJC, S50, KSLE, KIXD, KSDL (and any others found, incl. unknown), and PASS/FAIL on "all six deployed stations present".
Steps (runbook §0 exactly):
1. §0.2: uv sync if needed; verify torch/cuda/libsndfile; run `uv run pytest -q` (expect ~780 passed); run the bench command to runs/bench/cuda.json.
2. §0.3: if the clip archive is not yet extracted at reference-data-for-v1-run/airport_clips_v2, test + extract it per the runbook; copy data/real/calibration/*.wav in if not already present; compute per-station counts with the runbook's PowerShell one-liner.
3. If real-audio manifests came from the Mac, rewrite absolute paths once with scripts/lab/relocate.py (gpu-jobs skill).
Budget: 60 min. When exceeded, stop and report what completed.
Pre-authorized decisions: bench needs at least one section flag — use the runbook's exact flag set. If the archive tests corrupt or any of the six deployed stations is absent → STOP, write the report with FAIL and the missing station(s); do not improvise data.
Kill criteria: pytest failures that indicate a broken environment (import errors, CUDA absent) → stop and report; a handful of flaky test failures unrelated to env is reportable, not fatal — list them.
Do not: launch §1+ steps; touch data/real/kixd/kixd_locked_day.csv; change any config.
Dataset-read guardrail (standing lab rule, applies to every step here):
- NEVER list, glob, or read individual files inside dataset directories (reference-data-for-v1-run/airport_clips_v2 holds ~200k files; enumerating it via file tools hangs). No list_dir/file_search/read_file on dataset trees.
- All counting/sizing goes through one shell command (the runbook's PowerShell one-liner, `Get-ChildItem -File | Measure-Object`, etc.) — aggregate output only.
- Before touching any data directory, check its scale first with a single counting command; if a command that reads data runs > 5 min, kill it, note it in the report, and use a cheaper aggregate instead.
If your reply is lost: the report file is the result; the director reads it, not the chat.
