# Mission prod-v1 — V1 production run on the RTX 3080

Goal: produce `data/corpus/V1.0.0/` (corpus_train.csv + corpus_test.csv +
manifest.json) from the FROZEN V1 config, recalibrated on the full
multi-airport clip set. Runbook: `docs/runbook-v1-3080.md` — this file adds
lab structure only and contradicts nothing in it.

Kevin's prompt to the `lab-director` (VS Code, Copilot):
`Run lab/missions/prod-v1.md as mission prod-v1.` Pre-launch checklist for
Kevin: `HUMAN.md`.

Hard constraints:
- Frozen values (runbook §5) do not change. The only config the mission
  creates is `configs/mode2_v1.yaml` per runbook §1c (calibration paths,
  explicit `station_mix`, residual checkpoint). No arms, no searches.
- One GPU stream; every GPU step through `scripts/lab/jobs.py launch --gpu`.
- `data/real/kixd/kixd_locked_day.csv` is not read.
- Budget: not clock-boxed like win2, but each phase has a kill rule below.
  Rendering is ~4–6 h on the 3080 after the bench; plan the day around §3.

Phases (ids are the brief/job/report ids):

| id | runbook | agent | deliverable | kill / stop rule |
|---|---|---|---|---|
| prod-p0-setup | §0 | experiment-engineer | `lab/reports/prod-p0-setup.md`: CUDA True, test count, bench numbers, per-station clip counts | any expected station absent → stop, report to Kevin (data-handoff.md) |
| prod-p1-calib | §1 | experiment-engineer | presets_stats per-station table; `configs/mode2_v1.yaml` loads | a deployed station with < 30 kept presets → **branch B1**: keep it in `station_mix`, proceed, flag "thin calibration: <station> n=<k>" in the report and the addendum |
| prod-p2-resid | §2 | experiment-engineer, lab-assistant watches | `validation_report.json` selection block in the report | `selection.status != "selected"` → **branch B2**: results-auditor diagnoses from `evaluations[]` (every `kid_mean` null = eval never scored, a setup fault: fix and rerun the same seed; `gates_ok` false throughout = genuine failure: one retry as `prod-p2-resid-s1` with `--seed 1 --out runs/fastcut_v1_s1`). Second failure → STOP; never fall back to `G_ema.pt` or to `residual.enabled: false` |
| prod-fid | §5 fidelity | experiment-engineer, results-auditor recomputes | matched KID + LTAS table (100/200/400/1k/2k/3k/4k) | in-band gap > 2 dB, or the 4 kHz excess still > +8 dB with the residual on → **branch B3**: build the filter table (`scripts/analysis/filter_variants.py` on the fidelity render, then matched KID + LTAS for off / on / on+LP / on+LP+HP), write it to the report, then STOP for Kevin: a band edge is a frozen-config change and stays his |
| prod-p3-s1..s4, prod-p3-noise | §3 | experiment-engineer launches sequentially; lab-assistant watches each | four shard dirs + noise dir, `stats.json` present in each | a failed shard is re-rendered once with the same seed/out; twice → stop |
| prod-p4 | §4 | experiment-engineer | `data/corpus/V1.0.0/` + gate tier table per shard | export refuses → report, do not hand-edit CSVs |
| prod-close | — | lab-assistant transcribes, results-auditor audits, director signs | dated addendum in `docs/results.md`: station counts, presets per station, selection block, fidelity table, gate yields, corpus row counts, locked day untouched | — |

Pre-registered branches (ticked by Kevin 2026-09-01; the engineer follows
these without asking):
- **B1 thin station:** proceed with the station in `station_mix`, flagged. Not
  dropped: an airport rendered with other towers' channels is the bias the
  balancing exists to avoid, and a thin preset pool is still that receiver.
- **B2 residual not selected:** diagnose, one retry with `--seed 1`, then stop.
  No `G_ema.pt`, no residual-off render: both would ship a corpus the frozen
  recipe does not describe.
- **B3 fidelity miss:** produce the decision packet (4-row filter table with
  KID ± SE and LTAS at 100/200/400/1k/2k/3k/4k), then stop. If Kevin approves
  a band edge, it lands as one chain-step change in `configs/mode2_v1.yaml`,
  the fidelity check reruns, and only then does §3 start.
- **Shards:** a failed shard is re-rendered once, same seed and `--out`; a
  second failure stops the mission.

Decision rules:
- D1 (calibration balanced?): each of the six deployed stations (KEUG, KOJC, S50, KSLE, KIXD, KSDL) has n ≥ 30 in `presets_stats.json`, and `station_mix` lists exactly those → PASS. Non-deployed receivers (SEATTLE_CENTER, KSLE_GROUND) are not gated.
- D2 (residual selected?): `validation_report.json` `selection.status == "selected"` → PASS.
- D3 (fidelity): in-band LTAS gap ≤ 2 dB AND matched KID reported with SE → PASS. The 4 kHz / 100 Hz gaps are reported, not gated (owner decision in runbook §5).
- D4 (corpus complete): 4 × 38,944 + 4,800 manifest rows before gating; export writes all three files.

Deliverables live in the repo: reports under `lab/reports/`, the addendum in
`docs/results.md`, `configs/mode2_v1.yaml` committed, `runs/` left intact.
