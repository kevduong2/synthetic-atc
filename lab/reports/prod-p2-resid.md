# Report prod-p2-resid                (brief: lab/briefs/prod-p2-resid.md)
## Director summary (<=150 words)
Runbook section 2 FastCUT residual training completed all 5,000 steps as seed-0 job `prod-p2-resid` in 27m36s with exit 0. `selection.status` is `selected`; the selected checkpoint is `runs/fastcut_v1/G_selected.pt` at step 3500, with `kid_mean` 0.0057963989421491655, `kid_se` 0.0003688342183144903, rule `lexicographic_v1.1_fold_paired_tiebreak`, and SHA-256 `1f1d95f32adcfda772e089799e41f832beab7b039523378589c9cf4f972107dd`. D2 **PASS**. All 10 scheduled evaluations scored and had `gates_ok: true`, so pre-authorized branch B2 did not apply and no seed-1 retry ran. `configs/mode2_v1.yaml` now enables this checkpoint and passes strict loading. Fidelity and later phases did not run. Recommended next action: audit this decision packet, then begin `prod-fid` under D3.

## Results

Selection block from `runs/fastcut_v1/validation_report.json`:

```json
{
	"step": 3500,
	"kid_mean": 0.0057963989421491655,
	"kid_se": 0.0003688342183144903,
	"rule": "lexicographic_v1.1_fold_paired_tiebreak",
	"status": "selected",
	"sha256": "1f1d95f32adcfda772e089799e41f832beab7b039523378589c9cf4f972107dd"
}
```

| Field | Result |
|---|---|
| Job | `prod-p2-resid` |
| State | `finished`, exit 0 |
| Started / ended | 2026-09-02 04:25:16Z / 04:52:52Z |
| Elapsed | 1,656 s (27m36s) |
| Training | 5,000 steps, seed 0 |
| Evaluations | 10 at steps 500 through 5,000; all `gates_ok: true` |
| Selected checkpoint | `runs/fastcut_v1/G_selected.pt` (17,625,973 bytes) |
| Config state | `calibrated.residual.enabled: true`; checkpoint resolves |

## Decision rules

D2: `selection.status == "selected"` -> observed `selected` at step 3500 ->
**PASS**.

B2: not entered because D2 passed. No setup-fault rerun and no genuine-failure
seed-1 retry were required.

## Interpretation

At this validation budget, the selected step-3500 residual satisfies the
trainer's frozen fold gates and lexicographic selection rule. This supports
using `G_selected.pt` for the production fidelity phase; it is not yet a D3
fidelity result.

## Artifacts and exact commands

- Validation report: `runs/fastcut_v1/validation_report.json`
- Selected checkpoint: `runs/fastcut_v1/G_selected.pt`
- Job status: `lab/jobs/prod-p2-resid/status.json`
- Job log: `lab/jobs/prod-p2-resid/log.txt`
- Launch: `uv run python scripts/lab/jobs.py launch --gpu --id prod-p2-resid -- uv run python -m atcgen.channel.learned.residual_train --corpus runs/channel_data_v2/corpus.jsonl --split channel_train --val-split channel_val --tts-dir runs/gan_a_base_v1/clean --val-tts-dir runs/gan_val_base_v1/clean --presets runs/channel_data_v2/train/presets.jsonl --noise-bank runs/channel_data_v2/train/noise --out runs/fastcut_v1 --device cuda --steps 5000 --batch-size 12 --crop-frames 128 --lr 2e-4 --base 48 --n-res 6 --scales 1 2 4 --num-patches 256 --nce-mode source+identity --lambda-nce 10.0 --lambda-gan 1.0 --r1-gamma 1.0 --r1-every 16 --ema-decay 0.9995 --residual-scale-max 0.20 --a-renders 4 --eval-every 500 --eval-clips 64 --save-every 500 --seed 0`
- Selection extraction: `uv run python -c "import json; print(json.load(open('runs/fastcut_v1/validation_report.json'))['selection'])"`
- Checkpoint hash: `Get-FileHash runs/fastcut_v1/G_selected.pt -Algorithm SHA256`
- Config validation: `uv run python -c "from pathlib import Path; from atcgen.config import load_config; c=load_config('configs/mode2_v1.yaml'); r=c.calibrated.residual; assert r.enabled and Path(r.checkpoint).is_file()"`

## Not done / deviations

- No seed-1 retry ran because D2 passed on seed 0.
- Fidelity/runbook section 5 and all later production phases did not run.
- No deviations from the frozen runbook section 2 training command.