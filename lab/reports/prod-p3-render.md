# Report prod-p3-render                (brief: lab/briefs/prod-p3-render.md)

## Director summary

PASS: `prod-p3-s1` through `prod-p3-s4` each finished exit 0 and each output contains exactly 38,944 WAV files with `stats.json` present. `prod-p3-noise` also finished exit 0 with exactly 4,800 WAV files and `stats.json` present. All five runbook section 3 render outputs are complete; proceed to section 4 gate and export.

## Launch log

- 2026-09-03 03:49Z `prod-p3-noise`: PASS; finished exit 0 after 158 seconds; aggregate output count 4,800 WAV files; `stats.json` present.
- 2026-09-03 03:46Z `prod-p3-noise`: launched with the exact 4,800-row runbook section 3 command; status `running`, child PID 18480, GPU lock held, log empty at the 5-second startup check; watch pending.
- 2026-09-03 03:44Z `prod-p3-s4`: PASS; finished exit 0 after 9,976 seconds; aggregate output count 38,944 WAV files; `stats.json` present.
- 2026-09-03 00:54Z `prod-p3-s4`: launched with the exact runbook section 3 command; status `running`, child PID 62020, GPU lock held, log empty at the 4-second startup check; watch pending.
- 2026-09-03 00:50Z `prod-p3-s3`: PASS; finished exit 0 after 9,348 seconds; aggregate output count 38,944 WAV files; `stats.json` present.
- 2026-09-02 22:14Z `prod-p3-s3`: launched with the exact runbook section 3 command; status `running`, child PID 13408, GPU activity present, log empty at the 7-second startup check; watch pending.
- 2026-09-02 22:11Z `prod-p3-s2`: PASS; finished exit 0 after 10,573 seconds; aggregate output count 38,944 WAV files; `stats.json` present.
- 2026-09-02 19:14Z `prod-p3-s2`: launched with the exact runbook section 3 command; status `running`, child PID 21864, GPU activity present, log empty at the 2-second startup check; watch pending.
- 2026-09-02 19:09Z `prod-p3-s1`: PASS; finished exit 0 after 9,275 seconds; aggregate output count 38,944 WAV files; `stats.json` present.
- 2026-09-02 16:35Z `prod-p3-s1`: launched with the exact runbook section 3 command; status `running`, child PID 22204, GPU activity present, log empty at the 3-second startup check; watch pending.

## Preflight

- Test suite: 785 passed, 3 skipped.
- GPU lock: free before launch; held by `prod-p3-s1` after launch.
- Inputs: `scenes_v2.0.1_2view.shard1of4.jsonl` through `shard4of4.jsonl`, 38,944 rows each.

## Artifacts and exact commands

```powershell
uv run python scripts/lab/shard_text.py data/text/scenes_v2.0.1_2view.jsonl --n 4
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s1 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s1 --seed 7 --text sequential:data/text/scenes_v2.0.1_2view.shard1of4.jsonl --set dataset.noise_only_frac=0
uv run python scripts/lab/jobs.py status prod-p3-s1 --tail 20 --gpu
(Get-ChildItem -LiteralPath runs/train_v1_s1 -Recurse -File -Filter '*.wav' | Measure-Object).Count
Test-Path -LiteralPath runs/train_v1_s1/stats.json -PathType Leaf
uv run python scripts/lab/jobs.py lock status
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s2 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s2 --seed 17 --text sequential:data/text/scenes_v2.0.1_2view.shard2of4.jsonl --set dataset.noise_only_frac=0
uv run python scripts/lab/jobs.py status prod-p3-s2 --tail 20 --gpu
(Get-ChildItem -LiteralPath runs/train_v1_s2 -Recurse -File -Filter '*.wav' | Measure-Object).Count
Test-Path -LiteralPath runs/train_v1_s2/stats.json -PathType Leaf
uv run python scripts/lab/jobs.py lock status
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s3 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s3 --seed 27 --text sequential:data/text/scenes_v2.0.1_2view.shard3of4.jsonl --set dataset.noise_only_frac=0
uv run python scripts/lab/jobs.py status prod-p3-s3 --tail 20 --gpu
(Get-ChildItem -LiteralPath runs/train_v1_s3 -Recurse -File -Filter '*.wav' | Measure-Object).Count
Test-Path -LiteralPath runs/train_v1_s3/stats.json -PathType Leaf
uv run python scripts/lab/jobs.py lock status
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s4 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s4 --seed 37 --text sequential:data/text/scenes_v2.0.1_2view.shard4of4.jsonl --set dataset.noise_only_frac=0
uv run python scripts/lab/jobs.py status prod-p3-s4 --tail 20 --gpu
(Get-ChildItem -LiteralPath runs/train_v1_s4 -Recurse -File -Filter '*.wav' | Measure-Object).Count
Test-Path -LiteralPath runs/train_v1_s4/stats.json -PathType Leaf
uv run python scripts/lab/jobs.py lock status
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-noise -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 4800 --out runs/train_v1_noise --seed 8 --set dataset.noise_only_frac=1.0
uv run python scripts/lab/jobs.py status prod-p3-noise --tail 20 --gpu
```

## Not done / deviations

- None. Gate and export are tracked in `lab/reports/prod-p4.md`.