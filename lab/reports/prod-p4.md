# Report prod-p4                (brief: lab/briefs/prod-p4.md)

## Director summary

P4 D4 PASS. All four 38,944-row speech shards gated successfully: 19,694 gold, 29,976 silver, 43,765 adversarial, and 62,341 rejected (93,435 non-rejected of 155,776). The 4,800 noise-only rows required no separate gate. The runbook export accepted all 160,576 pre-gate manifest rows and wrote `data/corpus/V1.0.0/corpus_train.csv` (157,462 rows), `corpus_test.csv` (3,114 rows), and `manifest.json`. Validation found exactly those three files, both CSV SHA-256 digests match the manifest, and the row counts sum to 160,576. No runbook deviations; proceed to prod-close.

## Results

| Speech input | Total | Gold | Silver | Adversarial | Non-rejected | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| `runs/train_v1_s1` | 38,944 | 4,830 | 7,569 | 11,106 | 23,505 | 15,439 |
| `runs/train_v1_s2` | 38,944 | 4,935 | 7,422 | 10,934 | 23,291 | 15,653 |
| `runs/train_v1_s3` | 38,944 | 4,964 | 7,443 | 10,929 | 23,336 | 15,608 |
| `runs/train_v1_s4` | 38,944 | 4,965 | 7,542 | 10,796 | 23,303 | 15,641 |
| **Speech total** | **155,776** | **19,694** | **29,976** | **43,765** | **93,435** | **62,341** |

| Export artifact | Data rows | Validation |
|---|---:|---|
| `data/corpus/V1.0.0/corpus_train.csv` | 157,462 | SHA-256 matches manifest |
| `data/corpus/V1.0.0/corpus_test.csv` | 3,114 | SHA-256 matches manifest |
| `data/corpus/V1.0.0/manifest.json` | 160,576 source rows | present; 4,800 noise-only rows in train |
| **CSV total** | **160,576** | **matches pre-gate total: 4 x 38,944 + 4,800** |

## Decision rules

D4: 160,576 manifest rows before gating and export writes `corpus_train.csv`, `corpus_test.csv`, and `manifest.json` -> observed 160,576 source rows and all three files with validated counts and digests -> **PASS**.

## Artifacts and exact commands

```powershell
uv run pytest -q
uv run python scripts/lab/jobs.py lock status
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s1 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s1 --device cuda
uv run python scripts/lab/jobs.py status prod-p4-gate-s1 --tail 20 --gpu
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s2 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s2 --device cuda
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s3 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s3 --device cuda
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s4 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s4 --device cuda
uv run python scripts/export_corpus_csv.py --dataset runs/train_v1_s1 --dataset runs/train_v1_s2 --dataset runs/train_v1_s3 --dataset runs/train_v1_s4 --dataset runs/train_v1_noise --out data/corpus/V1.0.0 --version V1.0.0 --include-noise-only --reason "V1 production render, mode2_v1, fastcut_v1"
```

Shard 1 artifacts: `runs/train_v1_s1/manifest_gated.jsonl`, `runs/train_v1_s1/gate_stats.json`.
Shard 2 artifacts: `runs/train_v1_s2/manifest_gated.jsonl`, `runs/train_v1_s2/gate_stats.json`.
Shard 3 artifacts: `runs/train_v1_s3/manifest_gated.jsonl`, `runs/train_v1_s3/gate_stats.json`.
Shard 4 artifacts: `runs/train_v1_s4/manifest_gated.jsonl`, `runs/train_v1_s4/gate_stats.json`.
Export artifacts: `data/corpus/V1.0.0/corpus_train.csv`, `data/corpus/V1.0.0/corpus_test.csv`, `data/corpus/V1.0.0/manifest.json` (gitignored by the repository's `data/` rule).

## Not done / deviations

- None. The noise-only set went directly to export as specified by runbook section 4.