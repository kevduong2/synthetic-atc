# Report prod-p4b-tiers                (brief: lab/briefs/prod-p4b-tiers.md)

## Director summary

P4b PASS. The V1.0.0 export now ships `gate_tier` on every row: `gold`,
`silver`, `adversarial`, or `rejected` for speech, and `noise` for noise-only
rows. `corpus_train.csv` has 157,462 rows: 19,278 gold, 29,366 silver, 42,897
adversarial, 61,121 rejected, and 4,800 noise. `corpus_test.csv` has 3,114
rows: 416 gold, 610 silver, 868 adversarial, 1,220 rejected, and 0 noise.
Totals reconcile exactly to 19,694/29,976/43,765/62,341 speech tiers plus
4,800 noise (160,576 rows); no row has a null tier. `manifest.json` was
regenerated with the new CSV hashes, and both SHA-256 digests validate. The
full suite passes (787 passed, 3 skipped). No audio was rendered or re-gated.

## Results

| Split | Gold | Silver | Adversarial | Rejected | Noise | Null | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 19,278 | 29,366 | 42,897 | 61,121 | 4,800 | 0 | 157,462 |
| Test | 416 | 610 | 868 | 1,220 | 0 | 0 | 3,114 |
| **Total** | **19,694** | **29,976** | **43,765** | **62,341** | **4,800** | **0** | **160,576** |

| Artifact | SHA-256 | Manifest match |
|---|---|---|
| `data/corpus/V1.0.0/corpus_train.csv` | `441752f8588c758c38c25872979764d18ba43540ce7179aaca8812fe27880b17` | yes |
| `data/corpus/V1.0.0/corpus_test.csv` | `b21cb472a3ece5bfb9ca4e7808bfd7e117820eb72be636d02ba035299ce9da05` | yes |

## Decision rules

The join must have no unmatched clip IDs, totals must remain 157,462 train +
3,114 test, speech tiers must reconcile to the four gate reports, all rows must
have a tier, and manifest hashes must validate -> observed zero unmatched IDs,
the unchanged 160,576-row split, exact tier reconciliation, zero nulls, and two
matching hashes -> **PASS**.

## Artifacts and exact commands

Changed code and documentation:

- `scripts/export_corpus_csv.py`
- `tests/test_corpus_scripts.py`
- `docs/runbook-v1-3080.md`

Regenerated gitignored corpus artifacts:

- `data/corpus/V1.0.0/corpus_train.csv`
- `data/corpus/V1.0.0/corpus_test.csv`
- `data/corpus/V1.0.0/manifest.json`

```powershell
git log --oneline -5
git status --short
uv run pytest -q tests/test_corpus_scripts.py
uv run pytest -q
uv run python scripts/export_corpus_csv.py --dataset runs/train_v1_s1 --dataset runs/train_v1_s2 --dataset runs/train_v1_s3 --dataset runs/train_v1_s4 --dataset runs/train_v1_noise --out data/corpus/V1.0.0 --version V1.0.0 --include-noise-only --reason "V1 production render, mode2_v1, fastcut_v1"
$manifest = Get-Content data/corpus/V1.0.0/manifest.json -Raw | ConvertFrom-Json; foreach ($name in @('train','test')) { $path = "data/corpus/V1.0.0/corpus_$name.csv"; $rows = @(Import-Csv $path); $counts = $rows | Group-Object gate_tier | Sort-Object Name; [pscustomobject]@{split=$name; rows=$rows.Count; null_gate_tier=@($rows | Where-Object { [string]::IsNullOrWhiteSpace($_.gate_tier) }).Count; tiers=(($counts | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ', '); sha256=(Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant(); expected_sha256=$manifest.sha256."${name}_csv"} | Format-List }
```

## Not done / deviations

- Corpus CSVs and `manifest.json` remain gitignored and are not part of the
  commit; they were regenerated in place as required.
- No gate, render, or GPU command ran. No file under `data/real/` was touched.