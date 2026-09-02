# Report prod-fid                (brief: lab/briefs/prod-fid.md)
## Director summary
The frozen V1 150-clip render has matched WavLM KID 0.003443 +/- 0.000731 (CLAP 0.001003 +/- 0.000118). LTAS gaps at 100/200/400/1k/2k/3k/4k Hz are +16.3/+1.8/-1.0/+2.8/-1.7/+0.1/+16.4 dB: in-band max 2.8 dB and 4 kHz excess +16.4 dB. D3 FAILS the <=2 dB rule. B3's LP changes residual-on 4 kHz from +9.6 to -3.4 dB with WavLM KID 0.003364 +/- 0.000630; adding HP changes 100 Hz from +13.8 to -6.6 dB with KID 0.003599 +/- 0.000715. Every B3 row still exceeds the in-band limit. No Section 3 render started. STOP for Kevin's band-edge decision.

## Results
Primary fidelity result (gaps are relative to the fixed real reference curve):

| Render | n | WavLM matched KID | +/- | CLAP matched KID | +/- | 100 Hz | 200 Hz | 400 Hz | 1 kHz | 2 kHz | 3 kHz | 4 kHz | max abs 1-3 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 150 | 0.003443 | 0.000731 | 0.001003 | 0.000118 | +16.3 | +1.8 | -1.0 | +2.8 | -1.7 | +0.1 | +16.4 | 2.8 |

B3 filter packet (WavLM is the primary matched KID; CLAP is diagnostic):

| Variant | n | WavLM matched KID | +/- | CLAP matched KID | +/- | 100 Hz | 200 Hz | 400 Hz | 1 kHz | 2 kHz | 3 kHz | 4 kHz | max abs 1-3 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 77 | 0.003928 | 0.000737 | 0.001079 | 0.000113 | +17.3 | +3.0 | -1.6 | +2.3 | -0.8 | +1.3 | +18.6 | 2.3 |
| on | 73 | 0.003043 | 0.000525 | 0.000977 | 0.000097 | +13.8 | -0.9 | -1.2 | +2.4 | -3.7 | -2.7 | +9.6 | 3.7 |
| on+LP | 73 | 0.003364 | 0.000630 | 0.000922 | 0.000099 | +13.8 | -0.9 | -1.2 | +2.4 | -3.7 | -2.8 | -3.4 | 3.7 |
| on+LP+HP | 73 | 0.003599 | 0.000715 | 0.000875 | 0.000092 | -6.6 | -1.6 | -1.1 | +2.4 | -3.7 | -2.6 | -3.3 | 3.7 |

## Decision rules
D3 (in-band LTAS gap <= 2 dB and matched KID reported with SE): observed in-band maximum 2.8 dB -> **FAIL**. B3 is mandatory because the in-band gap exceeds 2 dB and residual-on 4 kHz excess is +9.6 dB, above +8 dB.

## Interpretation
The residual improves matched KID and reduces the upper-band excess, but its cohort has a larger 1-3 kHz LTAS miss than residual off. The 3.8 kHz LP removes the 4 kHz excess while its WavLM shift from residual on (+0.000321) remains within the reported subset spread. The 150 Hz HP overcorrects the 100 Hz gap to -6.6 dB and does not repair the in-band miss. These development measurements support taking the LP candidate to the owner; they do not authorize a frozen-config change.

## Artifacts and exact commands

Validation:

```powershell
uv run pytest -q
uv run python scripts/lab/jobs.py lock status
```

Result after the B3 helper change: 781 passed, 3 skipped.

Render launch:

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-fid -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 150 --out runs/prod_fid --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set dataset.noise_only_frac=0
```

Render result: finished, exit 0, 150/150 clips; manifest and stats written.

Primary analysis and B3 preparation:

```powershell
uv run python scripts/analysis/make_matched_sets.py --out runs/prod_fid/kid --real-dir runs/calib_v2/clips --syn v1=runs/prod_fid/wavs
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid/wavs --label real --label v1 --limit 1000 --json runs/prod_fid/ltas.json
uv run python scripts/analysis/filter_variants.py runs/prod_fid/wavs --out runs/prod_fid/variants --manifest runs/prod_fid/manifest.jsonl
uv run python scripts/analysis/make_matched_sets.py --out runs/prod_fid/kid_b3 --real-dir runs/calib_v2/clips --syn off=runs/prod_fid/variants/off --syn on=runs/prod_fid/variants/on --syn on_lp=runs/prod_fid/variants/on_lp --syn on_lp_hp=runs/prod_fid/variants/on_lp_hp
uv run python scripts/analysis/ltas_check.py runs/prod_fid/variants/off runs/prod_fid/variants/on runs/prod_fid/variants/on_lp runs/prod_fid/variants/on_lp_hp --label off --label on --label on+LP --label on+LP+HP --json runs/prod_fid/ltas_b3.json
```

Matched KID evaluations:

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-fid-kid-v1 -- uv run python -m atcgen.eval.embed_dist runs/prod_fid/kid/v1_matched runs/prod_fid/kid/ref_matched --device cuda --out runs/prod_fid/kid/kid_v1_matched.json
uv run python scripts/lab/jobs.py launch --gpu --id prod-fid-kid-off -- uv run python -m atcgen.eval.embed_dist runs/prod_fid/kid_b3/off_matched runs/prod_fid/kid_b3/ref_matched --device cuda --out runs/prod_fid/kid_b3/kid_off_matched.json
uv run python scripts/lab/jobs.py launch --gpu --id prod-fid-kid-on -- uv run python -m atcgen.eval.embed_dist runs/prod_fid/kid_b3/on_matched runs/prod_fid/kid_b3/ref_matched --device cuda --out runs/prod_fid/kid_b3/kid_on_matched.json
uv run python scripts/lab/jobs.py launch --gpu --id prod-fid-kid-on-lp -- uv run python -m atcgen.eval.embed_dist runs/prod_fid/kid_b3/on_lp_matched runs/prod_fid/kid_b3/ref_matched --device cuda --out runs/prod_fid/kid_b3/kid_on_lp_matched.json
uv run python scripts/lab/jobs.py launch --gpu --id prod-fid-kid-on-lp-hp -- uv run python -m atcgen.eval.embed_dist runs/prod_fid/kid_b3/on_lp_hp_matched runs/prod_fid/kid_b3/ref_matched --device cuda --out runs/prod_fid/kid_b3/kid_on_lp_hp_matched.json
```

All five evaluations finished exit 0. The four B3 jobs took 99-101 seconds each.

## Not done / deviations
No frozen band edge was changed, no Section 3 production render started, and the locked-day data was not read. Per B3, work stops with this packet pending Kevin's band-edge decision.