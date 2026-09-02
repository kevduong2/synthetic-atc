# Report prod-fid-rerun                (spec: lab/specs/prod-fid-bandedge.md; brief: lab/briefs/prod-fid-rerun.md)
## Director summary
The final Mode 2 chain step landed exactly as `primitive: lowpass`, `cutoff_hz: 3800`, `order: 8`, `zero_phase: true`. The fresh 150-clip seed-0 render finished exit 0. Matched WavLM KID is 0.004134 +/- 0.000797 (CLAP 0.000987 +/- 0.000127; 150/997 clips). Synthetic-minus-real-cohort LTAS gaps at 100/200/400/1k/2k/3k/4k Hz are +7.94/+1.77/-0.45/-4.69/-1.09/+0.02/-6.55 dB. The preregistered in-band maximum is 4.69 dB, above 2.0 dB. D3' **FAILS** and the mission **STOPS**; Section 3 and alternate filters did not run. The next action is a new preregistered recipe-correction mission.

## Results
Landed final Mode 2 chain step:

```yaml
calibrated:
	post_effects:
		chain:
			- primitive: lowpass
				cutoff_hz: 3800
				order: 8
				zero_phase: true
```

Matched embedding distances (energy-trimmed, RMS-normalized to -26 dB; fixed seed-0 reference subset):

| Render | Synthetic clips | Reference clips | WavLM matched KID | +/- SE | CLAP matched KID | +/- SE |
|---|---:|---:|---:|---:|---:|---:|
| V1 + LP | 150 | 997 | 0.004134 | 0.000797 | 0.000987 | 0.000127 |

LTAS gaps are synthetic-cohort dB minus the named reference. Real-cohort gaps are the D3' decision values; hardcoded-reference gaps are diagnostics only.

| Reference | Synthetic clips | Reference clips | 100 Hz | 200 Hz | 400 Hz | 1 kHz | 2 kHz | 3 kHz | 4 kHz | max abs 1/2/3 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real cohort | 150 | 1000 | +7.94 | +1.77 | -0.45 | -4.69 | -1.09 | +0.02 | -6.55 | 4.69 |
| Hardcoded KIXD curve | 150 | n/a | +13.48 | +0.97 | -1.55 | +2.68 | -1.14 | +0.68 | -0.44 | 2.68 |

Real and synthetic LTAS curves both peak at 484.375 Hz. The matched-set summary records 1,000 raw reference clips, 997 surviving matched reference clips, and all 150 synthetic clips surviving matching.

## Decision rules
D3' PASS if and only if the maximum absolute real-cohort LTAS gap at 1/2/3 kHz is <= 2.0 dB and primary WavLM KID is validly matched and reported with SE. Observed LTAS maximum: 4.69 dB. Observed matched WavLM KID: 0.004134 +/- 0.000797 with 150/997 clips. The KID limb is available; the LTAS limb fails. D3': **FAIL-STOP**.

## Interpretation
At this budget, the fresh render supports the preregistered STOP decision. The 3.8 kHz low-pass removes the prior upper-edge excess but does not repair the real-cohort in-band miss: the 1 kHz gap is -4.69 dB. The low- and high-edge gaps are diagnostic and do not alter the verdict. No matched-KID threshold was introduced after seeing the result.

## Artifacts and exact commands

Config smoke:

```powershell
uv run python -c "from atcgen.config import load_config; c=load_config('configs/mode2_v1.yaml'); s=c.calibrated.post_effects.chain[-1]; print(s.primitive, s.prob, {k: v.as_dict() for k, v in s.params.items()})"
```

Launch preflight: `uv run pytest -q` -> 782 passed, 3 skipped. Final validation after adding the cohort-gap regression test: 784 passed, 3 skipped.

Render launch:

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-fid-rerun -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 150 --out runs/prod_fid_d3prime --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set dataset.noise_only_frac=0
```

Render result: finished exit 0 in 59 seconds; 150/150 clips; fresh `manifest.jsonl`, `config.resolved.yaml`, and `stats.json` written.

Matched sets and KID:

```powershell
uv run python scripts/analysis/make_matched_sets.py --out runs/prod_fid_d3prime/kidsets --real-dir runs/calib_v2/clips --syn v1=runs/prod_fid_d3prime/wavs
uv run python -m atcgen.eval.embed_dist runs/prod_fid_d3prime/kidsets/v1_matched runs/prod_fid_d3prime/kidsets/ref_matched --device cuda --out runs/prod_fid_d3prime/kidsets/kid_v1_matched.json
```

Matched-set aggregate summary: `runs/prod_fid_d3prime/matched_sets.txt`. KID artifact: `runs/prod_fid_d3prime/kidsets/kid_v1_matched.json`.

LTAS:

```powershell
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid_d3prime/wavs --label real --label v1 --limit 1000 --cohort-reference real --json runs/prod_fid_d3prime/ltas.json
```

The LTAS artifact contains both measured curves, direct `v1 - real` gaps, clip counts, and peak frequencies: `runs/prod_fid_d3prime/ltas.json`.

## Not done / deviations
No deviations from the preregistration. Per the kill criterion, Section 3, the 150 Hz high-pass, other edge or mid-band filters, reference changes, and gate relaxation did not run. The locked-day data was not read.
