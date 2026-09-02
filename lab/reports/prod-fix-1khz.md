# Report prod-fix-1khz                (spec: lab/specs/prod-v1-recipe-fix.md; brief: lab/briefs/prod-fix-1khz.md)
## Director summary
Landed as specified: new `peaking_eq` primitive (inert unless named) plus one chain step in `configs/mode2_v1.yaml` after the low-pass — `f0_hz 1100`, `gain_db 7.0`, `q 1.7`, `prob 1.0`; 785 passed, 3 skipped. Attempt-1 render (150 clips, seed 0, exit 0, 61 s, QC 150/150) real-cohort LTAS gaps at 100/200/300/400/1k/2k/3k/3.4k/4k Hz: +7.47/+1.82/+2.85/+0.44/**-0.53**/**-0.34**/**-1.43**/+1.41/-2.86 dB, in-band maximum **1.43 dB** (limit 2.0). Matched WavLM KID **0.004331 +/- 0.000938** (150/997), under the 0.005728 guardrail; CLAP 0.001063 +/- 0.000133. All three D3'' limbs hold: **PASS**. Attempt 2 did not run and no fallback was taken. `configs/mode2_v1.yaml` is in its final render state with the `peaking_eq` step in place; Section 3 was not started.

## Results

Landed chain (`configs/mode2_v1.yaml`, `calibrated.post_effects.chain`; nothing else in the file changed):

```yaml
chain:
  - primitive: lowpass
    cutoff_hz: 3800
    order: 8
    zero_phase: true
  - primitive: peaking_eq
    prob: 1.0
    f0_hz: 1100
    gain_db: 7.0
    q: 1.7
```

LTAS gaps are synthetic-cohort dB minus the named reference. Real-cohort gaps are the D3'' decision values; hardcoded-reference gaps are diagnostics only.

| Reference | Syn clips | Ref clips | 100 Hz | 200 Hz | 300 Hz | 400 Hz | 1 kHz | 2 kHz | 3 kHz | 3.4 kHz | 4 kHz | max abs 1/2/3 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real cohort (attempt 1) | 150 | 1000 | +7.47 | +1.82 | +2.85 | +0.44 | -0.53 | -0.34 | -1.43 | +1.41 | -2.86 | **1.43** |
| Real cohort (baseline, prod-fid-rerun) | 150 | 1000 | +7.94 | +1.77 | +1.90 | -0.45 | -4.69 | -1.09 | +0.02 | +2.90 | -6.55 | 4.69 |
| Hardcoded KIXD curve (attempt 1) | 150 | n/a | +13.01 | +1.02 | +3.43 | -0.66 | +6.84 | -0.39 | -0.77 | +3.11 | +3.25 | 6.84 |
| Hardcoded KIXD curve (real cohort) | 1000 | n/a | +5.54 | -0.80 | +0.58 | -1.10 | +7.37 | -0.05 | +0.66 | +1.70 | +6.11 | 7.37 |

Peak frequencies: real cohort 484.375 Hz; attempt-1 synthetic **359.375 Hz**, i.e. the synthetic peak migrated *down* off 484.375 Hz (reported, not gated; the spec anticipated migration only upward, to 953 Hz, above +7.5 dB of gain). Matched-set survival: 1,000 raw reference -> 997 matched, 150/150 synthetic matched, both sides median RMS -26.00 dB.

Matched embedding distances (energy-trimmed at -35 dB, RMS-normalized to -26 dB, fixed seed-0 reference subset):

| Render | Syn clips | Ref clips | WavLM KID | +/- SE | CLAP KID | +/- SE |
|---|---:|---:|---:|---:|---:|---:|
| Attempt 1 (V1 + LP + bell) | 150 | 997 | 0.004331 | 0.000938 | 0.001063 | 0.000133 |
| Baseline (V1 + LP, prod-fid-rerun) | 150 | 997 | 0.004134 | 0.000797 | 0.000987 | 0.000127 |

Reference-set identity: the attempt-1 `ref_matched` is byte-identical to the baseline's (`runs/prod_fid_d3prime/kidsets/ref_matched`), n=997 both sides, under four independent aggregate content hashes (sha256 over name+digest `e9947cbc911629ffe180`, over digests `95a7e929b47b8bf072ea`, over `name:digest` lines `449bf5e0fed0c710ec10`, blake2b over digests `a465e56130e3ac0fe972`). The audit's own recipe for `b50c7b33046cc271da30` is not recorded anywhere in the repo and was not reproduced; byte-identity to the set that produced the 0.004134 guardrail baseline is established directly instead.

Render QC: `stats.json` reports 150 total, 150 kept, 0 discarded, no retry reasons; the job log contains no clipping, QC-retry, failure or error lines.

## Decision rules
D3'' PASS iff all three limbs hold.

| Limb | Threshold | Observed | Verdict |
|---|---|---|---|
| (a) LTAS | max abs real-cohort gap at 1/2/3 kHz <= 2.0 dB | 1.43 dB | PASS |
| (b) KID availability | matched WavLM KID with SE and clip counts | 0.004331 +/- 0.000938, 150/997 | PASS |
| (c) KID guardrail | WavLM KID <= 0.004134 + 2 x 0.000797 = 0.005728 | 0.004331 | PASS |

**D3'': PASS.** Branch taken: **PASS on attempt 1**. Attempt 2 (`gain_db` resize) was not run and its resize delta was not computed; the fallback revert was not exercised. `configs/mode2_v1.yaml` stays exactly as landed above — that is its final render state for V1.0.0.

## Interpretation
At this budget, one fixed peaking bell at 1100 Hz closes the in-band shape deficit against the real cohort: the 1 kHz gap moves from -4.69 dB to -0.53 dB and the in-band maximum from 4.69 dB to 1.43 dB, 0.57 dB of margin under the 2.0 dB limit on a cohort-to-cohort wander previously measured at +/-0.6 dB. This is development evidence from one seed-0 cohort, and the comparison to the baseline is **unpaired**: the extra chain step consumes a per-clip RNG draw, so attempt 1 is a fresh cohort, not the same clips. The measured 1 kHz movement (+4.16 dB) exceeds the spec's predicted +3.78 dB (0.54 dB/dB x 7.0), and the binding limb shifted from 1 kHz to 3 kHz (-1.43), both consistent with cohort draw plus the sizing model's approximation; neither was used to adjust any threshold. The synthetic LTAS peak moved to 359.375 Hz, which is a shape change at the low edge (100 Hz is +7.47 dB and 300 Hz +2.85 dB against the real cohort) — reported, outside the gated band, and unchanged in status by this run. Realism did not degrade measurably: matched WavLM KID 0.004331 +/- 0.000938 is inside one SE of the baseline 0.004134 +/- 0.000797 and well under the pre-registered guardrail, so the bell did not buy LTAS by colouring the audio out of distribution. No KID threshold, no LTAS limit and no reference set was introduced or changed after seeing a result.

## Artifacts and exact commands

Preflight (CPU):

```powershell
uv run pytest -q
uv run python -c "from atcgen.config import load_config; c=load_config('configs/mode2_v1.yaml'); print([(s.primitive, s.prob, {k: v.as_dict() for k, v in s.params.items()}) for s in c.calibrated.post_effects.chain])"
```

-> `785 passed, 3 skipped`; chain prints `[('lowpass', 1.0, {...}), ('peaking_eq', 1.0, {'f0_hz': {'const': 1100}, 'gain_db': {'const': 7.0}, 'q': {'const': 1.7}})]`.

Render, matched sets, KID, LTAS:

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-fix-1khz -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 150 --out runs/prod_fid_d3pp_a1 --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set dataset.noise_only_frac=0
uv run python scripts/analysis/make_matched_sets.py --out runs/prod_fid_d3pp_a1/kidsets --real-dir runs/calib_v2/clips --syn v1=runs/prod_fid_d3pp_a1/wavs
uv run python scripts/lab/jobs.py launch --gpu --id prod-fix-1khz-kid -- uv run python -m atcgen.eval.embed_dist runs/prod_fid_d3pp_a1/kidsets/v1_matched runs/prod_fid_d3pp_a1/kidsets/ref_matched --device cuda --out runs/prod_fid_d3pp_a1/kidsets/kid_v1_matched.json
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid_d3pp_a1/wavs --label real --label v1 --limit 1000 --cohort-reference real --json runs/prod_fid_d3pp_a1/ltas.json
```

Job records: `lab/jobs/prod-fix-1khz/status.json` (exit 0, 61 s, child pid 9088) and `lab/jobs/prod-fix-1khz-kid/status.json` (exit 0, 137 s) — both GPU steps ran under the lock, `embed_dist` included.

Run artifacts: `runs/prod_fid_d3pp_a1/manifest.jsonl` (150 rows; `peaking_eq` present in 150/150 per-clip channel step lists, always last and always immediately after `lowpass`, `gain_db` 7.0 in all 150), `runs/prod_fid_d3pp_a1/config.resolved.yaml`, `runs/prod_fid_d3pp_a1/stats.json` (`config_hash cf0bf1a9784541b454e3f68e45a14b47209255d4933114fcf8de96baf93925aa`), `runs/prod_fid_d3pp_a1/ltas.json`, `runs/prod_fid_d3pp_a1/matched_sets.txt`, `runs/prod_fid_d3pp_a1/kidsets/kid_v1_matched.json`.

Frozen values re-read from the resolved config (`load_config('runs/prod_fid_d3pp_a1/config.resolved.yaml')`): `seed 0`; `tts.speed uniform [1.0, 1.4]`; `pitch_semitones prob 0.5`, `tempo prob 0.3`, `eq_tilt_db prob 0.4`; residual enabled on `runs/fastcut_v1/G_selected.pt` at `apply_prob 0.5`, `alpha 1.0`, `residual_scale_max 0.35`; six-station uniform `station_mix` (KEUG/KIXD/KOJC/KSDL/KSLE/S50). Unchanged.

Code landed: `peaking_eq` in `atcgen/channel/primitives.py` (single `sosfilt` pass over one RBJ peaking biquad via the existing `_peaking_sos`, registered in `PRIMITIVES`) and `test_peaking_eq_realizes_its_gain_at_f0_and_stays_local` in `tests/test_primitives.py` (+7.0 dB at 1100 Hz within 0.2 dB, |response| < 0.5 dB at 110 Hz and 3300 Hz, RNG state unchanged).

Not run: attempt 2, the resize formula, the fallback revert, and runbook Section 3. `data/real/kixd/kixd_locked_day.csv` was not read; no file inside a dataset directory was listed or read individually.
