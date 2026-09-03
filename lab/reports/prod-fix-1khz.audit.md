# Audit prod-fix-1khz            (report: lab/reports/prod-fix-1khz.md; spec: lab/specs/prod-v1-recipe-fix.md)

**AUDIT: PASS-WITH-NOTES.** Every gating number recomputes from raw artifacts —
real-cohort in-band max **1.43 dB** and matched WavLM KID **0.004331 ± 0.000938
(150/997)** against a reference set proved content-identical to the one that
produced the 0.005728 guardrail — so **D3'' PASS stands and the V1 render is
cleared**; the notes are provenance wording, not results.

## Verdicts

| # | Check | Verdict |
|---|---|---|
| 1 | Pre-registration: rule, budget, kill criterion match the spec written first | CONFIRMED (note A) |
| 2a | Landed change = spec §2: one `peaking_eq` primitive + registry entry, one chain step (f0 1100, gain 7.0, q 1.7) after the LP | CONFIRMED |
| 2b | Primitive is inert unless named — no other config or profile picks it up | CONFIRMED |
| 2c | Fix commit contains nothing else beyond primitive + test + config step + lab paperwork | CONFIRMED (note B) |
| 2d | Test suite claim `785 passed, 3 skipped` | CONFIRMED |
| 3a | Real-cohort LTAS gaps 1/2/3 kHz = −0.53 / −0.34 / −1.43, in-band max 1.43 ≤ 2.0 | CONFIRMED |
| 3b | All reported non-gated bands, both peak frequencies (484.375 / 359.375 Hz) | CONFIRMED |
| 4a | Matched WavLM KID 0.004331 ± 0.000938, CLAP 0.001063 ± 0.000133, n = 150/997 | CONFIRMED (as recorded) |
| 4b | KID re-derived from audio through `embed_dist` | UNVERIFIABLE (note C) |
| 5a | Same 997-clip seed-0 reference as the guardrail baseline | CONFIRMED |
| 5b | Literal aggregate hash `b50c7b33046cc271da30` reproduced | UNVERIFIABLE (note D) |
| 5c | Trim/normalize hygiene: −35 dB energy trim, −26 dB RMS both sides, defaults unmodified, scored set derives from this render | CONFIRMED (note E) |
| 6 | Guardrail arithmetic 0.004134 + 2 × 0.000797 = 0.005728, applied one-sided | CONFIRMED |
| 7 | D3'' PASS follows from the spec's three limbs as worded | CONFIRMED |
| 8 | Render provenance: config identity, 150/150 `peaking_eq` rows, QC 150/150, both GPU steps under the lock | CONFIRMED |
| 9 | Data discipline: no locked-day reference anywhere in the run | CONFIRMED |
| 10 | Sizing model reproduces on identical clips (paired offline bell) | CONFIRMED |

## What was recomputed

**LTAS (item 3).** Fresh run, the report's own JSON untouched:

```powershell
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid_d3pp_a1/wavs --label real --label v1 --limit 1000 --cohort-reference real --json $env:TEMP\audit_fix1khz_ltas.json
```

| Curve | 100 | 200 | 300 | 400 | 1k | 2k | 3k | 3.4k | 4k | in-band | peak Hz | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real (`runs/calib_v2/clips`) | −16.26 | −11.20 | −5.12 | −1.10 | −1.43 | −12.15 | −23.34 | −33.70 | −49.29 | | 484.375 | 1000 |
| v1 (`runs/prod_fid_d3pp_a1/wavs`) | −8.79 | −9.38 | −2.27 | −0.66 | −1.96 | −12.49 | −24.77 | −32.29 | −52.15 | | 359.375 | 150 |
| **v1 − real** | **+7.47** | **+1.82** | **+2.85** | **+0.44** | **−0.53** | **−0.34** | **−1.43** | **+1.41** | **−2.86** | **1.43** | | |

Every real-cohort figure reproduces at printed precision, as do both
hardcoded-reference rows (+13.01 … and +5.54 …) and both peak frequencies.
`ltas_check.py` selects `1000 ≤ f ≤ 3000`, which on the measurement grid is
exactly the spec's `{1000, 2000, 3000}`; `max(0.53, 0.34, 1.43) = 1.43 ≤ 2.0` →
limb (a) PASS, correctly derived. The downward peak migration to 359.375 Hz is
real and is reported, not gated, as the spec directs.

**Reference-set identity and matched-set provenance (items 5a, 5c).** Aggregate
content hashes only, no per-file listing surfaced; fourteen independent recipes
(sha256/blake2b/md5/sha1 over digests, name+digest, `name:digest` lines, raw
digest bytes, sorted variants) agree pairwise across all three sets:

```
runs/prod_fid_d3pp_a1/kidsets/ref_matched   n=997  sha256(digests)=f30b06914d3cbe18cef6
runs/prod_fid_d3prime/kidsets/ref_matched   n=997  sha256(digests)=f30b06914d3cbe18cef6
runs/prod_fid/kid/ref_matched               n=997  sha256(digests)=f30b06914d3cbe18cef6
```

So the KID that was scored used byte-for-byte the reference set behind the
0.004134 ± 0.000797 guardrail baseline (and behind the earlier `prod-fid`
numbers). The synthetic side is genuinely this render and not a stale copy:
re-running the shipped `trim_and_norm` over `runs/prod_fid_d3pp_a1/wavs` into a
temp directory reproduces the scored set exactly, and the two renders differ —

```
a1 wavs (render)              n=150  agg=ed393887ffb9b5fb830e
d3prime wavs (baseline)       n=150  agg=39e17764a27d60a64f29
a1 v1_matched (as scored)     n=150  agg=4870ed87ea4f29f8ba88
rebuilt from a1 wavs          n=150  agg=4870ed87ea4f29f8ba88
```

Recomputed levels over the **full** sets (not the 400-file prefix
`make_matched_sets.py` prints): `ref_matched` n=997 rms_med −26.00 dB,
`v1_matched` n=150 rms_med −26.00 dB, both 16 kHz. Defaults `--n-ref 1000
--seed 0`, `TRIM_REL_DB −35.0`, `TARGET_RMS_DB −26.0` are unmodified by the
report's command.

**Sizing model, paired (item 10).** Applying the *shipped* primitive
(`f0 1100 / +7.0 dB / Q 1.7`) offline to the baseline render's 150 clips and
re-measuring against the real cohort reproduces the spec's chosen design row
exactly: gaps +7.07 / +0.88 / +1.59 / −0.70 / **−1.00 / −1.04 / −0.84** /
+1.96 / −7.63, in-band **1.04 dB**, peak unmoved at 484.375 Hz. The filter that
shipped is the filter that was sized, and the fresh-cohort 1.43 vs the
same-clips 1.04 is a cohort-draw difference of the size the report claims
(±0.6 dB), not a modelling error. The report's "unpaired" caveat is correct and
correctly worded.

**Render and process (items 2, 8, 9).** `git show --stat 72ba48a`: seven files —
`atcgen/channel/primitives.py` (+8: one function plus one `PRIMITIVES` entry),
`configs/mode2_v1.yaml` (+5: the chain step, LP untouched),
`tests/test_primitives.py` (+16), and the four lab documents. `peaking_eq`
appears in exactly one config (`configs/mode2_v1.yaml`); `load_config` has no
include/extends/inheritance path, so no other profile can acquire it — inert
unless named. `dataclasses.asdict(load_config(runs/prod_fid_d3pp_a1/config.resolved.yaml))`
is **identical** to `load_config('configs/mode2_v1.yaml')` with
`dataset.noise_only_frac=0`, which simultaneously proves the render used the
landed recipe, that every frozen value is unchanged (seed 0; `tts.speed
uniform [1.0, 1.4]`; pitch 0.5 / tempo 0.3 / eq_tilt 0.4; residual on
`runs/fastcut_v1/G_selected.pt`, `apply_prob 0.5`, `alpha 1.0`,
`residual_scale_max 0.35`; six-station uniform mix), and that the config is in
its final render state today. Manifest: 150 rows, `peaking_eq` in 150/150,
always last and immediately after `lowpass`, `gain_db 7.0` throughout. `stats.json`
QC 150/150 kept, 0 discarded, no reasons. `status.json` for both `prod-fix-1khz`
(exit 0, 61 s) and `prod-fix-1khz-kid` (exit 0, 137 s) carry `"gpu": true`, so
`embed_dist` ran under the lock as the spec requires. Job log: 168 lines, no
clipping, QC-retry, failure or error line — only library deprecation warnings
(one numba `invalid value encountered` warning appears; harmless here, since all
150 clips pass `ltas_check`'s finite/non-zero filter). No artifact, config or
command mentions `kixd_locked_day`, and no manifest row references the locked
day `20250808`.

`uv run pytest -q` → `785 passed, 3 skipped`, matching the report.

## Notes (nothing here changes the verdict)

**A — pre-registration rests on mtimes, not git.** Spec, brief, code, config,
report and board landed in one commit (`72ba48a`), so history alone cannot
separate design from result. File mtimes do, and they are consistent and
ordered: spec 16:07:14 → brief 16:07:54 → primitive 16:08:56 → test 16:09:33 →
config 16:09:38 → render 16:12:20–16:13:21 → `ltas.json` 16:14:29 →
`kid_v1_matched.json` 16:16:12 → report 16:19:35. The spec also pre-records its
own prediction (in-band ≈ 1.0 dB) and the instruction that the prediction must
not move the rule, and the observed 1.43 was *not* used to adjust anything.
Same structural caveat the `prod-fid-rerun` audit raised; fix for future
missions: commit the spec before launching the job.

**B — the commit carries lab paperwork too.** Beyond primitive + test + config
step, `72ba48a` also contains `lab/specs/prod-v1-recipe-fix.md`,
`lab/briefs/prod-fix-1khz.md`, `lab/STATE.md` and the report. No further code,
config, profile or run-directory change is in it. Also cosmetic: the spec §5
pre-registered job ids `prod-v1-fix-a1` / `-kid` and report path
`lab/reports/prod-v1-recipe-fix.md`; the brief renamed both to `prod-fix-1khz`,
so the deviation is authorized, but a reader following the spec alone will look
for artifacts that do not exist. Fix: one line in the report's "not done /
deviations" recording the rename.

**C — KID was not re-derived from audio (GPU).** This auditor is CPU-only; the
0.004331 ± 0.000938 is confirmed against `kid_v1_matched.json` and its inputs,
not against a fresh forward pass. The KID subset draw is seeded
(`np.random.default_rng(0)`, 100 subsets of 50), so a rerun should reproduce up
to GPU float nondeterminism. Settled by:

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-fix-1khz-kid-audit -- uv run python -m atcgen.eval.embed_dist runs/prod_fid_d3pp_a1/kidsets/v1_matched runs/prod_fid_d3pp_a1/kidsets/ref_matched --device cuda --out runs/prod_fid_d3pp_a1/kidsets/audit_kid.json
```

**D — the spec's `b50c7b33046cc271da30` is still unreproducible.** Fourteen
recipes over `ref_matched` produce none of it, matching the report's own
statement. The spec's limb (c) says "verify before quoting"; what is verifiable —
and was verified here — is n = 997 plus byte-identity to the exact set that
produced the guardrail baseline, which is what limb (c) actually needs. Fix for
future specs: quote a hash together with the command that generates it, e.g.
`sha256` over the concatenated sorted per-file digests (value here:
`f30b06914d3cbe18cef6…`).

**E — `matched_sets.txt` is a hand-saved transcript, and its reference rows are
a 400-clip prefix.** Its mtime (16:18:43) is after the KID run, and
`make_matched_sets.py` computes its printed duration/zero-fraction/RMS over
`sorted(...)[:400]`, so the `ref_raw` / `ref_matched` rows describe a prefix,
not the full sets (full-set `ref_raw` rms_med is −22.73 dB, not the −27.98 dB
printed). The decision-relevant claim — both scored sides at −26.00 dB median —
holds on the full sets, recomputed above, so nothing gated depends on the file.

**F — carried caveats, unchanged by this run.** The reference subset is drawn
from all of `runs/calib_v2/clips`, which includes the P1 preset-fit split, so
the *absolute* KID level is optimistic; limb (c) is a same-reference relative
guardrail, so the D3'' conclusion is unaffected. The synthetic side is 150
clips, exactly at the lab's reportability floor. The low-edge excess
(+7.47 dB at 100 Hz, +2.85 dB at 300 Hz) and the downward peak migration are
outside the gated band and remain open, quantified limitations — the report
states this correctly. The spec's own open question about
`residual_scale_max: 0.35` versus the 0.20 recorded as frozen in runbook §2 is
untouched by this fix and still awaits the director.
