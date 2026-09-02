# Audit prod-fid-rerun          (report: lab/reports/prod-fid-rerun.md; spec: lab/specs/prod-fid-bandedge.md; brief: lab/briefs/prod-fid-rerun.md; mission: lab/missions/prod-v1.md)

AUDIT: **PASS-WITH-NOTES** — every gating number reproduces exactly from the raw artifacts and the D3' FAIL-STOP is correctly derived and robust (the 1 kHz miss is real, pre-existed the low-pass, and was masked by the defective hardcoded reference), but the ~124 s matched-KID evaluation ran outside the `jobs.py` GPU lock and the report's cross-render wording implies an A/B that the render pair does not support (the RNG stream shifted, so only 2 of 150 clips share their voice/augment draws with the previous render).

## Verdict table

| # | Check | Verdict |
|---|---|---|
| 1 | Pre-registration: D3' rule, in-band set, 2.0 dB limit, budget and kill criterion match the spec as written before the run | CONFIRMED |
| 2 | LTAS gap table vs the REAL cohort independently recomputed with fresh `--out`, all seven printed bands | CONFIRMED |
| 3 | In-band (spec: 1/2/3 kHz) max abs gap = 4.69 dB > 2.0 dB → D3' FAIL derived correctly | CONFIRMED |
| 4 | Matched KID 0.004134 ± 0.000797 (CLAP 0.000987 ± 0.000127; 150/997) read from the raw `embed_dist` artifact, not a summary | CONFIRMED |
| 4b | Independent re-derivation of KID from embeddings | UNVERIFIABLE (GPU) |
| 5 | Matched-set hygiene: same 997-clip seed-0 reference as `prod-fid` (byte-identical), −35 dB trim, −26 dB RMS both sides, ≥150 synthetic clips | CONFIRMED |
| 6 | Puzzle arithmetic: +2.68 (rerun vs hardcoded) − 7.37 (real vs hardcoded) = −4.69 at 1 kHz; previous render was already −4.60 vs the real cohort | CONFIRMED |
| 7 | The LP changes nothing at or below 3 kHz (paired, same clips: +0.0 / +0.0 / −0.1 dB at 1/2/3 kHz; −13.0 dB at 4 kHz) | CONFIRMED |
| 8 | Config change is exactly one chain step between the two fid commits | CONFIRMED |
| 9 | Render provenance: exit 0 in 59 s under the GPU lock, 150/150 clips, LP recorded in 150/150 manifest rows, frozen values and `G_selected.pt` intact | CONFIRMED |
| 10 | Data discipline: no artifact, config or command reads `kixd_locked_day` | CONFIRMED |
| 11 | Preflight/validation test counts (784 passed, 3 skipped) | CONFIRMED |
| 12 | Process: all GPU work over a minute ran through `jobs.py launch --gpu` (one-GPU-stream rule) | **CONTRADICTED** |
| 13 | Cross-render claims are made on comparable cohorts, or the divergence is disclosed | **CONTRADICTED** |

## Evidence

**1. Pre-registration.** `lab/specs/prod-fid-bandedge.md` fixes the chain step, the real-cohort re-reference, `f in {1000, 2000, 3000}`, the 2.0 dB limit, the KID-availability limb, the one-cell budget and the FAIL-STOP kill rule. The report's decision line, budget and "not done" section restate them without alteration; the spec even pre-records the prediction that the in-band miss is ~4.6 dB and that the prediction must not change the rule — so the FAIL is not a post-hoc reading. Ordering is established by mtime: spec 10:01:47 → brief 10:02:20 → `configs/mode2_v1.yaml` 10:03:13 → `manifest.jsonl` 10:07:55 → `ltas.json` 10:13:00 → `kid_v1_matched.json` 10:15:30 → report 10:18:45. Note: the spec, brief, config, code and report all landed in one commit (`2e2dee2`), so git history alone does not separate preregistration from result; the mtimes and the spec's stated prediction are what carry it.

**2–3. LTAS recomputed.** Fresh independent run, the report's own JSON untouched:

```powershell
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid/wavs runs/prod_fid_d3prime/wavs --label real --label prev --label rerun --limit 1000 --cohort-reference real --json runs/prod_fid_d3prime/audit_ltas_renders.json
```

| Curve (dB re peak) | 100 | 200 | 400 | 1k | 2k | 3k | 4k | peak Hz | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real (`runs/calib_v2/clips`) | −16.26 | −11.20 | −1.10 | −1.43 | −12.15 | −23.34 | −49.29 | 484.375 | 1000 |
| rerun (`runs/prod_fid_d3prime/wavs`) | −8.32 | −9.43 | −1.55 | −6.12 | −13.24 | −23.32 | −55.84 | 484.375 | 150 |
| **rerun − real** | **+7.94** | **+1.77** | **−0.45** | **−4.69** | **−1.09** | **+0.02** | **−6.55** | | |

Every figure in the report's real-cohort row reproduces at printed precision, as does the hardcoded-reference row (+13.48 / +0.97 / −1.55 / +2.68 / −1.14 / +0.68 / −0.44, in-band max 2.68) and the both-peak-at-484.375 Hz claim. The spec's in-band set is `{1000, 2000, 3000}`; `ltas_check.py` selects `1000 <= f <= 3000`, which over the measurement grid is exactly those three points. `max(|−4.69|, |−1.09|, |+0.02|) = 4.69 > 2.0` → **D3' FAIL**, correctly derived. (Diagnostic, not gated: 300 Hz +1.90 and 3.4 kHz +2.90 vs the real cohort.)

**4–5. Matched KID and set hygiene.** `runs/prod_fid_d3prime/kidsets/kid_v1_matched.json` → `wavlm.kid = 0.004134`, `kid_std = 0.000797`, `n_synthetic = 150`, `n_real = 997`; `clap.kid = 0.000987`, `kid_std = 0.000127`. Exact match to the report; only `v1_matched` vs `ref_matched` is quoted, no raw KID. `matched_sets.txt`: `ref_raw n=1000` → `ref_matched n=997 rms_db_med=−26.00`, `v1_raw n=150 rms_db_med=−23.06` → `v1_matched n=150 rms_db_med=−26.00`. `make_matched_sets.py` defaults are `--n-ref 1000 --seed 0`, `TRIM_REL_DB = −35.0`, `TARGET_RMS_DB = −26.0`, and the report's command overrides none of them. Aggregate content hashes (no per-file listing surfaced):

```
runs/prod_fid/kid/ref_matched              n=997   agg=b50c7b33046cc271da30
runs/prod_fid_d3prime/kidsets/ref_matched  n=997   agg=b50c7b33046cc271da30
runs/prod_fid/wavs                         n=150   agg=ae3c997822c6b287434b
runs/prod_fid_d3prime/wavs                 n=150   agg=81bddd690764af76b62a
runs/prod_fid_d3prime/kidsets/v1_matched   n=150   agg=3dd5993db9bcdbe14dd3
```

The reference set is **byte-identical** to the one used in `prod-fid` (same aggregate hash as recorded in `prod-fid.audit.md`), so this KID is directly comparable to the earlier 0.003443 ± 0.000731; the two renders' wavs are genuinely different audio. Synthetic side is 150 clips, at the reportability floor. **4b:** re-deriving KID from the embeddings needs the GPU and was not run (auditor is CPU-only); it would be settled by `uv run python -m atcgen.eval.embed_dist runs/prod_fid_d3prime/kidsets/v1_matched runs/prod_fid_d3prime/kidsets/ref_matched --device cuda --out runs/prod_fid_d3prime/kidsets/audit_kid.json`. Carried-over caveat from `prod-fid.audit.md`: the reference is drawn from all of `runs/calib_v2/clips`, which includes the P1 preset-fit split, so the absolute KID level is optimistic — identically so for both renders, and D3' gates only availability, not a level.

**6–7. The puzzle, settled: the 1 kHz miss existed all along and was masked by the defective reference.** The two facts compose exactly, not just approximately, because both are gaps against the same hardcoded curve: `gap_vs_real = gap_vs_hardcoded(syn) − gap_vs_hardcoded(real)`. At 1 kHz, rerun `+2.68 − 7.37 = −4.69`; previous render `+2.77 − 7.37 = −4.60`. The previous render, measured against the real cohort, was already at **−4.60 dB** at 1 kHz — recomputed here, not quoted (`prev − real`: +10.75 / +2.65 / +0.11 / **−4.60** / −1.64 / −0.58 / +10.29, in-band max 4.60). The 0.09 dB difference between −4.60 and −4.69 is cohort-draw noise.

The low-pass is ruled out as a cause by a paired, same-clips measurement — the identical 8th-order 3.8 kHz zero-phase Butterworth (`filter_variants.py` uses `butter(8, 3800, 'low')` + `sosfiltfilt`, matching the shipped `primitives.lowpass`) applied to the same 73 clips:

```powershell
uv run python scripts/analysis/ltas_check.py runs/prod_fid/variants/on runs/prod_fid/variants/on_lp --label on --label on_lp --cohort-reference on --json runs/prod_fid_d3prime/audit_ltas_lp_paired.json
```

```
on_lp - on   100:+0.0  200:+0.0  400:+0.0  1k:+0.0  2k:+0.0  3k:-0.1  3.4k:-0.6  4k:-13.0
```

So the LP moves the gated band by at most 0.1 dB and cannot have created, hidden or worsened a 4.7 dB miss at 1 kHz. Direction check as well: the rerun's own in-band curve moved **up** relative to the previous render (1k −6.03 → −6.12, 2k −13.79 → −13.24, 3k −23.92 → −23.32), the opposite sign to any low-pass, confirming those ≤0.6 dB shifts are cohort noise rather than filter effect. Verdict on the report's interpretation ("removes the prior upper-edge excess but does not repair the real-cohort in-band miss"): correct.

**8. Config change.** `git diff 182ab36 2e2dee2 -- configs/mode2_v1.yaml` is exactly five added lines under `calibrated.post_effects` and nothing else:

```yaml
    chain:
      - primitive: lowpass
        cutoff_hz: 3800
        order: 8
        zero_phase: true
```

`runs/prod_fid_d3prime/config.resolved.yaml` carries it as `primitive: lowpass, prob: 1.0, cutoff_hz: {const: 3800}, order: {const: 8}, zero_phase: {const: true}` with every frozen value unchanged (`tts.speed [1.0, 1.4]`, `pitch_semitones.prob 0.5`, `tempo.prob 0.3`, `eq_tilt_db.prob 0.4`, the six-station `station_mix`, `residual.enabled: true` on `runs/fastcut_v1/G_selected.pt` at `apply_prob 0.5`, `dataset.noise_only_frac: 0.0`, `seed: 0`). The same commit also adds the supporting code the step needs (`primitives.lowpass`, `PostEffectsConfig.chain`, `CalibratedChannel._final_chain`) and the `--cohort-reference` flag on `ltas_check.py`; the render-path addition is a single appended call after the residual and touches no existing effect.

**9. Render provenance.** `lab/jobs/prod-fid-rerun/status.json`: `state finished`, `exit_code 0`, `elapsed_s 59`, `gpu true`, command identical to the report's. Manifest aggregates: 150 rows, `lowpass` present in 150/150 rows of the rerun and 0/150 of the previous render; per-clip chain is `calibrated_preset → codec_roundtrip → lowpass`; residual applied on 73/150 rows in both renders, consistent with `apply_prob 0.5`.

**10. Data discipline.** A repository-wide search for `kixd_locked_day` returns no hit in any `runs/prod_fid_d3prime` artifact, in `configs/mode2_v1.yaml`, in the spec or in the report; the only hit in this packet is the brief's "Do not" line. No EU-row or `channel_val` selection is involved in a fidelity phase.

**11. Tests.** `uv run pytest -q` reproduces **784 passed, 3 skipped** (113 s), matching the report's post-change count.

**12. CONTRADICTED — GPU work outside the lock.** The render went through `jobs.py launch --gpu` (job record above), but the matched-KID evaluation did not: there is no `lab/jobs/prod-fid-rerun-kid*` directory, and the report's command is a bare `uv run python -m atcgen.eval.embed_dist ... --device cuda`. The job took ~124 s of GPU time (`wavlm.seconds 43.9` + `clap.seconds 80.7`), well past the one-minute threshold in `.github/skills/gpu-jobs/SKILL.md` §1, and so ran unserialized against the lock. The `prod-fid` packet launched the same evaluation through `jobs.py` five times, so this is a regression in practice, not a convention change. No number is affected (nothing else held the GPU). Fix: run future `embed_dist` calls as `uv run python scripts/lab/jobs.py launch --gpu --id <id> -- uv run python -m atcgen.eval.embed_dist ...`, and record the deviation in "Not done / deviations" rather than stating "No deviations".

**13. CONTRADICTED — the two renders are not an A/B.** `_final_chain` draws from the same per-build `random.Random(config.seed)` stream that `build_dataset` threads through every clip, so adding one chain step shifts the stream for every subsequent clip. Measured on the manifests: only **2 of 150** clips share the same `voice/speed/pitch/tempo/eq_tilt_db` draws across the two renders, and the first divergence is at index 1 (the texts are identical, 150/150, since the source is sequential). The rerun is therefore a fresh cohort under a new recipe, not the previous cohort plus a filter — visible in the LTAS as a peak-bin shift (468.75 → 484.375 Hz) and ±0.6 dB in-band wander. Nothing in the report discloses this, while the summary and interpretation read as a before/after on one recipe ("removes the prior upper-edge excess"). The claim itself survives, because the paired same-clips `on` → `on_lp` evidence in item 7 supports it, but the support comes from that paired comparison and not from the two renders. Fix, wording only: state that the LP's band effect is established paired on identical clips (−13.0 dB at 4 kHz, ≤0.1 dB in band) and that the two 150-clip renders differ in every per-clip draw, so their band-by-band differences are not attributable to the filter.

## Bottom line for the mission

The kill criterion in `lab/specs/prod-fid-bandedge.md` fires on the LTAS limb alone, and that limb reproduces exactly: in-band max 4.69 dB against the real target cohort, more than twice the 2.0 dB limit, with the dominant term a −4.7 dB deficit at 1 kHz that pre-dates this rerun and is untouched by the shipped filter. The KID limb is available and valid (0.004134 ± 0.000797, 150/997, same reference as before). **The D3' FAIL and the `prod-v1` STOP stand.** Neither contradicted line touches a gating number.

## Recompute artifacts

`runs/prod_fid_d3prime/audit_ltas_renders.json` (real + both renders, direct gaps), `runs/prod_fid_d3prime/audit_ltas_lp_paired.json` (paired LP effect on identical clips). Fresh `--out` in both cases; the report's `ltas.json` was not touched. No report, spec, config, code or `docs/results.md` file was modified by this audit. No file inside a dataset directory was listed or read individually — counts and content hashes only.
