# Spec prod-v1-recipe-fix

Question: does one bounded, measurement-sized corrective EQ step close the V1
recipe's in-band LTAS shape deficit against the real cohort without costing
matched-KID realism?

Hypothesis: if the −4.69 dB gap at 1 kHz is a residual *source-spectrum* error
that the fitted per-clip channel only partly inverts, then a single fixed
peaking biquad centred in the deficit, sized from the measured gaps, brings
`max |LTAS_syn − LTAS_real|` over {1, 2, 3} kHz to ≤ 2.0 dB while matched WavLM
KID stays within 2 × SE of the current 0.004134.

Decision this changes: on PASS the corrected `configs/mode2_v1.yaml` is the
recipe that renders V1.0.0; on FAIL after two attempts the corrective step is
reverted and V1.0.0 renders from the frozen recipe with the deficit documented.
**The mission proceeds to runbook §3 in every branch** (pre-authorized by
Kevin, 2026-09-02).

Authority: Kevin authorized exactly ONE pre-registered frozen-value correction
for this deficit. This spec spends it. Everything else in runbook §5 stays
frozen; no arms, no search, no gate relaxation, no second lever.

## 1. Mechanism: where the 1 kHz deficit comes from

Measured CPU-only on existing artifacts (`runs/prod_fid_d3prime/mechanism_probe.py`,
read-only; `ltas_check.py` math — per-clip power normalization, 1024-pt Hann
Welch, peak-normalized dB; real curve read from `audit_ltas_renders.json`):

| curve (dB re own peak) | 100 | 200 | 300 | 400 | 1k | 2k | 3k | 3.4k | 4k | peak Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R real cohort (n=1000) | −16.26 | −11.20 | −5.12 | −1.10 | **−1.43** | −12.15 | −23.34 | −33.70 | −49.29 | 484.4 |
| P clean TTS probe (n=200, `runs/gan_a_base_v1/clean`) | −6.35 | −0.31 | −8.07 | −6.09 | **−15.08** | −21.69 | −23.58 | −24.81 | −25.12 | 156.2 |
| C = P + mean fitted preset EQ (n=1232 presets) | −16.82 | −13.24 | −4.38 | −1.25 | **−9.03** | −15.00 | −27.47 | −38.05 | −54.93 | 421.9 |
| S rendered V1 + LP (n=150) | −8.32 | −9.43 | −3.22 | −1.55 | **−6.12** | −13.24 | −23.32 | −30.80 | −55.84 | 484.4 |
| **S − R (the D3′ gaps)** | **+7.94** | **+1.77** | **+1.90** | **−0.45** | **−4.69** | **−1.09** | **+0.02** | **+2.90** | **−6.55** | |

Reading, in claim-discipline terms — this is development evidence from one
seed-0 cohort and a mean-EQ approximation, not a proof of causation:

- The deficit is **talker-side in origin**. Clean Kokoro TTS is 13.65 dB short
  at 1 kHz on a shape basis (P − R); its LTAS peaks at 156 Hz against the real
  cohort's 484 Hz and rolls off from ~600 Hz where real speech holds a plateau
  to ~1.1 kHz.
- The **fitted channel only partly inverts it**. `channel_fit` drives a TTS
  probe through the fitted chain and matches the real clip's LTAS, but the EQ
  is smoothness-regularized and shares the loss with envelope/bin/modulation
  terms; the mean fitted EQ sits at −1.87 dB at 1 kHz relative to its own
  maximum, i.e. it is near-flat through 400–2000 Hz and adds no mid lift. The
  EQ recovers ~6 dB of the 13.65, leaving 7.60 dB (C − R at 1 kHz).
- The **rest of the chain recovers ~3 dB more** (drive/AGC/bed/post-effects/
  residual: C − R = −7.60 → S − R = −4.69) and stops there.
- The **band-edge filters are not the cause**: the LP at 3.8 kHz moves 1/2/3 kHz
  by ≤ 0.1 dB paired on identical clips (`prod-fid-rerun.audit.md` item 7), and
  the same −4.6 dB existed in the pre-LP render.

**Lever chosen:** a fixed corrective peaking EQ as the last channel step. It is
bounded (one biquad, three constants), invertible (delete the step), testable
offline before it ships, and it does not touch calibration, the residual, the
talker, or any frozen value. The alternative — retuning upstream (refitting
presets, changing the probe corpus, or moving talker values) — is a multi-day
recalibration that also re-opens D1/D2, and is rejected for this correction.
Placing the bell on the source instead of the output is more physical but Mode 2
has no pre-channel chain hook; parked below.

## 2. The one change

`configs/mode2_v1.yaml`, one appended chain step (the low-pass stays exactly as
shipped; nothing else in the file changes):

```yaml
  post_effects:
    chain:
      - primitive: lowpass
        cutoff_hz: 3800
        order: 8
        zero_phase: true
      - primitive: peaking_eq          # attempt 1
        prob: 1.0
        f0_hz: 1100
        gain_db: 7.0
        q: 1.7
```

Supporting code (inert unless a config names it, same pattern as the shipped
`lowpass`): register one new primitive in `atcgen/channel/primitives.py`

```python
def peaking_eq(x: np.ndarray, sr: int, rng: random.Random, f0_hz: float = 1000.0,
               gain_db: float = 0.0, q: float = 1.0) -> np.ndarray:
    sos = np.array([_peaking_sos(sr, f0_hz, gain_db, q)])
    return signal.sosfilt(sos, x).astype(np.float32)
```

and add `"peaking_eq": peaking_eq` to `PRIMITIVES`. Single pass (minimum
phase), so `gain_db` is the realized gain at `f0_hz`; `zero_phase` would double
the dB curve and is not offered. `mic_coloration` cannot be used instead: its
peak gain is drawn `uniform(−peak_gain_db, +peak_gain_db)`, so it cannot deliver
a fixed +7.0 dB. New test in `tests/test_primitives.py`: the response at
`f0_hz` is +`gain_db` ± 0.2 dB, |response| ≤ 0.5 dB at 0.1 × and 3 × `f0_hz` for
these parameters, and the primitive consumes no RNG draw.

## 3. Sizing, from the measured gaps

Sized offline on the same 150 rendered clips (`runs/prod_fid_d3prime/wavs`) by
applying the exact biquad and recomputing the real-cohort gaps
(`runs/prod_fid_d3prime/eq_probe.py`; CPU-only, read-only on the render).
Because both LTAS curves are peak-normalized at ~484 Hz, a bell near 1 kHz also
lifts the normalizing peak: the measured efficiency is **0.54 dB of 1 kHz gap
closed per dB of design gain**, so the design gain is ~1.9× the 4.69 dB gap.
Measured sensitivities per dB of `gain_db` (f0 = 1100 Hz, Q = 1.7, below the
peak-migration point): **1 kHz +0.54, 2 kHz +0.02, 3 kHz −0.12**.

Design grid, real-cohort gaps after the bell (in-band = max |gap| at 1/2/3 kHz):

| f0 / gain / Q | 100 | 400 | 1k | 2k | 3k | 4k | in-band | peak Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none (baseline) | +7.94 | −0.45 | −4.69 | −1.09 | +0.02 | −6.55 | **4.69** | 484.4 |
| 1000 / +4.7 / 1.4 (naive: gain = gap) | +7.04 | −0.76 | −2.22 | −1.36 | −0.86 | −7.58 | 2.22 | 484.4 |
| 1000 / +6.0 / 1.4 | +6.78 | −0.84 | −1.53 | −1.43 | −1.11 | −7.89 | 1.53 | 484.4 |
| 1100 / +6.5 / 1.7 | +7.13 | −0.68 | −1.27 | −1.05 | −0.78 | −7.54 | 1.27 | 484.4 |
| **1100 / +7.0 / 1.7 (chosen)** | **+7.07** | **−0.70** | **−1.00** | **−1.04** | **−0.84** | **−7.63** | **1.04** | **484.4** |
| 1100 / +7.5 / 1.7 | +6.82 | −0.90 | −0.92 | −1.22 | −1.09 | −7.90 | 1.22 | 953.1 |
| 500 / −4.0 / 1.4 (cut the peak instead) | +10.58 | +0.03 | −2.88 | +1.59 | +2.73 | −3.83 | 2.88 | 359.4 |

Choices, each traceable to a measured number: **f0 = 1100 Hz** is the centre
that minimizes the measured in-band maximum over 1000/1100/1200 Hz (the deficit
spans ~600–1400 Hz, not a single bin). **Q = 1.7** ≈ the deficit's ~1-octave
width; wider drags 2/3 kHz down, narrower leaves the shoulder short.
**gain = +7.0 dB** equalizes the three in-band residuals at ~1.0 dB rather than
zeroing 1 kHz alone, and stays 0.5 dB below the gain at which the synthetic
LTAS peak migrates from 484 Hz to 953 Hz — above that point the measured
in-band maximum rises again (1.22 dB at +7.5, 1.40 dB at +8.0), so more gain
buys nothing. Cutting near the peak instead of boosting near 1 kHz is
rejected by measurement: it doubles the 100 Hz excess and pushes 3 kHz positive.

Expected on a fresh render: in-band ≈ 1.0 dB, i.e. ~1.0 dB of margin against a
2.0 dB gate, on a cohort-to-cohort wander of ±0.6 dB (audit-measured across two
same-recipe renders). The design is fitted to the seed-0 cohort's measured
gaps; the confirmation render is a **fresh cohort** — each chain step consumes
RNG draws, so per-clip voice/augment draws shift and the attempt render is not
paired with `prod_fid_d3prime`. These predictions must not alter the rule.

## 4. D3″ pre-registration

Candidate: fresh 150-clip seed-0 fidelity render under `runs/prod_fid_d3pp_a<k>`
with the frozen runbook §5 command and only the §2 change. No arms, no search.

**D3″ PASS if and only if all three limbs hold:**

- **(a) LTAS.** `max |LTAS_syn(f) − LTAS_real_cohort(f)| ≤ 2.0 dB` for
  `f ∈ {1000, 2000, 3000} Hz`, measured against the fixed real target cohort
  `runs/calib_v2/clips` with `--limit 1000 --cohort-reference real`. Identical
  to D3′.
- **(b) KID availability.** Primary WavLM KID computed on energy-trimmed,
  −26 dB RMS-normalized audio against the fixed seed-0 997-clip reference
  subset, reported as KID ± SE with clip counts. Identical to D3′.
- **(c) KID guardrail (new).** `KID_wavlm ≤ 0.004134 + 2 × 0.000797 = 0.005728`,
  on that byte-identical reference set (aggregate hash `b50c7b33046cc271da30`,
  n = 997; verify before quoting). One-sided: improvement is unbounded, and no
  other KID threshold is introduced after seeing a result. The guardrail exists
  so the EQ cannot buy LTAS by wrecking realism — it also catches the known side
  effect that a post-chain bell colours the real noise bed as well as the speech.

**Report but do not gate:** LTAS gaps at 100/200/300/400/3400/4000 Hz, all
hardcoded-reference gaps, matched CLAP KID ± SE, clip counts, both peak
frequencies (note if the synthetic peak migrates off 484 Hz), and any clipping
or QC-retry warnings in the render log.

Budget per attempt: one 150-clip render cell (~60 s), one matched-set build
(CPU), one matched-KID evaluation (~124 s), one LTAS run; ≤ 30 min wall-clock.
Every GPU step over a minute goes through `scripts/lab/jobs.py launch --gpu`,
**including `embed_dist`** (the `prod-fid-rerun` audit flagged that one running
outside the lock). Fresh `--out` per attempt; no directory is reused.

## 5. Attempt budget: 2, same lever only

**Attempt 1** = §2 exactly: `f0_hz 1100, gain_db 7.0, q 1.7`, out
`runs/prod_fid_d3pp_a1`, job ids `prod-v1-fix-a1` / `prod-v1-fix-a1-kid`.

**Attempt 2** = one resize of the same bell. `f0_hz` and `q` do not move; only
`gain_db` changes, to `G₂ = 7.0 + Δ`, `Δ` rounded to 0.1 dB and clamped to
`[−3.0, +2.0]` (so `G₂ ∈ [4.0, 9.0]`). Out `runs/prod_fid_d3pp_a2`, job ids
`prod-v1-fix-a2` / `prod-v1-fix-a2-kid`. `Δ` is computed from attempt 1's own
measured real-cohort gaps `r_1k, r_2k, r_3k` (syn − real, dB) and the sensitivities
`s = (+0.54, +0.02, −0.12)` dB per dB of gain:

> **Δ = argmin over Δ of max( |r_1k + 0.54 Δ|, |r_2k + 0.02 Δ|, |r_3k − 0.12 Δ| )**

a one-dimensional minimum of three linear functions — evaluate the pairwise
crossings and the two endpoints, take the smallest maximum. Closed forms for
the two expected regimes: if 1 kHz and 3 kHz are both still deficient,
`Δ = (r_3k − r_1k) / 0.66`; if only the 1 kHz limb misses, `Δ = −r_1k / 0.54`.

Three pre-registered exceptions, so attempt 2 is never spent on a miss it cannot
fix:

1. **2 kHz is the only binding limb** (`|r_2k| > 2.0` and 1/3 kHz pass): the
   bell cannot move 2 kHz (`s_2k ≈ 0`). Skip attempt 2, go to §6.
2. **LTAS passes but the KID guardrail fails**: attempt 2 is `Δ = −1.5`
   (`gain_db 5.5`), the mildest resize whose predicted in-band maximum
   (≈ 1.8 dB) still clears 2.0 dB. No other parameter changes.
3. **Both limbs fail**: apply the formula for the LTAS limb; if the resulting
   `|Δ| > 2.0` in the direction that increases colouration while KID has already
   failed, skip attempt 2 and go to §6.

No third lever, in any branch: no second EQ step, no high-pass, no change to
`f0_hz`/`q`, no calibration or residual change, no reference change, no
relaxation of the 2.0 dB limit or of the KID guardrail.

## 6. Kill / fallback (pre-authorized by Kevin, no reply needed)

If attempt 2 fails D3″ (or an exception above skips it): **revert the
`peaking_eq` step** from `configs/mode2_v1.yaml`, leaving the frozen recipe plus
the 3.8 kHz low-pass, and render V1.0.0 from that. The report records the
deficit as an open, quantified limitation (final gaps at all reported bands,
matched KID ± SE, the mechanism section above), and `docs/results.md` carries
the same line in the addendum. **Runbook §3 starts in every branch** — PASS
with the bell, or FAIL without it. `data/real/kixd/kixd_locked_day.csv` is not
read in any branch.

## 7. Artifacts and exact commands

Preflight (CPU): `uv run pytest -q` (expect the current count plus the new
primitive test), then confirm the resolved chain:

```powershell
uv run python -c "from atcgen.config import load_config; c=load_config('configs/mode2_v1.yaml'); print([(s.primitive, s.prob, {k: v.as_dict() for k, v in s.params.items()}) for s in c.calibrated.post_effects.chain])"
```

Per attempt `<k>` (`OUT = runs/prod_fid_d3pp_a<k>`):

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-v1-fix-a<k> -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 150 --out runs/prod_fid_d3pp_a<k> --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set dataset.noise_only_frac=0
uv run python scripts/analysis/make_matched_sets.py --out runs/prod_fid_d3pp_a<k>/kidsets --real-dir runs/calib_v2/clips --syn v1=runs/prod_fid_d3pp_a<k>/wavs
uv run python scripts/lab/jobs.py launch --gpu --id prod-v1-fix-a<k>-kid -- uv run python -m atcgen.eval.embed_dist runs/prod_fid_d3pp_a<k>/kidsets/v1_matched runs/prod_fid_d3pp_a<k>/kidsets/ref_matched --device cuda --out runs/prod_fid_d3pp_a<k>/kidsets/kid_v1_matched.json
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid_d3pp_a<k>/wavs --label real --label v1 --limit 1000 --cohort-reference real --json runs/prod_fid_d3pp_a<k>/ltas.json
```

Artifacts that prove it ran: `lab/jobs/prod-v1-fix-a<k>*/status.json` exit 0;
`manifest.jsonl` with 150 rows and `peaking_eq` in 150/150 per-clip chains;
`config.resolved.yaml` showing both chain steps and every frozen value unchanged
(`tts.speed [1.0, 1.4]`, `pitch 0.5`, `tempo 0.3`, `eq_tilt 0.4`, six-station
`station_mix`, residual on `runs/fastcut_v1/G_selected.pt` at `apply_prob 0.5`,
`seed 0`); `ltas.json` with both curves and direct gaps; `matched_sets.txt`;
`kid_v1_matched.json` with counts and SE. Report: `lab/reports/prod-v1-recipe-fix.md`,
audited before §3 starts.

Sizing/diagnostic scratch (CPU-only, gitignored, read-only on the render, kept
for the auditor): `runs/prod_fid_d3prime/eq_probe.py`,
`runs/prod_fid_d3prime/mechanism_probe.py`.

## Ideas (unscheduled)

- **Source-side placement of the same bell** (correct the talker before the
  channel, so the fitted EQ and the noise bed stay uncoloured): parked — Mode 2
  has no pre-channel chain hook, and adding one is a second change inside this
  correction. First candidate for V1.1 if the KID guardrail binds.
- **Probe-corpus mismatch at calibration time**: `channel_fit` used
  `--probe-dir runs/gan_a_base_v1/clean`; whether that corpus's talker
  distribution matches production (speed/pitch/tilt draws) is measurable and
  would explain part of the residual. Parked: it implies a refit, not a fix.
- **Fit-side lift**: relax `SMOOTH_REG` or reweight the `ltas` term in
  `channel_fit` so the fitted EQ inverts more of the source deficit. Parked:
  re-opens D1 and every preset.
- **`residual_scale_max: 0.35` in `configs/mode2_v1.yaml`** vs the 0.20 cap
  recorded as frozen in runbook §2 / the generator-config skill. Audit question
  for the director, deliberately **not** part of this spec's one change.
- **Per-station LTAS**: the aggregate gap may be one station's; measurable from
  `runs/calib_v2/clips` prefixes. Parked: cannot change a single global recipe.
- **150 Hz high-pass for the +7.9 dB at 100 Hz**: parked — reported-not-gated,
  and it overcorrected the low edge in the B3 table.
