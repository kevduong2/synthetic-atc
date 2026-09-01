# Experimental results

This is the in-repo snapshot of experimental evidence as of 2026-08-25.
The run directories under `runs/` are gitignored, so this document is the
only in-repo record of the snapshot. The pipeline and split rules are described
in [training-and-eval.md](training-and-eval.md),
[gate.md](gate.md), and [rl-loops.md](rl-loops.md).

## Setup

The student was `openai/whisper-tiny.en`. The proof-of-concept real corpus was
`jacktol/atc-dataset`. The split registry assigns these slices:

| Name | Slice | Use |
|---|---|---|
| `real_train` | `train[0:8000]` | Real training and upstream anchor data. |
| `reward_val` | `train[8000:9000]` | RL and bandit reward validation. |
| `model_select` | `train[9000:10000]` | Development model selection. |
| `locked_test` | `test[500:2500]` | Final matrix report. |
| `spent_test` | `test[0:500]` | Historical RL verification; spent. |

The matrix configuration in `runs/matrix_v1/matrix_config.json` used 8,000
synthetic clips from `configs/mode1_matched.yaml`,
`grammar:region=eu`, 2,000 SFT steps with batch 8 and learning rate `1e-5`,
a real mix ratio of `0.75`, 600 GRPO steps with learning rate `2e-6`, and
seed 0. Its generation seed was 101, and it ran arms A1, A2, A2u, A3, and A4.

## Matrix v1 locked-test results

The following values are from `runs/matrix_v1/summary_locked_test.json`. WER
and the three rates are shown as percentages; S/D/I are normalized error
counts.

| Arm | Normalized WER | S / D / I | Callsign accuracy | Entity F1 | Critical substitution rate |
|---|---:|---:|---:|---:|---:|
| A0 zero-shot | 133.12% | 12,261 / 2,975 / 12,020 | 0.99% | 6.16% | 1.62% |
| A1 real | 22.40% | 2,429 / 902 / 1,256 | 62.86% | 77.76% | 11.37% |
| A2 gated synthetic | 57.28% | 7,270 / 2,696 / 1,762 | 14.47% | 35.59% | 19.45% |
| A2u ungated synthetic | 62.68% | 8,378 / 1,787 / 2,668 | 13.15% | 29.74% | 25.43% |
| A3 mix SFT | 22.82% | 2,593 / 844 / 1,235 | 60.88% | 76.18% | 12.94% |
| A4 mix plus GRPO | 20.35% | 2,385 / 938 / 843 | 61.04% | 77.09% | 12.44% |

## Verdicts and paired bootstrap

`atcgen.rl.stats.paired_bootstrap` defines `delta` as
`WER_baseline - WER_challenger`. The positive direction therefore favors the
challenger. This is also the convention documented by `run_matrix.py`.

The stored summary uses these baseline/challenger pairs: A1/A3 for
`a3_vs_a1`, A3/A4 for `a4_vs_a3`, and A2u/A2 for `a2_vs_a2u`.

| Summary key | Baseline | Challenger | Delta (abs WER points) | 95% CI | p |
|---|---|---|---:|---:|---:|
| `a3_vs_a1` | A1 | A3 | -0.4152 | [-3.0333, 2.2336] | 0.738 |
| `a4_vs_a3` | A3 | A4 | 2.4714 | [1.1342, 4.2675] | 0.000 |
| `a2_vs_a2u` | A2u | A2 | 5.3971 | [3.3846, 7.3400] | 0.000 |

The headline A4-vs-A1 comparison was recomputed from
`runs/matrix_v1/eval/a1_real_locked_test_hyps.jsonl` and
`a4_mix_grpo_locked_test_hyps.jsonl` using the same call made by
`run_matrix.py`: 2,000 paired bootstrap resamples with seed 0. The stored
hypotheses were passed as raw strings; `wer_counts` applies
`training.normalize.normalize_atc`, as it does for the matrix comparison.
For 2,000 aligned utterances, the computed result is:

| Baseline | Challenger | Delta | 95% CI | p |
|---|---|---:|---:|---:|
| A1 WER `0.2240402462` | A4 WER `0.2034775813` | `0.0205626648` | `[0.0032154063, 0.0423340184]` | `0.012` |

That is a 2.0563-point A4 improvement, with a 0.3215--4.2334-point 95% CI.

## Gate yield on the 8k pool

`runs/matrix_v1/synth_pool/gate_stats.json` records 1,950 gold, 2,290
silver, 383 adversarial, and 3,377 rejected rows out of 8,000:

| Tier | Count | Fraction |
|---|---:|---:|
| Gold | 1,950 | 24.37% |
| Silver | 2,290 | 28.63% |
| Adversarial | 383 | 4.79% |
| Rejected | 3,377 | 42.21% |

Rounded for the matrix summary, the yield is 24% gold, 29% silver, 4.8%
adversarial, and 42% rejected. The gated manifest is
`runs/matrix_v1/synth_pool/manifest_gated.jsonl`.

## L3 bandit audit

The P4c exit bar was not met at PoC scale: hardness-selected synthetic data
lost to uniform sampling in the counterfactuals. In the naive design, the
first `runs/bandit_v2/counterfactuals.jsonl` row was `wer_init=0.19825`,
`wer_selected=0.23620`, and `wer_uniform=0.21149`, a selected-vs-uniform
delta of `-0.02472`, or -2.472 absolute WER points. The replay-corrected
artifact `runs/bandit_v2/counterfactual_replay.json` reports
`wer_init=0.1983`, selected `0.2202`, and uniform `0.2145`; its exact values
are `0.1982521848`, `0.2202247191`, and `0.2144818976`, with
`delta_selected_vs_uniform=-0.0057428215`, or -0.574 absolute points.

The window recalibration changed `tau2` from 0.40 in `runs/bandit_v1/state.json`
to 0.15 in `runs/bandit_v2/state.json`. The selected-buffer counts were 59 of
1,800 generated rows in v1 (3.3%) and 350 of 1,800 in v2 (19.4%). The audit
also records that the A4 student was saturated on this synthetic distribution:
median synthetic WER was 0.096 versus 0.561 for the frozen teacher. The
bandit's genuine result was diagnostic: the v2 posterior localized
`us_routine` as the EU-trained student's weak slice; its final state has
alpha 122 and beta 360 for that arm, a posterior mean of 0.2531.

The honest reading is that the counterfactual harness refused a dead proxy.
That is the system working, not evidence that hardness selection improved the
training distribution.

## Interpretation and open items

- The full recipe is the headline winner: A4 reaches 20.35% normalized WER
  versus 22.40% for real-only A1. GRPO also reduces insertions from 1,235 in
  A3 to 843 in A4.
- The gate earns its cost: gated A2 is 57.28% WER versus 62.68% for the
  otherwise matched ungated A2u, and the stored paired delta favors A2.
- Synthetic-only parity is not met. A2 trails A1 by 34.88 absolute WER
  points, far beyond the 1.5-point parity bar. The planned VC/accent branch
  and FM-TTS branch are deferred to the GPU wave; see
  [research-findings.md](research-findings.md) and
  [plans/research-integration.md](plans/research-integration.md).
- Mix-SFT is effectively tied with real-only at this 8k-real scale: A3 is
  22.82% versus A1 at 22.40%, consistent with the proposal's D10 real-data
  anchor.

The results are mirrored in the local MLBucket dashboard at
[http://localhost:8484](http://localhost:8484), under projects
`atcgan-rl-v1`, `atcgan-matrix-v1`, and `atcgan-bandit`.

## FastCUT go/no-go and fc_combo (2026-08-26)

The leakage repair passed its audit: 99 corpus clips, 85 `channel_train`
clips, 82 presets, and 127 noise-stat entries were checked, with no forbidden
source IDs and no mismatches. The channel split followed the plan's
development-only rule: `channel_train` and `channel_val` are grouped by
station plus capture time-block, and artifacts are derived from
`channel_train` only. There is still no `channel_test`, so this does not
support a receiver/session-generalization claim.

The G1-versus-G3 premise check was a tie: 68.55% for G1 procedural matched
versus 68.79% for G3 calibrated DSP. The paired S1 ablation selected
`M0/source+identity-NCE` over `M0/source-NCE`. S2 selected step 3500 under the
lexicographic rule: fold-KID was 0.003562 versus 0.004041 for DSP-only.

The ASR screen returned `conditional_go`, funding the full-scale FC matrix.
The seed-0 FC-A2-versus-FC-A4 result was flat at -0.13 points; seed 1 was
positive at +1.31 points [0.05, 2.94], p=0.036. The synth-only sensitive
instrument was +1.30 points, non-significant. G5 gate yield was 39.55% versus
39.45% for G3. The verdict recorded no material safety regression and noted
that noninferiority margins were not pre-frozen.

The scale-arm WERs were FC-A2 23.27%, FC-A3 22.73%, and FC-A4 24.64%; the
reported +GRPO values were 20.79% and 20.56%.

For `fc_combo`—75% `real_train` and 25% synthetic data, 2,000 SFT steps and
600 GRPO steps—the historical `locked_test` is now `spent_test_fastcut`:
one read per seed and development evidence. The paired results are below;
delta is baseline WER minus `fc_combo` WER, in percentage points.

| Seed | fc_combo WER | vs A4: delta / p | vs A1: delta / p |
|---:|---:|---:|---:|
| 0 | 19.87% | +0.47 / 0.482 | +2.53 / 0.004 |
| 1 | 19.51% | +0.84 / 0.1 | +2.89 / 0.0 |
| 2 | 20.61% | -0.26 / 0.727 | +1.80 / 0.136 |
| **Mean** | **20.00%** | — | — |

The baselines are A4 at 20.35% and A1 real-only at 22.40%. The
`heldout_tail_check` remains descriptive, with an approximately six-point
minimum detectable effect.

### Claim discipline

> Development/feasibility evidence. Seed-mean 20.00% beats A4's single-seed
> 20.35% (2 of 3 seeds below it); per-seed deltas vs A4 are inside this test's
> power, exactly as the plan's power analysis predicted for sub-point effects.
> The clean claim: fc_combo beats real-only SFT by 2.4 mean / 2.9 best abs WER
> (vs A4's 2.06) with no entity/callsign regression and record entity F1. A4
> itself is one seed; a 3-seed A4 replication would be needed for a fair
> seed-mean contrast. Confirmatory claims await a prospectively collected
> session-disjoint test.

## Overnight KIXD calibration + talker-knob session (2026-09-01)

The new real corpus contains 7,374 KIXD_TOWER clips from 2025-08-01 through
2025-08-08: 15.2 hours of 16 kHz mono audio. A rename-log join against ASR
V2.1.2 recovered 6,502 clean human transcripts. The day-disjoint reward split
uses 5,392 rows from days 1--6 for training and 200 rows from day 7 for
development. `data/real/kixd/kixd_locked_day.csv`, day 20250808 with 337 rows,
was deliberately never read. The decision retained `base`, so there was no
selection event; that day remains reserved for the final trained model.

The KIXD calibration chain produced 400 presets and 15,302 noise-bed segments,
76% of which passed the gate, using capture-block folds. Probe TTS required a
16 kHz resample; the correction and frozen workflow are recorded in
`docs/runbook-v1-3080.md`. Split and preset evidence is under
`runs/channel_data_kixd/`.

The reward is the `openai/whisper-tiny.en` fine-tune delta on KIXD development
data, using bounded per-row WER: each row's errors are capped at its reference
word count. This replaced unbounded WER after one looping row outweighed an
entire channel manipulation. Fine-tuning itself moved WER from about 0.75 to
about 0.50; the knobs below are second-order effects.

### Talker partition

The zero-shot bounded-WER baseline was 0.7334. There were four paired seeds per
talker arm versus `base`, except for the deliberately two-seed degraded arm.
Positive paired delta means that the arm hurts.

| Arm | Mean reward | SD | Paired delta vs base | Paired t | Direction |
|---|---:|---:|---:|---:|---:|
| `base` | +0.2217 | 0.0228 | — | — | — |
| `aug_off` | +0.1935 | 0.0143 | +0.0283 | 2.90 | 4/4 |
| `speed_fixed` | +0.2028 | 0.0227 | +0.0190 | 1.48 | 3/4 |
| `pitch_off` | +0.2025 | 0.0185 | +0.0192 | 1.76 | 3/4 |
| `voiceaug_off` | +0.2114 | 0.0031 | +0.0103 | 0.80 | 3/4 |
| `degraded` (2 seeds) | +0.2299 | 0.0129 | +0.0022 | 0.10 | 1/2 |

The four-seed t values have 3 degrees of freedom. The `aug_off` result puts the
combined talker-augmentation effect near 2.8 WER points, with all four paired
seeds in the same direction. The two-seed degraded null says this reward is
blind to channel quality at the 200-clip, 300-fine-tune-step, 200-row
development budget; channel choices therefore remain governed by KID and LTAS,
not a WER search.

The decomposition is almost exactly additive: `speed_fixed` +
`voiceaug_off` is +0.0293 versus +0.0283 for `aug_off`, a +0.0010 residual.
No individual component is separable at this budget; the component contrasts
are within approximately one standard error. The impossible ordering in which
`voiceaug_off` is cheaper than its own `pitch_off` subset is itself evidence
that the sub-effects are below the noise floor.

Talker augmentation therefore stays on at the baseline values: speed
`[1.0, 1.4]`, pitch probability 0.5, tempo probability 0.3, and EQ-tilt
probability 0.4. “Pitch off for the 2.6x KID gain” is rejected: the powered
follow-up (base and `pitch_off` extended to **10 paired seeds** after the
freeze, same harness) puts removing pitch at **+0.0089 bounded reward
(≈0.9 WER points), t=1.61 on 9 df, 8/10 seeds harmful, 95% CI −0.35 to
+2.1 points**. Not individually significant, but directionally consistent and
nowhere near "free"; with downstream WER as the product metric, pitch stays.
The KID-vs-WER tension is real and recorded — revisiting it requires a larger
dev slice, not more seeds.

`runs/power_check_kixd/summary.json` is mixed-metric: seeds 0--1 are unbounded
and later seeds are bounded. The table above instead comes from
`scripts/analysis/rescore_bounded.py` and
`scripts/analysis/paired_report.py`, recomputed on the
bounded metric from the 22 per-cell
`runs/power_check_kixd/trials/*/dev_rows.jsonl` files.

### KIXD fidelity and FastCUT smoke

Real KIXD LTAS peaked at 469 Hz and fell about 6 dB/octave from 1--3 kHz,
reproducing the research measurement within 2 dB. Mode 2's calibrated 1--3 kHz
LTAS gap was 1.4 dB, inside that measurement floor, so no shelf correction was
needed; mode 1's gap was 3.2 dB with a 200 Hz bulge. Both modes retained a
13--24 dB excess at 4 kHz, and mode 2 was 13.5 dB hot at 100 Hz. These are
post-calibration checks for the full run, not evidence that the residual model
will close them. The measurements are in
`runs/e1_artifacts/ltas_e1.json` and
`runs/e1_artifacts/ltas_mode1.json`; rendered Mode 2 is under
`runs/e1_mode2_kixd/`.

Reference clips contained 18.8% exact-zero samples and were 9 dB colder, a
padding/level confound in raw KID. Energy trimming and RMS matching on both
sides reduced KID by 35--42%. Matched KID was 0.00334 for mode 1 and 0.00299
for mode 2: mode 2 was ahead by about one standard deviation, consistent with
LTAS. Only matched KID is quoted as the fidelity metric from this point. The
evidence is in the session tmp directory's `kid2x2_mode1_matched.json` and
`kid2x2_mode2_matched.json`, produced by `make_matched_sets.py` in that same
directory.

The 300-step KIXD FastCUT smoke passed all 8 gates, wrote `G_selected.pt`, and
computed fold-KID at 0.64 steps/s under contention. This is an end-to-end input
validation, not a quality claim; its evidence is
`runs/fastcut_kixd_smoke/summary.json` and `validation_report.json`.

### Infrastructure and frozen production recipe

The session landed local CSV/JSONL development-corpus routing for the reward
harness and `rl_verify`, per-source WER and per-row dumps, bounded WER with
stale-cache invalidation, `talker_only`/`mode2_safe`/`default` search spaces
with CLI mode guards, and the normalizer x-ray fold fix. It also added
`SequentialTextSource` and `expand_text_views` for a 155,776-item paired
two-view schedule, plus 4,800 noise-only clips; the scene converter covers
77,888 utterances across six airports. The V2.1.2-schema CSV exporter supports
multi-run merge, keeps noise out of test, accepts `--set` overrides, and uses
1,024-sample length quantization. The suite had 761+ passing tests.

The v1 production recipe is Mode 2 calibrated plus a FastCUT residual using
source+identity NCE, a 0.20 scale cap, lexicographic selection, and
`G_selected.pt`, to be recalibrated on the full multi-airport set. Text is
rendered sequentially without replacement under the two-view policy. ASR
training uses a 50--75% real mix followed by a short real-only tail, never
synthetic-only. Licensing status is in `docs/data-licensing.md`; the complete
frozen procedure is in `docs/runbook-v1-3080.md`.

### Claim discipline

> Development and feasibility evidence only. The paired talker experiment
> supports retaining the complete baseline augmentation bundle, but does not
> identify a winning individual knob. The degraded null does not establish
> channel equivalence; it establishes that this WER reward and budget cannot
> select channel settings. Mode 2's LTAS and matched-KID advantage is a
> development fidelity result, and the FastCUT run is only a smoke test.
> Confirmatory model claims await the final trained model's single read of the
> untouched `kixd_locked_day`.

Primary run artifacts are `runs/power_check_kixd/` (22 cells),
`runs/e1_mode2_kixd/`, `runs/fastcut_kixd_smoke/`, and
`runs/channel_data_kixd/`.
