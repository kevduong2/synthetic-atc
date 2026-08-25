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
