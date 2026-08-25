# Training and evaluation

This document describes the current student-training and evaluation paths. The
pipeline context is in [architecture.md](architecture.md), audio production is
in [generation.md](generation.md), and the verification gate is in
[gate.md](gate.md). Command details are collected in
[cli-reference.md](cli-reference.md); the illustrated overview is
[systems-manual.html](systems-manual.html).

## Student training

`training/recipe.py` is the staged L2 student recipe. It has four arms:

- `real_only`: SFT on the selected real pool.
- `synth_only`: SFT on synthetic manifests.
- `mix`: SFT on a seeded real/synthetic mixture.
- `mix_grpo`: the same SFT mixture, followed by GRPO.

`RecipeConfig` defaults to `openai/whisper-tiny.en`, `mix`, 500 SFT
optimizer steps, batch size 8, learning rate `1e-5`, and mix ratio `0.75`
(real fraction). Its GRPO defaults are 300 steps, batch 4, group 6,
temperature `0.9`, learning rate `1e-6`, and KL weight `0.04`.

The arm-specific input checks are part of the CLI. `real_only`, `mix`, and
`mix_grpo` require `--real-indices LO:HI`; `synth_only`, `mix`, and `mix_grpo`
require at least one `--synth-manifest`. A synthetic manifest argument can be
repeated. `mixed_pool()` clamps `--mix-ratio` to `[0, 1]`, subsamples the
binding side with a seeded NumPy permutation, and shuffles the combined pool.

The recipe CLI flags are:

- `--arm` (`real_only`, `synth_only`, `mix`, or `mix_grpo`; default `mix`) and
  required `--out`.
- `--model` (default `openai/whisper-tiny.en`).
- `--real-corpus` (default `jacktol/atc-dataset`), `--real-split` (default
  `train`), and `--real-indices`.
- Repeatable `--synth-manifest` and `--mix-ratio` (default `0.75`).
- Repeatable `--dev-manifest`, `--dev-split`, `--dev-indices`, and
  `--dev-batch` (default 8).
- `--sft-steps` (500), `--sft-batch` (8), and `--sft-lr` (`1e-5`).
- `--grpo-steps` (300), `--grpo-batch` (4), `--grpo-group` (6),
  `--grpo-lr` (`1e-6`), `--grpo-beta` (`0.04`),
  `--grpo-temperature` (`0.9`), and `--grpo-eval-every` (50).
- `--w-rep` (`0.5`), `--w-len` (`0.3`), and `--w-hal` (1.0).
- `--seed` (0) and `--device` (auto-detected when omitted).

Budget matching is explicit. Every arm runs the same `--sft-steps`,
`--sft-batch`, and `--sft-lr`; `mix_grpo` differs by adding its GRPO stage,
not by receiving extra SFT optimizer updates. The experiment matrix passes
the same SFT budget to all arms. Its current defaults are 2,000 SFT steps,
batch 8, and learning rate `1e-5`; the matrix also passes 600 GRPO steps and
learning rate `2e-6` to `mix_grpo`.

Each SFT stage saves a `save_pretrained` checkpoint and processor. The recipe
writes `<out>/run.json` with the arm, resolved config, device, seed, final
checkpoint, stage records, pool sizes, step counts, batch sizes, learning
rates, samples seen, loss summaries, and wall times. A GRPO stage records its
best and last checkpoints as well. Checkpoints can be passed to
`training/evaluate.py --model`.

## Lightweight Whisper fine-tuning

`atcgen/rl/finetune_lite.py` is the fixed-step fine-tuner reused by the recipe,
the outer reward harness, and the L3 counterfactual. It does not use the HF
`Trainer`.

`prepare_features(dataset, processor)` expects each HF-style row to contain
`{"audio": {"array": ..., "sampling_rate": ...}, "text": ...}`. It runs
the processor's feature extractor and tokenizer and returns feature dicts with
`input_features` and `labels`. Whisper's feature extractor produces a fixed
`(80, 3000)` log-mel array after padding or truncating the audio to 30 seconds;
the collator therefore pads only labels with `-100` and removes a common
leading decoder-start token.

`finetune()` uses all model parameters with AdamW for exactly `steps` optimizer
steps. A seeded NumPy generator cycles through fresh permutations of the
feature list. The learning rate warms linearly for `max(1, int(0.1 * steps))`
steps and then stays constant; gradients are clipped to norm 1.0. Per-step
losses are stored on `model._ft_losses`.

`transcribe()` takes pre-extracted log-mel features, either the feature dicts
from `prepare_features()` or raw `(n_mels, 3000)` arrays. It batches them,
calls greedy `model.generate()` with `max_new_tokens=100` by default, and
decodes with `processor.batch_decode(..., skip_special_tokens=True)`.

The fine-tune is full. Whisper's log-mel encoder is structurally incompatible
with the HF PEFT LoRA path used by the research recipe, so this code passes the
complete model parameter set to AdamW rather than inserting a LoRA adapter.
The frozen copy used by GRPO is a separate full model.

`training/finetune_whisper.py` is the other current full-fine-tuning path. It
uses `Seq2SeqTrainer` and `Seq2SeqTrainingArguments`, supports epochs or a
per-phase `--max-steps`, and materializes prepared datasets from synthetic
manifests and/or the public real train split. Its regimes are synthetic-only,
`--real-only`, approximately one-to-one `--mix-real`, and sequential
`--curriculum` (synthetic phase followed by real phase). Its defaults are
`openai/whisper-small.en`, three epochs, batch size 8, learning rate `1e-5`,
warmup 100 steps, seed 42, and no FP16 unless `--fp16` is requested on CUDA.
It is a Trainer-based regime runner; `training/recipe.py` instead uses the
seeded fixed-step path above so all matrix arms share one budget.

## Student GRPO

`training/grpo.py` starts from an SFT checkpoint and samples a group of
hypotheses for every training clip. `GRPOConfig` and its CLI defaults are:

| setting | default |
| --- | ---: |
| `steps` | 300 |
| `batch` | 4 clips per update |
| `group` | 6 hypotheses per clip |
| `temperature` | 0.9 |
| `lr` | `1e-6` |
| `beta` | 0.04 |
| `grad_clip` | 1.0 |
| `max_new_tokens` | 64 |
| `eval_every` | 50 |
| `dev_batch` | 8 |
| `dev_max_new_tokens` | 100 |
| `seed` | 0 |

The reward is computed by `score_hypothesis()` after `normalize_atc()`:

```text
total = -(WER + w_rep * repetition + w_len * length)
```

The default weights are `w_rep=0.5`, `w_len=0.3`, and `w_hal=1.0`. WER is
capped at `wer_clip=2.0`. Repetition is the larger of duplicated 4-gram
fraction and a single-token-loop score. Length is word-count deviation beyond
the `len_tol=0.3` band, clipped at `len_clip=2.0`. For an empty reference,
WER, repetition, and length are not scored: a non-empty hypothesis receives
the hallucination penalty and an empty one receives zero.

`group_advantages()` normalizes rewards relative to each clip's group mean and
standard deviation. A group with zero spread is masked out rather than turning
floating-point noise into a full update. The policy loss uses the retained
group-relative advantages; the KL term is added separately.

The reference is a frozen copy of the initial SFT checkpoint. `sequence_kl()`
computes mean token-level `KL(policy || reference)` on the scored tokens, and
the update is `policy_loss + beta * KL`. `run_grpo()` writes the scalar and
component metrics as one JSON object per step to `metrics.jsonl`, writes
`best` checkpoints on improved dev WER, writes `last`, and records the run in
`run.json`. The entry point is `run_grpo`; the module CLI invokes it after
checking that a manifest or real split was supplied.

The decoder prompt is a Whisper-specific trap. `generate()` runs Whisper's
forced decoder prompt but its short-form return strips that prompt. Directly
teacher-forcing those returned IDs shifts every scored token by one position.
`decoder_prompt_ids()` reconstructs the start and forced IDs, and
`ensure_decoder_prompt()` puts them back before `prefix_length()` and
`score_sequences()` run. The `logp_token_mean` value in `metrics.jsonl` is the
canary: prompt misalignment drives it toward the roughly uniform-vocabulary
log probability (about `-10.9` in the code comment) while other metrics can
still look plausible.

The encoder is run once per input clip. Its hidden states are expanded with
`repeat_interleave(group, dim=0)` for the group scoring pass. The module's
performance note reports about a six-times policy-update speedup at group 6
relative to repeating the encoder work for each rollout. Sampling and scoring
use `model.eval()` so dropout does not change the on-policy distribution.

## Normalization and the evaluation CLI

`training/normalize.py` provides the frozen `normalize_atc()` used by WER and
reward code. It lowercases, turns hyphens and punctuation into spaces,
expands digit strings into digit words, and folds `niner/tree/fife/fower`,
`juliett`, `xray`, and `okay` to their comparison spellings. Keeping this
function unchanged across arms makes the WER comparison meaningful.

`training/evaluate.py` accepts a model ID or checkpoint and either a named
registry split or a manifest-backed dataset. The main flags are:

- `--model` (default `openai/whisper-small.en`), `--device`, and
  `--batch-size` (default 8).
- `--split-name` from `SPLIT_NAMES`, or `--dataset` (`real` means the public
  test split, otherwise a manifest path); these are mutually exclusive.
- `--report-out`, `--max-samples`, `--max-examples` (default 5), and
  `--hyps-out`.

The JSON report has `schema_version` 2, model/dataset/split provenance,
sample counts, `wer`, `entities`, `per_category`, `slices`, `callsign`, and
`hallucination` panels. `wer` contains both `raw` and `atc_normalized` WER.
The normalized pair also contains substitutions, deletions, insertions,
hits, and reference-word counts. WER excludes empty-reference rows; those
rows feed the hallucination panel instead. `slices.duration` uses `<3s`,
`3-6s`, and `>6s` bands, and `per_category` scores non-empty references by
their dataset category.

The `entities` panel comes from `entity_panel()`: exact callsign accuracy,
overall and per-type slot precision/recall/F1, and critical substitution
rate, plus counts and examples. Synthetic manifest entity labels are used
when present; otherwise the reference is parsed. See
[results.md](results.md) for how reports are summarized.

`--hyps-out` writes one JSON object per utterance with `index`, `reference`,
and `hypothesis`. `scripts/run_matrix.py` and `scripts/rl_verify.py` use
these paired files with `atcgen.rl.stats.paired_bootstrap()`.

## The experiment matrix

`scripts/run_matrix.py` runs these stages in order:

1. `generate`: build `synth_pool/manifest.jsonl`.
2. `gate`: write `manifest_gated.jsonl`.
3. `select`: retain gold, silver, and adversarial rows with a 5% adversarial
   cap in `manifest_selected.jsonl`.
4. `arms`: train `a1_real`, `a2_synth_gated`, `a2u_synth_ungated`, `a3_mix`,
   and `a4_mix_grpo`.
5. Development evaluations on `model_select`.
6. Locked evaluations on `locked_test`, unless `--skip-final` is set.

Stages are resumable by artifact. Generation is skipped when the manifest
exists with at least the requested row count; gate, select, arm, and eval
stages are skipped when their output artifact exists. Arm output is
`arms/<name>/run.json`; eval output is `eval/<tag>_<split>.json` plus the
paired hypotheses JSONL.

The matrix CLI exposes `--arm-workers` (default 2) and `--eval-workers`
(default 3) for concurrent subprocesses. The default of two arm workers is
intended to be safe on MPS; raising `--arm-workers` above two can exhaust MPS
memory and panic the machine on unified-memory Macs. See
[known-issues.md](known-issues.md) before running a matrix.
Other matrix controls include `--out`, `--model`, `--config`, `--text`,
`--n-synth`, `--gen-seed`, `--sft-steps`, `--sft-batch`, `--sft-lr`,
`--mix-ratio`, `--grpo-steps`, `--grpo-lr`, `--seed`, `--arms`, and
`--skip-final`.

`summarize()` extracts normalized/raw WER, S/D/I, callsign accuracy, entity
F1/recall, and critical substitution rate. Its verdicts compare synthetic
against real, mix against real, GRPO against SFT, and gated against ungated
synthetic. Its paired comparisons are `a3_vs_a1`, `a4_vs_a3`, and
`a2_vs_a2u`, computed from the per-utterance hypothesis files.

## Evaluation platform

The channel regression platform is composed from the following modules:

- `atcgen.eval.entity_metrics.entity_panel()` scores callsign accuracy,
  per-type slot F1, critical substitution rate, and structured error examples.
- `atcgen.eval.channel_stats` computes duration, RMS, peak, spectral upper and
  lower edges, frame-energy SNR, 2--8 Hz modulation energy, and a 32-band
  log-grid LTAS. `compare()` adds Wasserstein distances, real p10--p90 range
  checks, and LTAS L1 distance.
- `atcgen.eval.embed_dist` embeds with WavLM or CLAP. KID is the primary
  distance: an unbiased polynomial-kernel MMD averaged over random subsets.
  Frechet distance is also reported and marked unreliable when the set is not
  more than twice the embedding dimension.
- `atcgen.eval.probe` trains a k-fold logistic probe, optionally an MLP, to
  distinguish frozen WavLM real and synthetic embeddings. It reports balanced
  accuracy and a real-vs-real `null_control` floor. The current small
  calibration set saturates upward: [known-issues.md](known-issues.md) records
  0.98--1.00 real-vs-synthetic accuracy even for clean ungated TTS, so probe
  numbers are a trend here; KID and LTAS L1 are the iteration metrics.
- `atcgen.eval.qc` is Tier 0. `qc_sample()` checks finite audio, clipping,
  silence, duration 0.5--30 seconds, RMS -40 to -8 dB, and, when a reference
  exists, normalized round-trip WER at most 0.5. `QCTally` records discard
  reasons and rates. Noise-only rows skip the ASR round-trip check.
- `atcgen.eval.harness` composes Tier 0 recorded QC, Tier 1 channel statistics
  and embeddings, and Tier 2 probe results. Its verdict gates discard rate
  below 0.15, every scalar synthetic median inside the real p10--p90 range,
  and probe balanced accuracy at most 0.65; Tier 3 downstream WER is reported
  as not covered by this generator regression harness.
- `atcgen.eval.report` writes the one-page HTML report with QC, scalar
  distributions, LTAS plots, comparison tables, and a fixed audition list.

## Data discipline

`atcgen.dataset.splits.SPLITS` is the registry for
`jacktol/atc-dataset`:

| name | slice | purpose |
| --- | --- | --- |
| `real_train` | `train[0:8000]` | generator vocabulary, SFT arms, gate calibration |
| `reward_val` | `train[8000:9000]` | RL and bandit reward development |
| `model_select` | `train[9000:10000]` | checkpoint and arm selection |
| `train_tail` | `train[10000:]` | unassigned reserve |
| `locked_test` | `test[500:2500]` | final reports, one read per arm |
| `spent_test` | `test[0:500]` | historical RL verification; burned |

The registry uses half-open ranges. `_check_disjoint()` runs at import time
and raises if any two entries overlap within the same source split. The
`locked_test` policy is touch-once: it is read only for final per-arm reports,
never for tuning. `reward_val` and `model_select` are already spent by their
respective loops and are not headline metrics.

The PoC corpus is utterance-segmented. Index-disjoint train/test slices can
still share speakers, sectors, or callsigns, so they are not speaker-disjoint;
this is a known PoC caveat and is prohibited for production split design. See
[known-issues.md](known-issues.md) and [research-findings.md](research-findings.md).
