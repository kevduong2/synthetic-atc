# CLI reference

The entry points below are grouped by the stage they operate on. For the
pipeline context, see [architecture.md](architecture.md) and
[generation.md](generation.md). Training and evaluation conventions are in
[training-and-eval.md](training-and-eval.md); the outer loops are described in
[rl-loops.md](rl-loops.md).

## Generation

### `scripts/audition_presets.py`

Renders one phrase through fitted channel presets for listening.

| Flag | Default | Purpose |
|---|---:|---|
| `--presets` | `runs/calib_v2/presets.jsonl` | Fitted preset JSONL. |
| `--out` | `runs/audition_m22` | Output directory. |
| `--corpus` | `runs/calib_v1/corpus.jsonl` | Corpus manifest used to find each preset's real clip. |
| `-n`, `--count` | `10` | Maximum number of presets. |
| `--seed` | `0` | Python random seed. |

Example:

```bash
uv run python scripts/audition_presets.py \
    --presets runs/calib_v2/presets.jsonl --out runs/audition_m22
```

The command writes `00_clean.wav`, one degraded WAV per selected preset, any
matching `<preset>_REAL.wav` files found through `--corpus`, and `index.json`
under `--out`.

### `scripts/audition_primitives.py`

Renders a phrase through each channel primitive in isolation and through the
configured full chain.

| Flag | Default | Purpose |
|---|---:|---|
| `--config` | `configs/mode1_wide.yaml` | Channel profile. |
| `--out` | `runs/audition_p1` | Output directory. |
| `--seed` | `0` | Python random seed. |

Example:

```bash
uv run python scripts/audition_primitives.py \
    --config configs/mode1_wide.yaml --out runs/audition_p1
```

The command writes `00_clean.wav`, `01_noise_15db.wav`, the primitive
auditions, and `90_full_chain_1hop.wav` plus `90_full_chain_2hop.wav` under
`--out`. It prints the measured duration, RMS, and peak for each WAV; it does
not write a JSON index.

### `scripts/gen_profile_smoke.py`

Generates a small profile set and compares its channel statistics with a real
reference directory when that directory exists.

| Flag | Default | Purpose |
|---|---:|---|
| `--config` | required | Generator profile YAML. |
| `--out` | required | Output directory for smoke WAVs. |
| `-n`, `--count` | `100` | Number of clips. |
| `--seed` | `0` | Seed for text, TTS, and channel draws. |
| `--ref` | `data/real/calibration` | Real WAV directory used for comparison; comparison is skipped if absent. |
| `--stats-out` | `None` | Optional path for the full stats JSON. |
| `--tts-cache` | `None` | Optional clean-TTS cache; otherwise derived as `runs/p2_smoke/tts_<profile>`. |
| `--skip-generate` | `false` | Measure WAVs already in `--out`. |

Example:

```bash
uv run python scripts/gen_profile_smoke.py \
    --config configs/mode1_matched.yaml --out runs/p2_smoke/matched \
    --count 100 --ref data/real/calibration \
    --stats-out runs/p2_smoke/matched_stats.json
```

Generated files are `smoke_*.wav` under `--out`. The clean-TTS cache contains
its WAVs and `index.jsonl`. A JSON stats artifact is written only when
`--stats-out` is supplied; otherwise statistics and any comparison are printed
to stdout.

### `scripts/generate_dataset.py`

Builds a synthetic dataset from a profile and a grammar or JSONL text source.

| Flag | Default | Purpose |
|---|---:|---|
| `--config` | `configs/mode1_default.yaml` | Generator profile YAML. |
| `--n-samples` | required | Number of rows to generate. |
| `--out` | required | Dataset output directory. |
| `--seed` | `None` | Overrides the seed in the profile when supplied. |
| `--text` | `None` | Built-in grammar when omitted, or `grammar`/a JSONL text source. |

Example:

```bash
uv run python scripts/generate_dataset.py \
    --config configs/mode1_matched.yaml --n-samples 8000 \
    --out runs/matrix_v1/synth_pool --seed 101 --text grammar:region=eu
```

The output directory receives `wavs/000000.wav`-style files,
`manifest.jsonl`, `stats.json`, and `config.resolved.yaml`. The command prints
the manifest and stats paths.

### `scripts/harvest_vocab.py`

Harvests airline, station, and waypoint vocabulary from the first 8,000 real
training transcripts.

| Flag | Default | Purpose |
|---|---:|---|
| `--limit` | `8000` | Number of `train` rows to read; the script caps this at 8,000. |
| `--min-count` | `3` | Minimum count for a retained term. |
| `--out` | `data/vocab/real_anchor.json` | Vocabulary JSON path. |

Example:

```bash
uv run python scripts/harvest_vocab.py --limit 8000 \
    --min-count 3 --out data/vocab/real_anchor.json
```

The command writes the JSON object at `--out`, creating its parent directory,
and prints counts for the three vocabulary sections.

## Gate

The verification gate is described in [gate.md](gate.md).

### `scripts/gate_dataset.py`

Runs the frozen teacher panel over a built dataset and assigns every row a gate
tier.

| Flag | Default | Purpose |
|---|---:|---|
| `--dataset` | required | Built dataset directory containing `manifest.jsonl` and `wavs/`. |
| `--out` | `None` | Output directory; `--dataset` is used when omitted. |
| `--max-samples` | `None` | Optional row limit. |
| `--batch` | `8` | Clips per teacher call. |
| `--device` | `None` | `mps`, `cpu`, or `cuda`; `None` selects automatically. |
| `--quiet` | `false` | Suppresses the progress bar. |

Threshold flags are generated from `GateConfig`; when omitted, the effective
defaults are:

| Flag | Default |
|---|---:|
| `--gold-wer` | `0.25` |
| `--silver-wer` | `0.50` |
| `--adversarial-wer` | `0.90` |
| `--gold-critical-recall` | `0.50` |
| `--adversarial-critical-recall` | `0.50` |
| `--noise-max-words` | `2` |
| `--noise-requires-consensus` | `true` |
| `--repeat-threshold` | `0.80` |
| `--repeat-min-lag-sec` | `0.25` |
| `--repeat-frame-sec` | `0.02` |
| `--adversarial-cap` | `0.05` |
| `--min-duration` | `0.5` |
| `--max-duration` | `30.0` |
| `--max-clip-frac` | `0.01` |
| `--min-rms-db` | `-40.0` |
| `--max-rms-db` | `-8.0` |

Example:

```bash
uv run python scripts/gate_dataset.py --dataset runs/matrix_v1/synth_pool \
    --batch 8 --device mps
```

The command writes `manifest_gated.jsonl` and `gate_stats.json` under the
output directory. The gated manifest preserves every input row and adds its
tier and gate details; rows are not deleted.

## Evaluation

### `scripts/eval_synthetic.py`

Runs the synthetic-audio regression harness over channel statistics, optional
embeddings, and the channel probe.

| Argument or flag | Default | Purpose |
|---|---:|---|
| `run_dir` | required | Generated run directory. |
| `--ref` | required | Real calibration WAV directory. |
| `--out` | `runs/eval/versions` | Directory for version JSON artifacts. |
| `--skip-embeddings` | `false` | Skip embedding and probe tiers. |
| `--n-max` | `None` | Limit both WAV sets to their first N files. |
| `--wavs-only` | `false` | Treat `run_dir` as a bare WAV directory; Tier 0 is unavailable. |
| `--html` | `false` | Write an HTML report next to the version JSON. |
| `--diff OLD_JSON` | `None` | Print metric deltas against an older JSON report. |

Example:

```bash
uv run python scripts/eval_synthetic.py runs/matrix_v1/synth_pool \
    --ref data/real/calibration --skip-embeddings --html
```

The harness writes one timestamped JSON file under `--out`. With `--html`, it
also writes the matching HTML file beside that JSON. It prints the verdict and
the artifact paths.

### `training/evaluate.py`

Evaluates a Whisper checkpoint with normalized/raw WER, S/D/I, callsign and
entity metrics, and hallucination reporting.

| Flag | Default | Purpose |
|---|---:|---|
| `--model` | `openai/whisper-small.en` | Model ID or checkpoint directory. |
| `--split-name` | `None` | Named split from `atcgen.dataset.splits`; mutually exclusive with `--dataset`. |
| `--dataset` | `None` | `real` for the public test split, or a manifest path; if neither source flag is given, `real` is used. |
| `--report-out` | `None` | Optional complete report JSON path. |
| `--device` | `None` | Torch device; automatic selection when omitted. |
| `--batch-size` | `8` | Decode batch size. |
| `--max-samples` | `None` | Optional evaluation-row limit. |
| `--max-examples` | `5` | Worst-entity examples retained in the report. |
| `--hyps-out` | `None` | Optional per-utterance hypothesis JSONL path. |

Example:

```bash
uv run python training/evaluate.py --model openai/whisper-tiny.en \
    --split-name locked_test \
    --report-out runs/eval/a0_locked_test.json \
    --hyps-out runs/eval/a0_locked_test_hyps.jsonl
```

The full report is printed to stdout and is written to `--report-out` only
when that flag is supplied. `--hyps-out` receives one JSON object per evaluated
utterance with `index`, `reference`, and `hypothesis`.

## Training and matrix

### `training/recipe.py`

Runs one staged student arm: SFT on real, synthetic, or a real/synthetic mix,
with an optional GRPO stage for `mix_grpo`.

| Flag | Default | Purpose |
|---|---:|---|
| `--arm` | `mix` | `real_only`, `synth_only`, `mix`, or `mix_grpo`. |
| `--out` | required | Run directory. |
| `--model` | `openai/whisper-tiny.en` | Initial Whisper checkpoint. |
| `--real-corpus` | `jacktol/atc-dataset` | Real dataset. |
| `--real-split` | `train` | Real split name. |
| `--real-indices` | `None` | `LO:HI` slice of the real split. |
| `--synth-manifest` | `[]` | Repeatable synthetic dataset directory or manifest. |
| `--mix-ratio` | `0.75` | Real fraction of a mix. |
| `--dev-manifest` | `[]` | Repeatable development manifest. |
| `--dev-split` | `None` | Development split. |
| `--dev-indices` | `None` | `LO:HI` development slice. |
| `--dev-batch` | `8` | Development batch size. |
| `--sft-steps` | `500` | SFT optimizer steps. |
| `--sft-batch` | `8` | SFT batch size. |
| `--sft-lr` | `1e-05` | SFT learning rate. |
| `--grpo-steps` | `300` | GRPO steps for `mix_grpo`. |
| `--grpo-batch` | `4` | GRPO batch size. |
| `--grpo-group` | `6` | Hypotheses sampled per clip. |
| `--grpo-lr` | `1e-06` | GRPO learning rate. |
| `--grpo-beta` | `0.04` | KL weight. |
| `--grpo-temperature` | `0.9` | Sampling temperature. |
| `--grpo-eval-every` | `50` | GRPO development-evaluation cadence. |
| `--w-rep` / `--w-len` / `--w-hal` | `0.5` / `0.3` / `1.0` | Repetition, length, and hallucination reward weights. |
| `--seed` | `0` | Training and pool seed. |
| `--device` | `None` | Torch device; automatic selection when omitted. |

Example:

```bash
uv run python training/recipe.py --arm mix_grpo \
    --real-split train --real-indices 0:8000 \
    --synth-manifest runs/matrix_v1/synth_pool/manifest_selected.jsonl \
    --mix-ratio 0.75 --sft-steps 2000 --sft-batch 8 --sft-lr 1e-5 \
    --grpo-steps 600 --grpo-lr 2e-6 \
    --dev-split train --dev-indices 9000:9400 --out runs/matrix_v1/arms/a4_mix_grpo
```

The run writes the SFT checkpoint under `--out/sft`, the GRPO artifacts under
`--out/grpo` for `mix_grpo`, and `run.json` at `--out/run.json`. GRPO writes
`metrics.jsonl`, `last/`, and its own `run.json` under `--out/grpo`; when a
development pool is supplied, it also writes the selected `best/` checkpoint.

### `training/grpo.py`

Runs GRPO from an SFT checkpoint on one or more synthetic manifests and/or a
real split.

| Flag | Default | Purpose |
|---|---:|---|
| `--init` | `openai/whisper-tiny.en` | SFT checkpoint or model ID. |
| `--out` | required | GRPO run directory. |
| `--manifest` | `[]` | Repeatable synthetic dataset directory or manifest. |
| `--real-corpus` | `jacktol/atc-dataset` | Real dataset. |
| `--real-split` / `--real-indices` | `None` / `None` | Real split and `LO:HI` slice. |
| `--dev-manifest` | `[]` | Repeatable development manifest. |
| `--dev-split` / `--dev-indices` | `None` / `None` | Development split and slice. |
| `--steps` | `300` | GRPO optimizer steps. |
| `--batch` / `--group` | `4` / `6` | Clips per update and hypotheses per clip. |
| `--temperature` / `--max-new-tokens` | `0.9` / `64` | Sampling controls. |
| `--lr` / `--beta` | `1e-06` / `0.04` | Learning rate and KL weight. |
| `--grad-clip` | `1.0` | Gradient norm clip. |
| `--w-rep` / `--w-len` / `--w-hal` | `0.5` / `0.3` / `1.0` | Reward penalty weights. |
| `--wer-clip` / `--len-tol` | `2.0` / `0.3` | Reward limits. |
| `--eval-every` / `--dev-batch` / `--dev-max-new-tokens` | `50` / `8` / `100` | Development evaluation controls. |
| `--seed` | `0` | Training seed. |
| `--device` | `None` | Torch device; automatic selection when omitted. |

Example:

```bash
uv run python training/grpo.py --init runs/a3/sft \
    --manifest runs/matrix_v1/synth_pool/manifest_selected.jsonl \
    --real-split train --real-indices 0:8000 \
    --dev-split train --dev-indices 9000:9400 \
    --steps 600 --lr 2e-6 --out runs/a4/grpo
```

The run writes `metrics.jsonl`, `last/`, and `run.json` under `--out`; when a
development pool is supplied, it also writes the selected `best/` checkpoint.
The command also prints the run summary as JSON.

### `training/finetune_whisper.py`

Fine-tunes Whisper under the synthetic-only, real-only, mixed, or curriculum
training regimes.

| Flag | Default | Purpose |
|---|---:|---|
| `--manifest` | `[]` | Repeatable synthetic manifest. |
| `--real-manifest` | `[]` | Repeatable local real manifest; public real training data is used when omitted for a real regime. |
| `--mix-real` / `--curriculum` / `--real-only` | `false` / `false` / `false` | Select a mutually exclusive training regime. |
| `--eval-set` | `real` | `real` public test split or `holdout` from the training source. |
| `--eval-samples` | `200` | Maximum real evaluation rows. |
| `--model` | `openai/whisper-small.en` | Initial checkpoint. |
| `--out` | required | Output checkpoint directory. |
| `--epochs` | `3` | Epochs per phase. |
| `--max-steps` | `-1` | Maximum steps per phase; `-1` uses epochs. |
| `--batch-size` | `8` | Train and evaluation batch size. |
| `--lr` | `1e-05` | Learning rate. |
| `--fp16` | `false` | Request FP16 when CUDA is available. |
| `--eval-holdout` | `0.02` | Holdout fraction for `--eval-set holdout`. |
| `--seed` | `42` | Trainer seed. |

Example:

```bash
uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl \
    --model openai/whisper-tiny.en --out runs/whisper_synthetic_only \
    --max-steps 300 --batch-size 8
```

The final model and processor are saved under `--out`. A curriculum also
creates `phase_1_synthetic/` and `phase_2_real/` training output directories
before saving the final model at `--out`.

### `scripts/run_matrix.py`

Runs the A0--A4 validation matrix, including generation, gating, training,
model selection, locked-test evaluation, and paired summaries.

| Flag | Default | Purpose |
|---|---:|---|
| `--out` | `runs/matrix_v1` | Matrix run directory. |
| `--model` | `openai/whisper-tiny.en` | Student checkpoint/model. |
| `--config` | `configs/mode1_matched.yaml` | Generator profile. |
| `--text` | `grammar:region=eu` | Matrix text source. |
| `--n-synth` | `8000` | Synthetic pool size. |
| `--gen-seed` | `101` | Synthetic generation seed. |
| `--sft-steps` / `--sft-batch` / `--sft-lr` | `2000` / `8` / `1e-05` | SFT budget. |
| `--mix-ratio` | `0.75` | Real fraction in mix arms. |
| `--grpo-steps` / `--grpo-lr` | `600` / `2e-06` | GRPO budget. |
| `--seed` | `0` | Training seed. |
| `--arms` | `a1_real,a2_synth_gated,a2u_synth_ungated,a3_mix,a4_mix_grpo` | Comma-separated arm subset. |
| `--arm-workers` | `5` | Concurrent training arms. |
| `--eval-workers` | `3` | Concurrent evaluation subprocesses. |
| `--skip-final` | `false` | Stop before the `locked_test` reads. |

Example:

```bash
uv run python scripts/run_matrix.py --out runs/matrix_v1 \
    --arm-workers 1 --eval-workers 1
```

The run writes `matrix_config.json`, `synth_pool/` with the raw, gated, and
selected manifests, `arms/<arm>/` checkpoints and `run.json` files,
`eval/` reports and hypothesis JSONL files, `summary_model_select.json`,
`summary_locked_test.json`, and `logs/`. On MPS, concurrent Whisper trainers
share unified memory; the default `--arm-workers 5` can exceed available
memory. Use conservative worker counts such as `1` and see
[known-issues.md](known-issues.md) before running a matrix.

## RL

### `scripts/rl_loop.py`

Searches generator configuration knobs with CEM, REINFORCE, or random search
against a fixed downstream ASR reward.

| Flag | Default | Purpose |
|---|---:|---|
| `--base-config` | `configs/mode1_matched.yaml` | Profile whose knobs are searched. |
| `--out` | required | Search run directory. |
| `--optimizer` | `cem` | `cem`, `reinforce`, or `random`. |
| `--iterations` | `4` | Candidate batches. |
| `--pop-size` | `4` | Candidates per batch. |
| `--seed` | `0` | Optimizer sampling seed. |
| `--no-seed-default` | `false` | Skip the base-profile trial. |
| `--no-resume` | `false` | Restart numbering and truncate the trial log. |
| `--n-synth` | `200` | Synthetic clips per trial. |
| `--ft-steps` | `300` | Fine-tuning steps per trial. |
| `--dev-indices` | `0:200` | Real development slice. |
| `--text-pool` | `400` | Shared text-pool utterances. |
| `--device` | `None` | Torch device; automatic selection when omitted. |

Example:

```bash
uv run python scripts/rl_loop.py --out runs/rl_v1 \
    --iterations 4 --pop-size 4 --device mps
```

The run writes `trials.jsonl`, `optimizer_state.json`, `best.json`, and
`best_config.yaml` under `--out`. Each candidate is under
`trials/NNN/` with its resolved config and synthetic dataset; the shared
reward harness is under `harness/`.

### `scripts/rl_recipe_bandit.py`

Uses Thompson sampling to choose synthetic recipe buckets with a teacher-bounded
student-hardness window and periodic counterfactuals.

| Flag | Default | Purpose |
|---|---:|---|
| `--out` | required | Bandit run directory. |
| `--base-config` | `configs/mode1_matched.yaml` | Profile mutated by recipes. |
| `--pulls` | `30` | Total pulls. |
| `--n-batch` | `60` | Clips per pull. |
| `--student` | `openai/whisper-tiny.en` | Student used for the hardness window. |
| `--teacher` | `openai/whisper-base.en` | Frozen teacher. |
| `--tau1` / `--tau2` / `--tau3` | `0.8` / `0.4` / `1.2` | Teacher ceiling and student hardness-window bounds. |
| `--counterfactual-every` | `8` | Recalibration cadence; `0` disables scheduled and final counterfactuals. |
| `--cf-steps` / `--cf-m` / `--cf-eval-n` | `300` / `150` / `400` | Counterfactual fine-tuning steps, clips per arm, and real evaluation rows. |
| `--cf-init` | `None` | Frozen counterfactual init; defaults to `--student`. |
| `--seed` | `20260824` | Thompson and generation seed. |
| `--noise-only-frac` | `0.0` | Noise-only fraction forced into each pull. |
| `--asr-batch` | `16` | Decode batch size. |
| `--table-every` | `5` | Posterior-table cadence. |
| `--device` | `None` | Torch device; automatic selection when omitted. |

Example:

```bash
uv run python scripts/rl_recipe_bandit.py --out runs/bandit_v1 \
    --pulls 30 --n-batch 60 --counterfactual-every 15 \
    --cf-m 150 --cf-steps 250
```

The run writes `pulls.jsonl`, `state.json`, `counterfactuals.jsonl`,
`pulls/NNN/` pull configs and synthetic datasets, `selected/manifest.jsonl`,
`spillover/manifest.jsonl`, and `cf/NNN/` counterfactual artifacts.

### `scripts/rl_verify.py`

Compares the hand-tuned and searched generator profiles on a fresh held-out
test slice with zero-shot, base, and best arms.

| Flag | Default | Purpose |
|---|---:|---|
| `--run` | required | Search run containing `best_config.yaml`. |
| `--base-config` | `configs/mode1_matched.yaml` | Hand-tuned comparison profile. |
| `--out` | `None` | Output directory; defaults to `<run>/verify`. |
| `--test-corpus` | `jacktol/atc-dataset` | Held-out corpus. |
| `--test-split` | `test` | Held-out corpus split. |
| `--test-indices` | `0:500` | Blind test slice. |
| `--n-synth` | `600` | Synthetic clips per fine-tuned arm. |
| `--ft-steps` / `--ft-batch` / `--ft-lr` | `600` / `8` / `1e-05` | Fine-tuning budget. |
| `--text-pool` / `--text-seed` | `1200` / `4321` | Fresh common-random-number text pool. |
| `--device` | `None` | Torch device; automatic selection when omitted. |
| `--save-models` | `false` | Save base and best models under `<out>/models/`. |

Example:

```bash
uv run python scripts/rl_verify.py --run runs/rl_v1 \
    --test-indices 0:500 --n-synth 600 --ft-steps 600 --save-models
```

The run writes `text_pool.jsonl`, `arms/base/` and `arms/best/` configs and
synthetic datasets, and `verify_report.json` under the output directory. With
`--save-models`, it also writes `models/base/` and `models/best/`.
