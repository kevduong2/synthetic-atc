---
name: asr-feedback-loop
description: Closing the loop between atc-gan and the sibling asr repo (reference-data-for-v1-run/asr) whose optimization/ package trains the production Whisper model. Exporting gated synthetic audio in the asr V2 CSV shape, registering it as a synthetic source in optimization/config.yaml, running python -m optimization.train under the GPU lock, reading back best_model.pth with its provenance, evaluating it, and feeding the checkpoint back into atc-gan as an evaluation or reward model. Use for any task that touches the asr trainer or the real-model feedback loop.
---

# ASR feedback loop

Two repos, one shared surface. atc-gan makes audio; the asr repo
(`reference-data-for-v1-run/asr`, Windows-first, its own venv, no CLI flags on
the trainer) trains and evaluates the real model. The only contract between
them is the corpus CSV: `audio,text,suspect`, keyed by **basename**.

| | atc-gan | asr |
|---|---|---|
| trainer | `training/recipe.py`, `atcgen/rl/finetune_lite.py` (whisper-tiny/small, fixed steps) | `python -m optimization.train` reading `optimization/config.yaml` only |
| base model | `openai/whisper-tiny.en` (reward), `small.en` (transfer check) | `openai/whisper-medium`, batch 1 × grad-accum 32, fp16 off, grad-checkpointing on |
| WER | `training/normalize.normalize_atc` + `training/evaluate.py` | `utils.compute_wer` (wildcard-absorbing) + aviation-critical AWER |
| checkpoint | HF `save_pretrained` dir | `outputs/run-<ts>/best_model.pth`, an `asr_checkpoint_v1` blob with the full config, dataset version, git rev and epoch metrics embedded |
| selection | paired reward on `kixd_dev` | lowest `real_val_awer`; synthetic val/test tracked separately |

The two WER definitions are not interchangeable. Never compare a number from
one evaluator with a number from the other; pick one per claim and say which.

## 1. Export synthetic audio for the asr trainer

```
uv run python scripts/gate_dataset.py --dataset runs/<render> --device cuda
uv run python scripts/export_corpus_csv.py --dataset runs/<render> --out data/corpus/V<maj>.<min>.<patch> --version V<maj>.<min>.<patch> --reason "<one line>"
```

The exporter writes `corpus_train.csv`, `corpus_test.csv` and `manifest.json`
(sha256) with absolute audio paths, drops empty-text noise-only rows unless
`--include-noise-only`, and splits by transcript group stratified by airport.
The asr side wants this layout, so copy or link:

```
asr/resources/synthetic_clips/<version>/synthetic_corpus_train.csv
asr/resources/synthetic_clips/<version>/synthetic_corpus_test.csv
asr/resources/synthetic_clips/<version>/audio/<unique basenames>.wav
```

Basename trap: `AviationDataset` discards directories and looks the basename
up under `clips_dir`. Two atc-gan renders both contain `000000.wav`; when
merging renders (main + noise-only), prefix the copied files with the run name
and rewrite the `audio` column to match. One render per synthetic version is
simpler.

## 2. Register it in `optimization/config.yaml`

Set `data.synthetic.train_csv`, `data.synthetic.test_csv`, `data.synthetic.clips_dir`
to the three paths above (all three or none; the loader raises otherwise). Real
data keeps coming from `dataset_versioning.dataset_provenance(2)`, i.e.
`asr/resources/datasets/V2.1.2/`. Mixing is plain concatenation with no ratio
knob, so the real share is set by row counts: for real fraction r with N_real
rows, export about N_real × (1/r − 1) synthetic rows (runbook guidance: 50–75%
real, never synthetic-only, finish with a short real-only phase). Selection
stays on `real_val_awer`, so synthetic data can only win through the real
metric, which is the point.

Also check `output_dir`, `model_type: whisper`, `model_name`, and the epoch
count before launching; the resolved config is copied into the run dir.

## 3. Run it under the lab's GPU lock

The asr repo is not a uv project; build its venv once as its `readme.txt` says
(`python -m venv .venv_build`, `pip install -r requirements.txt`) and launch
with that interpreter from its own working directory:

```
uv run python scripts/lab/jobs.py launch --gpu --id asr-<version> --cwd reference-data-for-v1-run/asr -- .venv_build/Scripts/python.exe -m optimization.train
```

whisper-medium full fine-tuning at batch 1 with gradient checkpointing has not
been measured on a 10 GB card. Treat the first epoch as a smoke test with a
pre-authorized kill on OOM; if it does not fit, the fallback is
`model_name: openai/whisper-small` in the asr config for the development loop,
with medium reserved for the machine that trains production.

Outputs in `outputs/run-<ts>/`: `best_model.pth`, `checkpoint-epoch-N.pth`,
`final/model.pth`, `training_history.json` (real vs synthetic val/test WER per
epoch), `training_curves.png`, `console_output.log`, and the config used.

## 4. Evaluate the checkpoint

On the asr side (their metric, their normalizer), the pattern in
`asr/run_test_clips.py`:

```python
from asr_model import WhisperModel
m = WhisperModel(checkpoint_path="outputs/run-.../best_model.pth"); m.load()
hyp = m.transcribe(wav_path)
```

and `utils.normalize_text` / `compute_wer` / `compute_aviation_wer` for the
scores; `models.save_run(...)` persists an `eval_runs/run_<ts>.json` with the
checkpoint's provenance, including which synthetic version trained it
(`checkpoint_meta.synthetic_dataset_version_from_metadata`).

On the atc-gan side (`training/evaluate.py --model <dir>`), the checkpoint must
first be converted to an HF directory: load it as above, then
`m.model.save_pretrained(dir)` and `WhisperProcessor.from_pretrained("openai/whisper-medium").save_pretrained(dir)`
(the asr processor is a hand-rolled log-mel front end; the HF one matches for
80-mel Whisper sizes). Record which path produced every number.

## 5. Feed it back into atc-gan

Three uses, in increasing cost:

1. **Evaluation model.** `training/evaluate.py --model <hf dir> --split-name ...`
   or the dev CSV via `--dataset`; and, exactly once, the final
   `kixd_locked_day` read. This is the cheap, high-value use.
2. **Gate teacher.** The gate's teacher ids live in `atcgen/gate/teachers.py`
   (`whisper-base.en` + wav2vec2). A stronger in-domain teacher changes tier
   yields; that is an owner decision after a yield-vs-ceiling table, not a
   silent swap.
3. **Reward model.** `atcgen/rl/reward.py` fine-tunes `openai/whisper-tiny.en`
   per trial (hardcoded `base_model`; a `--model` passthrough on
   `rl_power_check.py` is the Phase 2B task). A medium-sized model per cell is
   not affordable at production budgets on 10 GB; use the real checkpoint to
   *score* candidate renders zero-shot or with a short decoder-only fine-tune,
   and pre-register a transfer check (tiny → real model agreement on
   `aug_off`) before trusting it for selection. New model id ⇒ fresh `--out`.

## 6. The loop, one cycle

```
render (frozen V1 or a pre-registered variant) → gate → export V<x>
  → asr train with synthetic V<x> → real_val_awer vs the no-synthetic run (same seed, same real data)
  → if it helps: freeze V<x>, evaluate the checkpoint on kixd_dev with atc-gan's evaluator
  → use the checkpoint to re-score generator arms (evaluation, not fine-tune reward)
  → spec the next generator variant → next cycle
```

Each cycle is a mission with its own spec, decision rule (e.g. real_val_awer
improves by ≥ 2× its epoch-to-epoch spread, and no entity regression) and a
report. The checkpoint carries `synthetic_dataset_version`, so provenance
survives the round trip; the corpus `manifest.json` sha256 pins the audio.
