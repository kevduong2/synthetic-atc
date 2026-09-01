---
name: gpu-jobs
description: Running atc-gan jobs on the Windows RTX 3080 box. The one-GPU-stream rule and lock, launching detached jobs with scripts/lab/jobs.py (launch, status, watch, kill), budgets in units of C, per-script resumability, VRAM and OOM handling, and Windows/PowerShell specifics (uv run, no heredocs, path rewriting with scripts/lab/relocate.py, CUDA sanity checks, fp16 caveat, first-run model downloads). Use before starting any job that runs longer than a minute.
---

# GPU jobs

## 1. One GPU stream

Two GPU-heavy jobs at once slowed both by 40%. `launch --gpu` takes the lock in
`lab/GPU_LOCK`; a second `--gpu` launch is refused (exit 4) while the holder's
process is alive. The lock is released when the job exits, is killed, or is
found dead. CPU-only analysis (`paired_report.py`, `relocate.py`, tests that
skip CUDA) may run alongside. Check first:

```
uv run python scripts/lab/jobs.py lock status
```

## 2. Launch, status, watch, kill

```
uv run python scripts/lab/jobs.py launch --gpu --id win2-p1-gate -- uv run python scripts/rl_power_check.py --out runs/win2_gate --base-config configs/mode1_matched_kixd.yaml --arms base,degraded --seeds 0,1,2 --dev-corpus data/real/kixd/kixd_dev.csv --n-synth 400 --ft-steps 500 --device cuda
uv run python scripts/lab/jobs.py status win2-p1-gate --tail 20 --gpu
uv run python scripts/lab/jobs.py watch  win2-p1-gate --interval 300 --max-wait 1500
uv run python scripts/lab/jobs.py kill   win2-p1-gate
uv run python scripts/lab/jobs.py status            # all jobs + lock
```

`launch` detaches the command from your terminal (it survives the tool call
ending), appends stdout+stderr to `lab/jobs/<id>/log.txt`, writes
`cmd.txt`/`cmd.json`, and records pids and the exit code in `status.json`.
Everything after `--` is the exact command; keep it on one line in PowerShell
(or use the backtick for continuation, never `\`). Use `--cwd <dir>` to run
inside another repo (the asr trainer) and `--env KEY=VALUE` for per-job
environment. `PYTHONUNBUFFERED=1` is set for you so progress lines land in the
log immediately.

Hand the watch to the lab-assistant with a watch brief (`monitor-run` skill);
do not poll in your own context.

## 3. Budgets

C = wall-clock of one production-budget reward cell (`--n-synth 400
--ft-steps 500`, whisper-tiny). Measure it on the first cell of a session and
write it at the top of your report; every spec's budget is in units of C. If
C > 8 min, halve seeds (min 4 for claims, 2 for gates) and drop stretch
phases, per the mission file. Other reference points come from
`scripts/bench_devices.py --device cuda` (TTS s/render, FastCUT s/step,
whisper-tiny SFT s/step) and replace every MPS-extrapolated number in the docs.

## 4. Resumability (rerun the same command, same `--out`)

| script | resume behaviour |
|---|---|
| `scripts/rl_power_check.py` | per cell: finished cells are read back from `results.jsonl` |
| `scripts/rl_loop.py` | resumes trial numbering and optimizer state; `--no-resume` restarts |
| `atcgen.channel.learned.residual_train` | `--resume <ckpt>`; saves every `--save-every` steps |
| `scripts/generate_dataset.py` | not resumable (`stats.json` only at the end): shard the text with `scripts/lab/shard_text.py --n 4`, one seed and `--out` per shard, export together |
| `scripts/gate_dataset.py` | rerun from scratch; cheap relative to rendering |
| `training/finetune_whisper.py`, `training/recipe.py` | rerun from scratch |

A fresh `--out` is mandatory whenever the metric, model id or dev slice
changes; caches key off corpus and would silently mix definitions.

## 5. VRAM (10 GB) and OOM

- whisper-tiny/base reward cells: fine at batch 8. whisper-small.en fine-tune:
  batch 8 usually fits; on OOM use batch 4 + grad-accumulation ×2 (pre-authorize
  this in the brief so the watcher can act).
- FastCUT residual at `--batch-size 12 --crop-frames 128`: expected to fit;
  drop to 8 on OOM and say so in the report (it changes the recipe).
- Gate teachers (`whisper-base.en` + wav2vec2) at `--batch 8`: fine.
- `nvidia-smi` is the source of truth; `jobs.py status --gpu` includes it.
- `training/finetune_whisper.py --fp16` was a silent no-op on the Mac (MPS) and
  is live on CUDA. Leave it off unless the spec asks for it; it changes numerics.

## 6. Windows specifics

- PowerShell, not bash: no heredocs (put snippets in `scripts/analysis/`; the
  probe resample is `scripts/lab/resample_probes.py`), no
  `\` continuation, no `export` (`$env:NAME = "value"`), `unzip` is
  `Expand-Archive`. Quote `--set key=value` overrides as one token.
- Paths: forward slashes work everywhere in this repo; every manifest under
  `data/real/` and `runs/*/corpus.jsonl` stores absolute paths. After copying
  the tree, rewrite them once:

  ```
  uv run python scripts/lab/relocate.py --from /Users/kevin/repos/ai/atc-gan --to C:/path/to/atc-gan data/real runs/channel_data_kixd runs/calib_kixd --check
  uv run python scripts/lab/relocate.py --from /Users/kevin/repos/ai/atc-gan --to C:/path/to/atc-gan data/real runs/channel_data_kixd runs/calib_kixd --check --apply
  ```

- CUDA sanity before the first job:

  ```
  uv run python -c "import torch, soundfile; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), soundfile.__libsndfile_version__)"
  ```

  `cuda.is_available()` must be True (the Windows torch wheel comes from the
  CUDA index declared in `pyproject.toml`; a CPU-only wheel means `uv sync`
  resolved from PyPI). `libsndfile` should read 1.2.2; the MP3 codec's
  compression-to-bitrate table was measured there.
- Pass `--device cuda` explicitly to every script that accepts it, even where
  autodetect would pick it; it makes the log self-describing.
- First run downloads models into the HF cache: `openai/whisper-tiny.en`,
  `openai/whisper-base.en`, `facebook/wav2vec2-base-960h`, `hexgrad/Kokoro-82M`,
  `microsoft/wavlm-base-plus`, `laion/clap-htsat-unfused`, and
  `openai/whisper-small.en` for the transfer check (~3 GB total). Set
  `$env:HF_HOME` if the system drive is small.
- `ffmpeg` is only needed for the `aac` codec kind; without it AAC silently
  passes through. The frozen profiles use `mp3` (libsndfile-native).
- Kokoro needs no system `espeak-ng` (the loader wheel bundles it) but does
  need the spaCy model `en_core_web_sm`, which `uv sync` installs from the
  pinned wheel in `pyproject.toml`.
