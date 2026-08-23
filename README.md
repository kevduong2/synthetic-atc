# atc-gan

Synthetic ATC radio audio generator for training ASR models (Whisper) to transcribe air traffic control communications.

Pipeline (mirrors the SOTA approach from the Interspeech 2026 synthetic-ATC paper, arXiv:2606.21340):

```
ATC phraseology grammar ──► TTS (Kokoro, multi-voice, fast delivery)
        │                        │
   transcript label              ▼
        │            radio channel degradation
        │            ├── DSP simulator (parametric VHF/AM chain)
        │            └── CycleGAN channel model (learned from real ATC audio)
        ▼                        ▼
        (audio, transcript) pairs ──► fine-tune Whisper ──► eval WER on real ATC
```

## Quick start

```bash
uv sync --extra dev
uv run pytest                        # 21 tests

# generate a listenable sample set (DSP channel)
uv run python scripts/generate_dataset.py --out data/smoke --n-samples 10

# harvest real noise beds once (static/carrier hiss between transmissions)
uv run python -c "from atcgen.dataset.real_atc import export_noise_beds; \
    print(export_noise_beds('data/noise_beds'))"

# generate a training set (~10-20h ≈ 8k-16k samples)
uv run python scripts/generate_dataset.py --out data/train_v1 --n-samples 10000 \
    --noise-dir data/noise_beds
```

Channel realism knobs baked into generation: low-bitrate MP3 codec round-trip
(matches LiveATC/ATCO2 delivery), real ATC noise beds via `--noise-dir`, ~3%
noise-only samples with empty transcripts (Whisper hallucination control,
`--noise-only-frac`), and a probabilistic double radio hop for pilot utterances.

### Bring your own transcripts

Teammates' text generators plug in via JSONL (`{"spoken": ..., "transcript": ...}` or `{"text": ...}` per line):

```bash
uv run python scripts/generate_dataset.py --text my_transcripts.jsonl --out data/custom --n-samples 5000
```

Or implement the `TextSource` protocol in `atcgen/text/sources.py`.

## GAN channel model (on the CUDA box)

```bash
# 1. clean TTS audio for domain A
uv run python scripts/generate_dataset.py --out data/clean --n-samples 2000 --channel clean

# 2. real ATC audio for domain B
uv run python -c "from atcgen.dataset.real_atc import export_gan_domain_audio; \
    print(export_gan_domain_audio('data/real_atc', max_clips=2000))"

# 3. train CycleGAN (audition wavs written per epoch)
uv run python -m atcgen.channel.gan.train --domain-a data/clean/wavs \
    --domain-b data/real_atc --out runs/cyclegan --device cuda --epochs 40

# 4. generate with the learned channel (or --channel mix for 50/50 DSP/GAN)
uv run python scripts/generate_dataset.py --out data/train_gan --n-samples 10000 \
    --channel gan --gan-checkpoint runs/cyclegan/G_ab_latest.pt
```

The GAN operates on log-magnitude STFTs and reuses source phase at inference — no vocoder needed; it learns spectral coloration/band-limiting/noise floor from real ATCO2/UWB-ATCC audio. Because the learned channel is deterministic (one "radio"), GAN samples also get a mild randomized DSP pass (SNR/band/codec) for per-sample diversity.

## Fine-tune + evaluate Whisper

```bash
# baseline WER on real ATC test data
uv run python training/evaluate.py --model openai/whisper-small.en --dataset real

# fine-tune; --mix-real is recommended (real data upsampled to ~1:1 with synthetic —
# the ratio the synthetic-ATC literature found optimal; synthetic-only underperforms)
uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl --mix-real \
    --model openai/whisper-small.en --out runs/whisper_atc --epochs 3 --batch-size 16 --fp16

# WER after fine-tuning
uv run python training/evaluate.py --model runs/whisper_atc --dataset real
```

WER uses ATC-aware normalization (`training/normalize.py`): digit expansion, `niner→nine`/`tree→three`/`fife→five` folding.

## Layout

- `atcgen/text/` — phraseology grammar (FAA 7110.65/ICAO patterns) + pluggable text sources
- `atcgen/tts/` — Kokoro TTS wrapper (voice/speed randomization; speed 1.15–1.55× for fast controller delivery)
- `atcgen/channel/primitives.py` + `chain.py` — parametric VHF channel as a config-declared chain of per-effect primitives: narrowband resample, 300–3400 Hz bandpass, AM distortion, AGC pumping, static at 3–25 dB SNR (or real noise beds), hum, crackle, squelch clicks, dropouts, heterodyne, co-channel interference, low-bitrate MP3 round-trip, pilot double-hop
- `atcgen/channel/gan/` — CycleGAN channel model (train on 5080, `--device cuda`)
- `atcgen/dataset/` — dataset builder + real-corpus prep (`jacktol/atc-dataset`, `Jzuluaga/uwb_atcc`)
- `training/` — Whisper fine-tune, WER eval, ATC text normalization
