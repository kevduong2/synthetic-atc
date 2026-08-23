# Shared Architecture & Config Design

Common infrastructure for both generation modes. Read with [01-codebase-analysis.md](01-codebase-analysis.md) (current state) and the per-mode plans ([03](03-mode1-procedural-plan.md), [04](04-mode2-calibrated-plan.md)).

## 1. Scope and principles

- **Audio only.** Input is text (`spoken` + `transcript`) from any `TextSource`; output is `(wav, manifest record)` pairs. Text/scenario generation stays behind the existing `TextSource` protocol (`atcgen/text/sources.py`) — we extend its record contract, never its content.
- **Kokoro TTS is a fixed dependency**, wrapped by the existing `TTSEngine` protocol. Voice-level variation (speed, pitch, accent/voice pool) is our scope and sits in a TTS-adjacent augment stage, not inside the engine.
- **One pipeline, two channel backends.** Mode 1 and Mode 2 differ only in the *channel stage*. TTS, voice augmentation, dataset building, manifests, evaluation, and config are shared.
- **Everything sampled is seeded and logged.** A dataset must be reproducible from `(config, seed, text source)`; every sample's manifest record carries what was actually drawn.

## 2. Data flow

```
                       ┌────────────────────────────────────────────┐
 TextSource (theirs)   │                generator                   │
 ──spoken/transcript──►│ TTS (Kokoro) ─► voice augment ─► channel ──┼─► wavs/ + manifest.jsonl
   weight/category     │  voice, speed    pitch/tempo      stage    │      (16 kHz mono)
                       └────────────────────────────────┬───────────┘
                                                        │
                              ┌─────────────────────────┴────────────────────────┐
                              │ Mode 1: procedural chain (channel/chain.py)      │
                              │   config-driven primitive chain, randomized      │
                              │ Mode 2: calibrated (channel/learned/)            │
                              │   learned core + shared primitives for the       │
                              │   effects the model can't produce (clicks, codec)│
                              └──────────────────────────────────────────────────┘

 data/real/* ──► local_corpus.py ──► noise beds / training clips / eval reference
                                          (Mode 1 optional input; Mode 2 required)
```

Both channel backends implement one interface:

```
ChannelBackend.__call__(wav: f32 @ 16 kHz, rng, meta: UtteranceMeta) -> (wav16k, ChannelRecord)
```

`UtteranceMeta` carries role (pilot/controller → double-hop eligibility) and category; `ChannelRecord` is the provenance blob for the manifest.

## 3. Pipeline stages

1. **TTS** — Kokoro via `TTSEngine`; voice + speed drawn from config-defined pools/ranges (today's hardcoded lists move to config).
2. **Voice augment** (new, shared) — applied to clean 24 kHz TTS output before the channel: pitch shift (semitones), tempo (beyond Kokoro's speed knob, for compression artifact-free variety), light formant/EQ tilt for speaker-timbre spread. Each is an independent primitive with a probability + range in config.
3. **Channel** — Mode 1 or Mode 2 backend (or `mix`, weighted).
4. **Post** — peak safety normalization, target loudness jitter, final resample to 16 kHz (already largely present).
5. **Writer** — wav + manifest record; noise-only samples and silence padding policies live here, not in the channel.

## 4. Config system

New `atcgen/config.py`: typed dataclasses mirrored by YAML profiles under `configs/`. CLI flags become `--config path.yaml` plus a few overrides (`--n-samples`, `--out`, `--seed`, `--text`).

Every random knob is declared as a **distribution spec**, not a value: `{uniform: [3, 25]}`, `{choice: [6000, 8000, 8000]}`, `{const: 0.5}`, optionally gated by `prob:`. This replaces the logic currently frozen inside `ChannelParams.sample()` and makes `harsh`/`matched`/`mild` variants pure data.

```yaml
# configs/mode1_default.yaml (sketch — authoritative schema in 03/04 plans)
mode: procedural            # procedural | calibrated | mix
seed: 0
output: { sample_rate: 16000, format: wav, loudness_db: {uniform: [-23, -17]} }

tts:
  voices: [af_heart, af_bella, am_adam, bm_george, ...]
  speed: {uniform: [0.95, 1.55]}

voice_augment:
  pitch_semitones: {prob: 0.5, uniform: [-2, 2]}
  tempo: {prob: 0.3, uniform: [0.9, 1.1]}
  eq_tilt_db: {prob: 0.4, uniform: [-3, 3]}

dataset:
  noise_only_frac: 0.03
  pilot_double_hop_prob: 0.5
  category_quotas: {emergency: 0.10, rare_vocab: 0.10}   # oversampling targets

channel:                    # Mode 1 body — full schema in 03-mode1 plan
  profile: matched          # named base; keys below override
  bandpass: {low: {uniform: [200, 350]}, high: {uniform: [2200, 3000]}}
  snr_db: {uniform: [8, 30]}
  noise: {beds_dir: data/noise_beds, bed_prob: 0.6, colors: [pink, pink, white]}
  effects: {squelch_gate: {prob: 0.8}, dropouts: {...}, codec: {...}, ...}

calibrated:                 # Mode 2 body — full schema in 04-mode2 plan
  checkpoint: runs/channel_v2/G_latest.pt
  post_dsp_profile: mild
```

Rules: unknown keys are errors; the resolved config (defaults merged) plus its hash are written next to the manifest; `mix` mode takes a list of `{backend, weight}`.

## 5. Text-source contract extension (interface only)

JSONL records gain optional fields the builder respects: `weight` (sampling weight), `category` (e.g. `emergency`, `rare_vocab`, `routine` — used by `category_quotas` and eval slicing), `role`, `kind` (already supported). Uniform sampling remains the default. Quota logic: sample by weight within category, top up categories to quota fractions when the source provides labels; log achieved fractions.

## 6. Output format

Unchanged base (`wavs/NNNNNN.wav` + `manifest.jsonl`), extended records:

```json
{"audio": "wavs/000123.wav", "text": "...", "role": "pilot", "kind": "clearance",
 "category": "emergency", "duration": 4.2,
 "gen": {"mode": "procedural", "config_hash": "ab12...", "seed_index": 123,
          "voice": "am_adam", "speed": 1.31, "pitch": -1.5,
          "channel": {"snr_db": 14.2, "bp": [280, 2650], "codec": "mp3@16k", "hops": 2}}}
```

Plus per-run: `config.resolved.yaml`, `stats.json` (category/duration/SNR histograms). `load_manifest` keeps working (extra keys ignored by training).

## 7. Repo layout, dependencies, hardware

Target layout is in [01-codebase-analysis.md §4](01-codebase-analysis.md). New runtime deps stay minimal: YAML (`pyyaml` or `ruamel`), optional `pyloudnorm`; eval deps (embedding models) are an extra (`[eval]`). Generation (Mode 1) runs on CPU/MPS; Mode 2 training targets the CUDA 5080 box; Mode 2 inference must also run acceptably on CPU/MPS for small batches.

## 8. Migration

Nothing is in production — no backwards compatibility is required. `dsp.py` is replaced outright by `primitives.py`/`chain.py` (existing effect implementations are the reference for porting, then deleted); tests are ported/rewritten against the primitives; the CLI moves directly to `--config` with a few overrides; the manifest format changes freely to the schema in §6. The only discipline kept: land the port with parameter values matching today's behavior first, then retune distributions in a separate commit, so audible regressions are bisectable.
