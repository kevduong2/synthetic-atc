# Mode 2 Plan — Calibrated Channel (learned from real samples)

Design and implementation plan for the calibrated generator: learn the noise/channel characteristics of a small real dataset (~1k clips) and stamp them onto clean TTS audio. Target use case: expand ~1,000 labeled real samples to ~10,000 (9k synthetic), oversampling emergency/rare-vocabulary cases.

Grounding: research findings in [00-research-findings.md](00-research-findings.md); gaps in [01-codebase-analysis.md](01-codebase-analysis.md); shared interfaces in [02-architecture.md](02-architecture.md); acceptance metrics in [05-evaluation-plan.md](05-evaluation-plan.md).

## 1. Method choice (research-driven)

The survey ranked four families for the ~1k-clip regime (details and citations in 00 §3):

| Option | Verdict |
|---|---|
| Conditional diffusion / flow | **Defer.** Best diversity, but unproven at 1k clips for degradation-direction modeling (a literature gap, not a disproof), costliest to train/iterate. Revisit in a later phase if the chosen approach plateaus. |
| Pure CycleGAN (current repo v1) | **Demote.** Deterministic mapping averages the ~4 stations/channels in our data into one "radio"; cycle-consistency provokes the known steganography failure on information-destroying mappings (1712.02950); no interpretable knobs. |
| **DSP-hybrid: per-clip differentiable channel fitting + real noise bank** | **Adopt as the backbone.** MicAugment (2010.09658) proves few-second channel identification; persoDA (2501.09113) proves VAD-harvested noise + SNR-matched mixing. Turns 1k clips into ~1k interpretable channel presets — diversity by sampling, zero mode-collapse risk, CPU inference. |
| **Residual translator: CUT/FastCUT (GAN)** | **Adopt as stage 2, gated.** Learns only what the DSP fit misses. CUT (2007.15651) over CycleGAN: one-sided (we never need radio→clean), ~2× cheaper, avoids the cycle failure. Small-data stack: DiffAugment/ADA + R1 (sweep γ) + generator EMA + multi-res STFT discriminator (00 §5). Ships only if it improves the channel probe and downstream WER (05 §3). |

The existing CycleGAN (`atcgen/channel/gan/`) stays as a comparison baseline until M2.4, then is retired or kept behind `method: cyclegan_v1`.

## 2. Architecture

```
                     ┌───────────────── calibration (offline, per real corpus) ─────────────────┐
 data/real/*.wav ──► │ local_corpus.py: QC, trim, split          ──► corpus manifest            │
                     │ noise_harvest.py: VAD → noise bank        ──► noise/*.wav + stats        │
                     │ channel_fit.py: per-clip differentiable   ──► presets.jsonl (~1k rows:   │
                     │   chain fit (EQ/band, nonlinearity, AGC,       filter coefs, clip drive, │
                     │   noise gain) by gradient descent              SNR, squelch stats, ...)  │
                     │ residual_train.py: CUT clean-DSP → real   ──► G_residual.pt  [gated]     │
                     └──────────────────────────────────────────────────────────────────────────┘

                     ┌───────────────── generation (CalibratedChannel backend) ─────────────────┐
 clean TTS 16 kHz ──►│ draw (preset, noise crop, SNR jitter) ─► apply fitted DSP chain          │
                     │ ─► [G_residual, prob r] ─► shared primitives the model can't produce     │──► wav
                     │      (squelch clicks, dropouts, codec round-trip — from Mode 1 chain)    │
                     └──────────────────────────────────────────────────────────────────────────┘
```

Implements the shared `ChannelBackend` interface (02 §2); everything downstream (builder, manifest, eval) is common.

### 2.1 Per-clip channel fitting (`channel_fit.py`)
Differentiable parametric chain in PyTorch, fitted per real clip by gradient descent (MicAugment-style):

- Components: learned FIR/biquad EQ (captures band edges + mic/handset tilt; init from Mode 1 defaults), memoryless nonlinearity (tanh-drive + polynomial), AGC time constant, additive noise gain against a noise embedding from the same clip.
- Loss: multi-scale spectral loss between (chain applied to an enhanced/VAD-speech version of the clip — or a matched clean proxy) and the original clip; where no clean counterpart is derivable, fit against LTAS + modulation statistics instead (weakly supervised fit). Start with the statistics-matching variant: it is simpler and needs no enhancement model; upgrade to enhanced-proxy fitting if presets look too blurry.
- Output: one JSONL row per clip: `{clip_id, station, eq_coefs, drive, agc, snr_est, band_edges, fit_loss}`. Fits are embarrassingly parallel; minutes on the 5080, feasible on CPU.
- Presets with outlier fit_loss are dropped (fit QC).

### 2.2 Noise bank (`noise_harvest.py`)
Replaces the current quietest-window heuristic with VAD-based harvesting (persoDA-style): energy+VAD segmentation → non-speech segments ≥200 ms → store with per-segment stats (RMS, LTAS centroid, squelch-gated flag). The measured squelch gating in local clips (01 §2) is recorded so generation can reproduce gated vs continuous noise floors.

### 2.3 Sampling model
Per synthetic utterance: draw a preset (optionally stratified by station to match a target mix), a noise crop from the same station when available, SNR jittered around the preset's estimate (±3 dB), then squelch/PTT/codec effects from shared primitives with probabilities estimated from the corpus. Correlated draws (preset + its own noise) keep combinations physically plausible; cross-station mixing is allowed at low probability for extra diversity.

### 2.4 Residual CUT translator (`residual_train.py`, gated)
- Domain A: clean TTS **already passed through the fitted DSP sampling model**; Domain B: real clips. The translator therefore models only the residual gap.
- Recipe (00 §5): FastCUT, patchNCE, DiffAugment on both real/fake spectrogram patches (time shifts/masks, gain — no frequency flips), R1 with γ swept log-scale, generator EMA 0.999–0.9999, multi-resolution STFT discriminator, batch 8–16, KID-vs-real tracked every N steps with early best-checkpoint selection (small-data GANs peak early then degrade).
- Waveform handling: keep the repo's phase-reuse STFT approach (no vocoder — clean-trained vocoders fail on noisy spectrograms, 2305.12460); residual magnitudes only, bounded by a residual-scale clamp so the translator cannot destroy content.
- Applied at probability `r` (config; default 0.5) so the pure-DSP path remains represented.
- **Gates to ship** (05): channel-probe accuracy improves vs DSP-only; ASR round-trip WER delta ≤ 2% absolute vs its input (ROSE guard); Tier 3 WER not worse.

### 2.5 Expansion workflow (`dataset/expand.py`)
The 1k→10k orchestration:
1. Input: labeled real manifest (audio + transcripts) + target size + category quotas (e.g. emergency 10%, rare_vocab 10%).
2. Split real set: calibration/train vs held-out test (speaker/station-aware split; test never feeds calibration).
3. Texts for the 9k synthetic: re-synthesize real transcripts (with voice/rate variation) + external JSONL for oversampled categories (other team supplies emergency/rare texts; we only consume). Quota logic from 02 §5.
4. Generate via CalibratedChannel; Tier 0 QC gates inline.
5. Emit combined manifest with `origin: real|synthetic` per record + stats report; fine-tuning consumes it with the synthetic→real curriculum flag (00 §6).

## 3. Config schema (Mode 2 body)

```yaml
mode: calibrated
calibration:
  corpus_dir: data/real/calibration      # or the full ~1k labeled set
  presets: runs/calib_v1/presets.jsonl   # produced by channel_fit
  noise_bank: runs/calib_v1/noise/
  station_mix: {KSDL_TOWER: 0.4, SEATTLE_CENTER: 0.3, KSLE: 0.3}  # optional; default: corpus empirical
  snr_jitter_db: {uniform: [-3, 3]}
  cross_station_prob: 0.1
residual:
  enabled: true
  checkpoint: runs/cut_v1/G_ema.pt
  apply_prob: 0.5
  residual_scale_max: 0.35
post_effects:            # shared Mode 1 primitives layered on top
  squelch: {prob: 0.8, gated_floor_prob: 0.6}   # gated_floor from corpus stats
  dropouts: {prob: 0.15}
  codec: {prob: 0.5, kind: mp3, quality: {uniform: [0.75, 0.95]}}
expansion:
  real_manifest: data/real/labeled/manifest.jsonl
  target_total: 10000
  category_quotas: {emergency: 0.10, rare_vocab: 0.10}
  holdout_frac: 0.15
```

## 4. Implementation roadmap

Each task is a clean handoff boundary with stated acceptance criteria. Prereqs: shared refactor S1–S3 from 02/03 (primitives + config), eval E1–E2 from 05.

- **M2.1 — Local corpus module** (`dataset/local_corpus.py`, `noise_harvest.py`). Ingest `data/real/*`, QC (dedupe, silence-only detection), station parsing from filenames, splits, VAD noise bank with stats. *Accept:* corpus manifest + noise bank for the 100 calibration clips; unit tests on synthetic fixtures. No GPU.
- **M2.2 — Channel fitting** (`channel_fit.py`). Statistics-matching fit first; presets.jsonl + fit-QC; audition script rendering one TTS sample through each of 10 random presets. *Accept:* fitted presets reproduce each clip's LTAS within tolerance; presets audibly differ across stations. GPU optional.
- **M2.3 — CalibratedChannel backend** (`channel/learned/backend.py`). Sampling model + shared post-effects; wired into builder + config; Tier 1 metrics vs real reference. *Accept:* channel probe ≤ 0.7 on calibration set; beats Mode 1 `matched` profile on Tier 1 distances.
- **M2.4 — Residual CUT** (`residual_train.py` + inference). Full small-data stack; KID tracking; gate evaluation vs M2.3. *Accept:* gates in §2.4; else feature-flagged off and documented. 5080; ~0.5–1 day training per run.
- **M2.5 — Expansion workflow** (`dataset/expand.py`). *Accept:* end-to-end 1k→10k dry run on synthetic stand-in labels; quota + provenance stats correct; Tier 0 discard < 15%.
- **M2.6 — Tier 3 validation.** Fine-tune Whisper-small per 05 protocol on Mode 1 vs Mode 2 vs mixed; decide default backend and `mix` weights. ~1–2 days on the 5080.
- **M2.7 (optional, later) — Diffusion/flow spike.** Only if M2.4 fails its gates or probe accuracy stalls > 0.7: small conditional flow-matching model, clean→real conditioned on preset params; time-boxed.

Risks: (1) statistics-only fitting may under-determine the nonlinearity — mitigated by upgrade path in M2.2; (2) unlabeled local clips mean local Tier 3 needs the labeled ~1k set from the collected data — until then benchmark WER runs on public ATC test sets; (3) CUT on spectrograms is less published than CycleGAN-VC — mitigated by keeping v1 CycleGAN as baseline and the DSP backbone as the floor.
