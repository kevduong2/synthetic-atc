# Phase 2 — Codebase Analysis

Status of the existing `atc-gan` repo against the two-mode brief (Mode 1 procedural, Mode 2 calibrated), what is reusable, what is missing, and where new components should live.

Companion docs: [00-research-findings.md](00-research-findings.md) (Phase 1 survey), [02-architecture.md](02-architecture.md) (shared design), [03-mode1-procedural-plan.md](03-mode1-procedural-plan.md), [04-mode2-calibrated-plan.md](04-mode2-calibrated-plan.md), [05-evaluation-plan.md](05-evaluation-plan.md).

## 1. What exists today

The repo is not a blank slate: it already implements a first version of both modes end-to-end, plus training/eval scaffolding.

| Area | Module | State |
|---|---|---|
| TTS | `atcgen/tts/synthesize.py` | Kokoro wrapper, 13 US/UK voices, speed 0.95–1.55×, engine-agnostic `TTSEngine` protocol. Solid; pitch/accent variation missing. |
| Procedural channel (≈ Mode 1) | `atcgen/channel/dsp.py` | `RadioChannelSim` + `ChannelParams.sample()`: narrowband resample, 300–3400 Hz bandpass, AGC wander, AM distortion/soft clip, static (white/pink or real noise beds) at 3–25 dB SNR, hum, crackle, squelch clicks, dropouts, heterodyne, co-channel interference, MP3 round-trip, pilot double-hop. Per-sample randomized. |
| Learned channel (≈ Mode 2) | `atcgen/channel/gan/` | Spectrogram CycleGAN (log-mag STFT, 512/128, phase reuse — no vocoder), ResNet generator + PatchGAN discriminator, LSGAN + cycle + identity losses, resume + per-epoch audition wavs. Inference via `GanChannel`, plus a `ChannelParams.mild()` DSP pass for per-sample diversity. |
| Text | `atcgen/text/` | FAA/ICAO phraseology grammar + `TextSource` protocol with JSONL adapter for the other team's transcripts. Out of scope for this effort but the plug-in seam matters. |
| Dataset builder | `atcgen/dataset/build.py` | text → TTS → channel (`dsp`/`gan`/`mix`/`clean`) → wav + `manifest.jsonl`; noise-only anti-hallucination samples (~3%); co-channel reuse of previous sample. |
| Real-corpus prep | `atcgen/dataset/real_atc.py` | HF corpora only (`jacktol/atc-dataset`, `Jzuluaga/uwb_atcc`): GAN domain-B export, noise-bed harvesting (quietest-window heuristic), eval loading. |
| Training/eval | `training/` | Whisper fine-tune with `--mix-real` (~1:1 upsampling), WER eval with ATC normalization (`niner→nine` etc.). |
| Tests | `tests/` | 21 unit tests over channel, GAN shapes, text, normalization. |

## 2. The local real dataset (new)

`data/real/calibration/` now holds 100 clips randomly sampled from `asr-demo`'s OLA libraries (50 from each source dir): KSDL tower (50), Seattle Center (25), KSLE tower (16), KSLE ground (9). Measured properties:

- **Format:** all 16 kHz, 16-bit mono WAV. Durations 0.5–13 s, median 4.1 s (100 clips ≈ 7.5 min total).
- **Bandwidth:** 98% of spectral power sits between ~200 Hz and a **median ~2.4 kHz upper edge** (p75 ≈ 2.6 kHz, max ≈ 2.97 kHz). The current sim's `bp_high` of 2800–3600 Hz is wider than any of these clips — a concrete miscalibration.
- **SNR (frame-energy estimate):** median ~23 dB, p25 ≈ 17 dB; a few clips are outliers — near-silent (peak 0.01, likely squelch tail/noise-only) or with near-digital-silence floors (SNR estimate >60 dB), which indicates **receiver squelch gating**: noise is not a continuous bed but is muted outside the carrier. The current sim adds continuous noise across the whole clip, including padding — a second miscalibration.
- **No transcripts.** These clips are unlabeled, so locally they can serve channel learning, noise-bed harvesting, and acoustic-match evaluation — not WER ground truth. (The full source libraries hold ~2.5k clips; ~1k labeled samples for the Mode 2 expansion use case are expected to come from the collected datasets the brief mentions.)

## 3. Gap analysis against the brief

### 3.1 Shared infrastructure

- **No config system.** Every knob lives either in code constants (`ChannelParams.sample()` distributions, `PILOT_DOUBLE_HOP_PROB`, `KOKORO_VOICES`, `speed_range`) or in ad-hoc CLI flags. The brief requires two configurable modes with a shared config schema — this is the biggest structural gap. → `atcgen/config.py` + YAML profiles (see 02-architecture).
- **Augmentation primitives are not composable.** `RadioChannelSim._hop` is one monolithic function; effects can't be individually enabled/reordered/reweighted from config, and Mode 2 wants to reuse individual primitives (e.g. squelch clicks, codec) around a learned core. → refactor `dsp.py` into per-effect primitives with a chain runner (behavior-preserving).
- **Voice-level variation incomplete.** Speed exists; pitch shift and accent coverage do not (Kokoro's 13 en voices give limited accent spread; ATC is accent-heavy). → TTS-layer augment stage (pitch/formant/tempo perturbation) + config-driven voice pools.
- **No sample-weighting hook.** `build_dataset` samples text uniformly; the brief's "oversample emergency/rare vocabulary" needs a weighting/quota mechanism at the builder level (the text itself remains the other team's scope) — e.g. per-record `weight`/`category` fields in the JSONL contract plus target category quotas in config.
- **Manifest is good but minimal.** Add generator provenance (mode, config hash, channel params used, TTS voice/speed, source real-clip references) for reproducibility and eval slicing.

### 3.2 Mode 1 (procedural)

Largely built; the gaps are calibration bounds, coverage, and structure:

- Parameter distributions don't match measured local data (bandpass too wide; SNR range 3–25 dB skews harsher than the measured median ~23 dB — defensible for robustness, but should become an explicit config choice, e.g. `harsh`/`matched` profiles).
- Missing artifacts seen in real clips/literature: squelch **gating** of the noise floor (noise mutes when carrier drops), PTT truncation of first/last phonemes, receiver AGC attack at squelch open, longer fading, transmitter mic/handset coloration (spectral tilt EQ), inter-transmission silence handling.
- No stochastic-order/skip control: chain order is fixed; effects toggle only via sampled params.
- Only MP3 codec; real delivery paths also include very low bitrate streams and resample chains.

### 3.3 Mode 2 (calibrated)

A v1 CycleGAN exists but predates the local dataset and the research review:

- **Trained on the wrong domain B.** `real_atc.py` only feeds HF corpora; there is no loader/prep path for local OLA clips (silence trimming, level normalization, squelch-tail removal, train/holdout split). Our actual target is the local receiver chain.
- **Determinism / diversity.** One trained G_ab = one "radio"; diversity currently comes only from the post-hoc mild DSP pass. No noise/condition input to the generator, no multi-checkpoint or per-station conditioning.
- **Method choice unvalidated.** GAN vs diffusion vs DSP-parameter-fitting vs noise-resynthesis was never evaluated; the research phase (00-research-findings) weighs these for the ~1k-clip regime and the design picks accordingly.
- **Small-data regime risks unaddressed:** no D augmentation/regularization for ~1k clips, no overfitting checks, fixed 1-s crops discard longer-range structure.
- **No expansion workflow.** The "1k real → 10k (9k synthetic, oversampled rare cases)" use case has no orchestration: nothing consumes a labeled real manifest, re-synthesizes its texts (plus rare-case oversampling) through the calibrated channel, and emits a combined train set.

### 3.4 Evaluation

Only downstream WER exists. Missing entirely: synthetic-vs-real acoustic match metrics (FAD-style embedding distance, channel-statistics comparison: LTAS, band edges, SNR/modulation distributions), a fixed audition/listening protocol, and per-slice WER (emergency, rare-vocab, station). → new `atcgen/eval/` module (see 05-evaluation-plan).

## 4. Reuse verdict and target layout

Keep (as-is or lightly refactored): TTS wrapper and protocol, all DSP effect implementations (reorganized into primitives), CycleGAN model/training loop (upgraded per Mode 2 plan), dataset builder skeleton + manifest format, noise-bed concept, training/eval scripts, tests.

Discard/replace: hardcoded parameter distributions (→ config), monolithic `_hop` (→ primitive chain), HF-only real-data assumption (→ local corpus module).

```
atcgen/
  config.py              # NEW  typed config schema + YAML load/validate (shared)
  channel/
    primitives.py        # NEW  per-effect functions extracted from dsp.py
    chain.py             # NEW  config-driven chain runner (Mode 1 core)
    (dsp.py retired)     #      effects ported into primitives.py, then deleted
    learned/             # NEW  Mode 2 (renamed from gan/; method per research)
  dataset/
    local_corpus.py      # NEW  OLA clip ingestion: QC, trim, split, noise beds
    real_atc.py          #      HF corpora (kept for benchmarking)
    build.py             #      + weighted sampling, provenance, expansion mode
    expand.py            # NEW  1k→10k real-dataset expansion orchestration
  eval/                  # NEW  acoustic-match metrics + report (05-evaluation-plan)
configs/                 # NEW  YAML profiles (mode1_*.yaml, mode2_*.yaml)
data/real/calibration/   # DONE 100 sampled OLA clips
```
