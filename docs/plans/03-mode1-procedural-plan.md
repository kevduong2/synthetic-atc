# Mode 1 Plan — Procedural Radio Channel (fully independent baseline)

Design and implementation plan for the procedural generator: a randomized DSP degradation chain over TTS output, with no dependency on collected data, varied enough that models trained on it generalize to unknown ATC audio conditions.

Grounding: [00-research-findings.md](00-research-findings.md) (esp. the channel-simulation survey, `docs/research/channel-simulation.md`); current-state gaps in [01-codebase-analysis.md](01-codebase-analysis.md) §3.2; shared interfaces in [02-architecture.md](02-architecture.md); metrics in [05-evaluation-plan.md](05-evaluation-plan.md).

## 1. Design position

Mode 1 is **domain randomization**, not domain imitation: per-sample independent draws over wide parameter ranges so the real world looks like "just another variation" (Tobin et al. 2017; Google's MTR pattern, arXiv:1808.05312 — uniform draws over a small transform menu, "zero or more perturbations" per utterance, is the published state of practice; learned augmentation policies have thin evidence for channel simulation). The literature says this is high-leverage: Bagat et al.'s three-step channel sim alone was worth 37% relative WER on synthetic ATC data (arXiv:2606.21340), and codec augmentation generalizes to unseen codecs (~20% relative protection, arXiv:1808.05312) — so we favor breadth of variation over exact fidelity to any one receiver.

Two named base profiles express the two legitimate uses:
- **`wide`** (default): generalization to unknown conditions — ranges at the union of literature values and beyond our local measurements (SNR 0–30 dB, band corners HP 200–400 / LP 2300–3600 Hz, all effects active with their probabilities).
- **`matched`**: distributions centered on the measured local calibration set (01 §2: upper band edge ~2.2–3.0 kHz, SNR skewed clean with median ~23 dB, squelch-gated floors) — the fair DSP baseline for Mode 2 comparisons and the profile for local-receiver-targeted sets.

A configurable fraction of samples stays fully clean or lightly degraded (MTR "zero effects" arm; also hedges the finding that heavy channel augmentation can hurt large Whisper variants, arXiv:2502.20311).

## 2. Refactor: primitives + chain

Extract each effect from `RadioChannelSim._hop` into a pure function in `atcgen/channel/primitives.py` with signature `effect(x, sr, rng, **params) -> x`, unit-testable in isolation:

Existing (moved, behavior-preserving): `narrowband_roundtrip`, `bandpass`, `agc_wander`, `am_distortion`, `soft_clip`, `dropouts`, `additive_noise` (white/pink/NoiseBank), `hum`, `crackle`, `heterodyne`, `squelch_clicks`, `cochannel_mix`, `codec_roundtrip`, `double_hop` (composition).

New primitives (gaps found in 01 §3.2 and the survey):
- `squelch_gate` — noise floor gated by carrier presence: attenuate/mute noise outside the transmission, with attack/release ramps and occasional squelch tail bursts. Directly motivated by the measured local clips (near-digital-silence floors) — the current sim's continuous noise is its most audible mismatch.
- `ptt_truncation` — clip 20–120 ms from utterance start/end (PTT pressed late / released early), with transcript unchanged (real ATC labels contain these).
- `mic_coloration` — random low-order EQ tilt/peaking (handset/boom-mic variation) applied pre-channel.
- `fading` — slow multiplicative envelope (0.2–2 Hz, a few dB) for mobile/edge-of-range signals; distinct from short dropouts.
- `agc_attack` — brief gain surge at squelch open (receiver AGC settling).
- `resample_chain` — extra 4–6 kHz decimation arm for aliasing character (survey takeaway 2).
- `codec_roundtrip` extended: bitrate-parameterized MP3 **{16, 23, 32, 64} kbps** + "none" (survey takeaway 5), keeping soundfile MP3; optional AAC via ffmpeg behind a config flag.

`atcgen/channel/chain.py` runs a config-declared list of `{primitive, prob, params(distribution specs)}` in order, with per-sample draws, returning the applied-params record for the manifest. Chain order is config data; a `shuffle_groups` option permits randomized ordering within marked groups (cheap extra variance). `dsp.py` is retired once its effects are ported (no production users — no compat wrapper); its tests are ported to per-primitive tests.

## 3. Parameter table (base `wide` profile)

Values reconcile the survey's literature ranges with the current code and local measurements; all are config, none are code constants.

| Effect | Prob | Parameters (distributions) | Basis |
|---|---|---|---|
| mic_coloration | 0.5 | tilt ±4 dB, 1–2 peaks ±6 dB | practitioner (no published params) |
| narrowband_roundtrip | 0.8 | sr ∈ {6k, 8k×2, 11k}; extra 4–6k arm p=0.1 | 1808.05312; Bagat 8k round-trip |
| bandpass | 1.0 | HP 200–400 Hz, LP 2300–3600 Hz | ICAO 300–2500 nominal; local measured LP ~2.4k median |
| agc_wander / agc_attack | 0.6 / 0.5 | strength 0–0.6; attack 50–200 ms | existing + physical |
| am_distortion + soft_clip | 1.0 | depth 0–0.25; drive 1–4; hard-clip p=0.05 | existing; survey takeaway 4 |
| dropouts | 0.2 | 1–4 drops, 10–50 ms | existing |
| fading | 0.15 | 0.2–2 Hz, 2–6 dB depth | new; physical |
| additive_noise | 1.0 | SNR 0–30 dB skewed clean (e.g. beta toward high); pink/white; bed_prob 0.6 when beds exist | 1808.05312; ATCO2 0–40 dB "majority clean"; local median ~23 dB |
| squelch_gate | 0.6 | gated-floor depth 20–60 dB, attack/release 5–50 ms, tail-burst p=0.3 | local measurement; no published params |
| hum / crackle / heterodyne | 0.3 / 0.6 / 0.08 | as existing; heterodyne 300–2000 Hz | existing; survey (whistle range) |
| squelch_clicks | 0.8 | as existing | existing |
| ptt_truncation | 0.25 | 20–120 ms head/tail | new; real-label realism |
| cochannel_mix | 0.1 | level 0.05–0.2, prev-sample source | existing |
| codec_roundtrip | 0.6 | MP3 {16,23,32,64} kbps ∪ none | 1808.05312 |
| double_hop (pilot) | 0.5 | second radio SNR 10–25 dB | existing |
| clean/light arm | 0.07 | skip most effects | MTR zero-effects arm; 2502.20311 caution |

The `matched` profile narrows: LP 2200–3000 Hz, SNR 12–32 dB, squelch_gate prob 0.8, codec on (LiveATC-delivery-like), bed_prob 0.8.

## 4. Voice-layer variation (shared stage, spec'd here)

Per research (00 §2): speaking-rate variation is well-supported (keep/extend 0.95–1.55×); pitch evidence is mixed — implement `pitch_semitones` (±2) at p≈0.5 but treat as an ablation knob, not a certainty; accent breadth is the known Kokoro limitation (13 en voices) — mitigations in priority order: (a) use every usable Kokoro voice incl. non-`a` lang codes rendering English, (b) config hook for a future VC/accent-conversion stage (Bagat's biggest win; out of Mode 1's no-real-data constraint but the interface should exist), (c) EQ/formant tilt for timbre spread. TTS-level QC (Tier 0 ASR round-trip) catches Kokoro failures on ATC-ese.

## 5. Config schema (Mode 1 body)

```yaml
mode: procedural
channel:
  profile: wide                  # wide | matched | named custom
  clean_arm_prob: 0.07
  chain:                         # optional explicit override of the profile's chain
    - {primitive: mic_coloration, prob: 0.5, tilt_db: {uniform: [-4, 4]}}
    - {primitive: narrowband_roundtrip, prob: 0.8, sr: {choice: [6000, 8000, 8000, 11025]}}
    - {primitive: bandpass, prob: 1.0, low: {uniform: [200, 400]}, high: {uniform: [2300, 3600]}}
    - {primitive: additive_noise, prob: 1.0, snr_db: {beta_scaled: [2, 1.2, 0, 30]},
       beds_dir: data/noise_beds, bed_prob: 0.6}
    - {primitive: squelch_gate, prob: 0.6, floor_db: {uniform: [-60, -20]}}
    # ... remaining effects per §3
  hops: {pilot_double_hop_prob: 0.5}
```

## 6. Implementation roadmap

Shared prerequisites (from 02): **S1** config system (`atcgen/config.py`, distribution specs, YAML profiles, resolved-config dump); **S2** primitives extraction + chain runner, `dsp.py` retired, tests ported to per-primitive tests (initial parameter values match current behavior; retuning is a separate commit); **S3** builder integration (config-driven, provenance manifest records, category quotas). Then:

- **P1 — new primitives**: `squelch_gate`, `ptt_truncation`, `mic_coloration`, `fading`, `agc_attack`, extended codec. Unit tests per primitive (energy/band/duration assertions on synthetic fixtures). *Accept:* each audibly correct in an audition script; tests green. CPU only.
- **P2 — profiles**: `wide` + `matched` YAMLs per §3; smoke set (100 samples each) + Tier 1 stats vs the calibration clips. *Accept:* `matched` medians inside real p10–p90 for band edge, SNR, LTAS; `wide` strictly broader than `matched` on every histogram.
- **P3 — voice layer**: pitch/tempo/EQ-tilt stage + config; VC-stage interface stub. *Accept:* ablation configs exist (pitch on/off).
- **P4 — validation**: 5k-sample `wide` and `matched` sets → Tier 2 channel probe + Tier 3 fine-tune per 05 protocol vs the `legacy` chain. *Accept:* documented WER table; regressions vs legacy investigated before switching defaults.

Risks: over-aggressive `wide` ranges could hurt if the eval model grows (2502.20311) — mitigated by the clean arm + profile ablations; squelch-gate params are unmeasured in literature — mitigated by estimating gate statistics from the calibration clips (shares Mode 2's corpus stats code) and by perceptual audition.
