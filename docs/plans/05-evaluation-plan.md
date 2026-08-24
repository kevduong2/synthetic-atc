# Evaluation Plan — Synthetic ATC Audio Quality

How we decide whether generated audio is good, whether Mode 2 beats Mode 1, and when a generator version is fit to ship into training runs. Sources cited by arXiv ID; full context in [00-research-findings.md](00-research-findings.md).

## 1. Principles

- **The end metric is downstream WER** of Whisper fine-tuned on the synthetic data, evaluated on *real* ATC test sets. Everything else is a cheap proxy to iterate faster than a fine-tune cycle.
- **Match distributions, not maximize quality.** Synthetic audio should match the real clips' distribution of SNR, bandwidth, loudness, and MOS scores — a synthetic set with *higher* DNSMOS than the real set is miscalibrated (2606.29031: degrading perceptual quality is what closed the gap).
- **Preserve linguistic content.** ROSE (2312.06118) showed generative processing of ATC audio can improve perceptual metrics while destroying ASR-relevant features. Every generative (Mode 2) component must pass ASR-consistency checks.
- **Small reference set caveat.** Our local reference is ~100 clips now, ~1k later. FAD is biased at small n; prefer KID/KAD and use sample-size extrapolation (2311.01616) when comparing across set sizes.

## 2. Metric tiers

### Tier 0 — per-sample QC gates (run inside generation, block bad samples)
| Check | Rule | Source |
|---|---|---|
| ASR round-trip | Whisper (pretrained, non-fine-tuned) transcribes the *clean TTS* and the *degraded* sample; discard if degraded-vs-transcript WER > threshold (start 50%, per Bagat's recipe which discarded ~35% of accent-converted clips) | 2606.21340, 2508.21631 |
| Clipping/silence | reject digital clipping > 1% of samples, all-silence, NaN/Inf | — |
| Duration/level | duration within configured bounds; loudness within target window | — |

Log discard reasons and rates per run (`stats.json`); a rising discard rate is itself a regression signal.

### Tier 1 — distribution match (minutes; run per generator version)
Computed on N≥500 synthetic samples vs the real calibration set:

- **Channel statistics** (`atcgen/eval/channel_stats.py`): long-term average spectrum (LTAS) distance; spectral-edge distribution (freq below which 98% power sits — real median ≈ 2.4 kHz); frame-energy SNR estimate distribution; loudness/peak distributions; modulation spectrum (4 Hz-region syllable-rate energy, sensitive to speaking-rate realism). Report per-statistic Wasserstein distances + overlay plots.
- **Embedding distances**: **KAD/KID (primary)** and FAD (secondary) on two embedding families — CLAP and WavLM mid-layers (WavLM layers encode channel info, 2501.05310). VGGish is known-poorly correlated (2311.01616); don't use it.
- **DNSMOS distribution match**: Wasserstein distance between synthetic and real DNSMOS score distributions (not the mean).

### Tier 2 — channel probe (hours; per candidate generator)
Train a small logistic/MLP classifier on frozen WavLM embeddings: real vs synthetic, k-fold. **Target: accuracy near chance (≤0.65 acceptable, ≤0.55 good).** A high-accuracy probe both quantifies the gap and — via feature inspection — localizes it (which layers/bands separate the domains). This is the primary iteration metric for Mode 2 and the harshest test for Mode 1.

### Tier 3 — downstream WER (day-scale; per release candidate)
Protocol fixed once so results are comparable across versions:

1. Fine-tune `whisper-small.en` on the candidate synthetic set under **two regimes**: synthetic-only, and synthetic→real curriculum (synthetic first, real last — 2408.09215, 2606.17662).
2. Evaluate on: (a) held-out **local labeled real data** when available (primary), (b) ATCO2-1h + UWB-ATCC test via existing `training/evaluate.py` (public benchmark anchor), (c) speaker-split where possible (random splits leak — WhisperATC's 1.17% vs 3.88%).
3. Report raw **and** ATC-normalized WER (normalization alone moves zero-shot WER from ~72% to ~29%; never compare across normalization schemes), plus:
   - **callsign accuracy** (callsign-bearing slice; aggregate WER under-measures operational risk),
   - **per-category WER** (emergency, rare_vocab, routine — the oversampling target must be shown to pay off),
   - **hallucination rate** on a noise-only test slice (fraction of non-empty hypotheses on speech-free audio; Whisper hallucinates on ~40% of non-speech, 2501.11378).
4. Baselines to beat, in order: zero-shot Whisper-small; fine-tuned on real-only; Mode 1 synthetic; Mode 2 synthetic; mixed modes.

#### Protocol as implemented (E3)

Run these commands on the 5080 box with the model and datasets already cached. The two fixed baselines are zero-shot `whisper-small.en` and real-only fine-tuning:

```bash
uv run python training/evaluate.py --model openai/whisper-small.en --dataset real --out reports/zero_shot_small_en.json
uv run python training/finetune_whisper.py --real-only --model openai/whisper-small.en --out runs/whisper_real_only --epochs 3 --batch-size 16 --fp16
```

Train every candidate under both required regimes. `--manifest` may be repeated for multiple synthetic manifests; `--real-manifest` may likewise be repeated to use local labeled-real training manifests instead of the cached public real train split. `--mix-real` remains available as the joint shuffled ~1:1 ablation, but is not a substitute for either fixed regime.

```bash
# synthetic-only
uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl --model openai/whisper-small.en --out runs/whisper_synthetic_only --epochs 3 --batch-size 16 --fp16

# synthetic-first -> real-last curriculum (3 epochs in each phase)
uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl --curriculum --model openai/whisper-small.en --out runs/whisper_curriculum --epochs 3 --batch-size 16 --fp16

# optional joint-mixing ablation
uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl --mix-real --model openai/whisper-small.en --out runs/whisper_joint --epochs 3 --batch-size 16 --fp16
```

Evaluate each resulting checkpoint on the held-out local labeled-real manifest (primary) and the cached public ATCO2-1h + UWB-ATCC test dataset anchor. Evaluation manifests carry `category`; include their empty-reference `category: "noise"` records so hallucination rate is measured in the same report.

```bash
uv run python training/evaluate.py --model runs/whisper_curriculum --dataset data/eval/local_real/manifest.jsonl --out reports/whisper_curriculum_local.json
uv run python training/evaluate.py --model runs/whisper_curriculum --dataset real --out reports/whisper_curriculum_public.json
```

Every `--out` file is one JSON object with this stable E3 shape (WER values are ratios, not percentages; unavailable slice metrics are `null`):

```text
schema_version, model, dataset
samples: {total, speech, noise_only}
wer: {raw, atc_normalized}
per_category.<category>: {samples, wer: {raw, atc_normalized}}
callsign: {samples, wer: {raw, atc_normalized}, reference_sequences,
           exact_sequences, token_accuracy}
hallucination: {samples, non_empty_hypotheses, rate}
```

Noise-only records are excluded from aggregate, category, and callsign WER. Callsign accuracy is exact normalized token-sequence reproduction. Results are comparable only when produced with the same normalization scheme; raw and ATC-normalized WER must never be compared to one another.

### Listening protocol (qualitative, every version)
Fixed 20-sample audition sheet per generator version (same seeds/texts across versions): 5 routine, 5 emergency, 5 extreme-parameter draws, 5 noise-only, rendered next to 5 random real clips. One-page HTML report with players + Tier 1 plots. Human A/B spot-check: can a team member pick the synthetic one >70% of the time? (Informal probe complement.)

## 3. Acceptance criteria

A generator version is **train-ready** when: Tier 0 discard rate < 15%; every Tier 1 statistic's synthetic median falls within the real set's p10–p90; channel probe ≤ 0.65; and Tier 3 shows synthetic+real ≥ real-only baseline (the literature's realistic bar: synthetic-only plateaus ~1.5–2× real-only WER — 2508.21631 — so synthetic-only parity is *not* required).

Mode 2 justifies its complexity over Mode 1 only if it improves the channel probe **and** Tier 3 WER on the local test set; otherwise ship Mode 1 (the Bagat ablation shows plain DSP channel simulation already carries most of the gain: 53.9%→33.8% WER).

## 4. Module layout

```
atcgen/eval/
  qc.py             # Tier 0 gates (used by the builder at generation time)
  channel_stats.py  # Tier 1 statistics + plots
  embed_dist.py     # KAD/KID/FAD over CLAP + WavLM (extra dep group [eval])
  probe.py          # Tier 2 real-vs-synthetic probe
  report.py         # HTML report: stats, plots, audition players
scripts/eval_synthetic.py   # run tiers 0-2 + report against a manifest + real ref dir
```
Tier 3 reuses `training/finetune_whisper.py` + `training/evaluate.py` (extended with per-slice WER and hallucination-rate flags).

## 5. Roadmap

1. **E1 — QC gates + channel stats** (no GPU): `qc.py`, `channel_stats.py`, `report.py`; wire Tier 0 into the builder behind config. Immediately usable to recalibrate Mode 1 against the 100 real clips.
2. **E2 — embeddings + probe**: `embed_dist.py`, `probe.py` ([eval] extra: `laion-clap`, WavLM via transformers). Validate embedding choice by confirming the probe separates *clean TTS* from real with ~100% accuracy (sanity), then measure current DSP output.
3. **E3 — Tier 3 protocol**: per-slice WER + hallucination-rate in `training/evaluate.py`; curriculum flag in fine-tune script; document the fixed protocol in this file.
4. **E4 — regression harness**: `eval_synthetic.py` one-command run; store per-version JSON so versions are diffable.
