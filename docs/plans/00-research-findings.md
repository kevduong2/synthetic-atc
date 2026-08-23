# Phase 1 — Research Findings (2026 state of the art)

Synthesis of five literature surveys conducted 2026-08-23. Full sourced reports live in `docs/research/`:

- [tts-augmentation-for-asr.md](../research/tts-augmentation-for-asr.md) — synthetic speech augmentation for ASR
- [channel-simulation.md](../research/channel-simulation.md) — radio/VHF channel simulation
- [generative-channel-modeling.md](../research/generative-channel-modeling.md) — GAN vs diffusion vs alternatives for learned channels
- [atc-asr-landscape.md](../research/atc-asr-landscape.md) — ATC corpora, benchmarks, ATC-specific synthetic-data work
- [training-recipes.md](../research/training-recipes.md) — small-data GAN and Whisper fine-tuning practice

This file states the conclusions that drive the design; citations abbreviated to arXiv IDs (full context in the reports).

## 1. The anchor prior art

**Bagat, Zhang, Yamagishi, Illina, Vincent — "Synthetic Audio Generation Framework for ATC Speech Recognition," Interspeech 2026 (arXiv:2606.21340)** — the paper the repo README references — is verified and is the direct predecessor of this project. F5-TTS + kNN-VC voice conversion + L1→L2 accent conversion + a *deliberately minimal* DSP channel sim (8 kHz round-trip, 200 Hz high-pass, real ATCO2 noise beds). Results (Whisper-small, ATCO2 test): zero-shot 63.3%; real-only fine-tune 22.7%; synthetic-only 24.2%; best synthetic+real 21.6%. Two ablation facts anchor everything below:

1. **Channel simulation alone: 53.9% → 33.8% WER (37% relative).** The single highest-leverage realism component.
2. **Accent diversity > speaker diversity** (L1→L2 accent conversion was their best addition); channel sim is necessary for synthetic data to be useful at all.

Their QC recipe — Whisper round-trip on every clip, discard >50% WER (~35% of accent-converted clips rejected) — is adopted in our eval plan.

Open ground the surveys confirmed (searched, nothing published): learned/GAN channel transfer for ATC ASR; emergency-phraseology (MAYDAY/PAN-PAN) recognition or test sets; noise-only hallucination-control training for ATC specifically. All three are things this project does.

## 2. What makes synthetic audio useful for ASR

- **"Match the channel, not the studio."** Degrading synthetic audio toward the target channel is what closes the gap — RIR/channel augmentation helped ASR *because* it tanked perceptual quality (2606.29031); vocoder/TTS fidelity barely matters for ASR utility (2407.17997). Perceptual quality is a distribution-matching target, not a maximization target.
- **Synthetic-only has a floor** (~1.5–2× real-only WER even after pipeline optimization, 2508.21631); small amounts of real data anchor large synthetic corpora. The goal is stretching real hours, not replacing them.
- **Mixing ratio is a hyperparameter, not a constant**: evidence spans 10–30% synthetic optimal (2606.29031) to ~1:1 (IEEE 11252042, abstract-only); pure-synthetic fine-tuning degrades Whisper (2606.17662). Gains from added synthetic appear early (+20%) and saturate (~+80%) (2606.17662).
- **Curriculum beats naive mixing**: synthetic-first → real-last sequential fine-tuning outperformed joint mixing (2408.09215, 2606.17662).
- **Diversity ordering**: accent > new plausible timbres (VC/latent mixup, 2511.20534) > speaking rate (well-supported) > pitch (mixed evidence — 2503.08954 found it ineffective, 2606.29031 recommends it) > spectrogram-level perturbation (weakest, 2511.20534).
- **Label QC is mandatory**: TTS silently corrupts labels; generator-verifier ASR round-trip filtering is standard (2508.21631, 2606.21340).
- **Whisper hallucinates on non-speech** (~40% of noise inputs, 2501.11378); noise-only empty-transcript training samples (already in the repo) are the accepted counter (2408.16589), plus optional bag-of-hallucinations post-filtering.

## 3. Procedural channel simulation (Mode 1 basis)

- Physical anchor: ICAO VHF DSB-AM audio band ≈ **300–2500 Hz** nominal; ATCO2 measured live VHF at **SNR 0–40 dB, majority clean** — matching our local measurement (median ~23 dB, band edge ~2.4 kHz).
- Published state of practice is **uniform random draws over a small transform menu, zero-or-more effects per utterance** (Google MTR, 1808.05312); learned augmentation policies have no demonstrated edge for channel sim. Domain-randomization logic favors over-wide ranges over over-narrow.
- **Codec augmentation transfers to unseen codecs** (~20% relative WER protection at MP3 23 kbps; Opus/SBC generalization, 1808.05312) — exact LiveATC bitrate matching is non-critical; MP3 {16–64} kbps ∪ none suffices.
- **No published parameter distributions exist** for squelch tails, PTT clicks, heterodynes, AGC dynamics — implement physically, estimate statistics from our own clips, validate perceptually. (DARPA RATS physically retransmitted 400 h rather than simulate — a caution about fidelity claims; no rigorous sim-vs-real comparison exists.)
- **Caution**: heavy channel augmentation helped Whisper-small but was neutral-to-harmful for Whisper Large v3 Turbo in one ATC study (2502.20311) — keep a clean arm and A/B per eval model.

## 4. Learned channel modeling (Mode 2 basis)

Method comparison for the ~1k-clip regime (full table in the generative report):

- **DSP-hybrid wins on every axis but residual realism**: MicAugment (2010.09658) proves per-clip differentiable channel identification from seconds of audio; persoDA (2501.09113) proves VAD-harvested noise + SNR-matched mixing. ~1k clips become ~1k interpretable channel presets — diversity by sampling, no mode collapse, CPU inference.
- **GANs are viable at this scale** (CycleGAN-family results at ~3 *minutes* of data; clean→noisy radio translation demonstrated on RATS/VHF, 2305.12460) but pure CycleGAN has structural problems for us: deterministic mapping averages the corpus into one "radio"; cycle-consistency on information-destroying mappings induces steganographic cheating (1712.02950); clean-trained vocoders fail on noisy spectrograms (2305.12460 — validates the repo's phase-reuse choice). **CUT** (2007.15651) is the data-efficient one-sided successor — no cycle, half the cost.
- **Small-data GAN stack is settled practice**: ADA (2006.06676) or DiffAugment (2006.10738) + R1 (γ swept log-scale, 1801.04406) + generator EMA (1806.04498) + multi-resolution STFT discriminator (2103.05236); KID over FID at n≈1k (1801.01401); best checkpoint arrives early.
- **Diffusion for degradation-direction modeling is a literature gap** at this scale — attractive diversity, unproven, costliest; flow-matching/consistency has made inference cheap (2509.21522 et al.) but our generation is offline anyway. Defer.
- **ROSE negative result (2312.06118)**: generative processing of ATC audio improved perceptual metrics while *degrading* ASR accuracy → any learned component must pass ASR-consistency gates.

→ Mode 2 = DSP-hybrid backbone (per-clip fitted presets + real noise bank) with a gated residual CUT translator; diffusion deferred. Detail: [04-mode2-calibrated-plan.md](04-mode2-calibrated-plan.md).

## 5. Evaluation of synthetic-vs-real match

- **KAD/KID preferred over FAD** with a ~1k reference set (FAD small-n bias; 2311.01616 for extrapolation); embedding choice dominates — use CLAP + WavLM (WavLM layers encode channel information, 2501.05310), never VGGish.
- **Channel probe**: real-vs-synthetic classifier on frozen SSL embeddings; near-chance accuracy = domain match. Doubles as a diagnostic.
- **DNSMOS as distribution match**, not maximization.
- **End metric: downstream WER** on real test sets, with ATC-aware normalization (normalization alone moves zero-shot WER ~72%→~29%, WhisperATC), speaker-aware splits (random splits leak: 1.17% vs 3.88% on ATCOSIM), plus callsign accuracy and per-slice WER — aggregate WER under-measures operational risk.
- Detail: [05-evaluation-plan.md](05-evaluation-plan.md).

## 6. Benchmarks and data context

- Fine-tuned WER floors to date: ~13.5% ATCO2 (Whisper large-v2), ~15% on the jacktol ATCO2+UWB mix (medium.en), 1–4% on clean simulated ATCOSIM, <5% stated operational target. Zero-shot Whisper is 60–95% on ATC — the domain gap is enormous and mostly bridgeable.
- Free data for benchmarking/domain-B: UWB-ATCC (~20 h), ATCOSIM (10.7 h), ATCO2-1h, jacktol HF mix (deprecation notice — check successor), **ATCCaps (2606.22399, 203 h with ADS-B-derived callsigns — evaluate license; potentially a large domain-B/benchmark upgrade)**.
- Whisper fine-tuning recipe (training report): compare full FT of small vs LoRA of medium; lr 1e-5 / ~1e-3 (LoRA); keep encoder trainable (the shift is acoustic); SpecAugment with the zero-masking/padding caveat; ~10–20% general-English rehearsal if forgetting appears; noise-only samples ~5–10%. All jobs fit the 5080; full program ≈ 1–2 weeks of compute.
- **US-vs-Europe matters**: a European-trained ATC model degraded to ~30% WER on American audio — phraseology/accent match of text and voices is as important as channel realism. Our corpora (KSDL/KSLE/Seattle Center) are US; public benchmarks are mostly European — expect and report this skew.

## 7. Design decisions locked by research

| Decision | Basis |
|---|---|
| Keep and recalibrate the DSP chain as Mode 1; randomized wide ranges + clean arm | §3 |
| Mode 2 backbone = per-clip fitted channel presets + VAD noise bank; residual CUT gated on metrics; defer diffusion | §4 |
| ASR round-trip QC gate on all generated audio | §1, §2 |
| Noise-only empty-transcript samples retained (~3–10%) | §2 |
| Eval = KAD/KID + channel probe + DNSMOS-distribution + sliced WER protocol | §5 |
| Synthetic→real curriculum; mixing ratio exposed as sweep, not constant | §2 |
| Speed/tempo variation priority; pitch as ablation; VC/accent-conversion interface reserved (biggest known upside not yet in scope) | §2, §1 |
