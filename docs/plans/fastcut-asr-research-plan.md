# FastCUT radio-channel augmentation for ATC ASR

- **Status:** research conclusion and implementation plan — revised 2026-08-25
  after adversarial review (code audit, methodology, feasibility)
- **Research cutoff:** 2026-08-25
- **Project status:** pre-release
- **Primary decision:** proceed with a descoped, bounded go/no-go on the
  corrected residual (2–3 weeks), preregistered as development/feasibility
  evidence; run the full comparison program only on a “go,” and reserve
  confirmatory claims for prospectively collected, session-disjoint real data

## Executive conclusion

We should proceed, but first only through a bounded go/no-go. The project
already contains most of the necessary model and inference path, but it has not
produced a trained GAN checkpoint yet. As of this audit,
`runs/cut_v1/G_ema.pt` does not exist. The large saved `model.safetensors`
files under `runs/matrix_v1/` are fine-tuned Whisper ASR checkpoints, not audio
generators.

The intended system is:

```text
ATC text and entities
        |
        v
Kokoro TTS -- immutable clean waveform
        |
        +-------------------------------+
        |                               |
        v                               v
procedural radio DSP             calibrated radio DSP
                                        |
                                        v
                              bounded FastCUT residual
                                        |
        +-------------------------------+
        v
verified synthetic (audio, transcript, entity) pairs
        |
        v
fine-tune and evaluate the ASR model on real ATC speech
```

The GAN is therefore not expected to invent speech or transcripts. Kokoro
does that. The GAN translates a synthetic radio-channel representation toward
the distribution of real receiver audio while PatchNCE and explicit content
guards try to preserve what was said. The ASR model is then trained on the
resulting labeled audio.

The strongest research hypothesis is deliberately narrower than “GAN beats
DSP”:

> A moderate, content-preserving unpaired target-audio residual after calibrated
> DSP covers acoustic structure missed by the fitted simulator. A controlled
> mixture of DSP and DSP-plus-residual examples will improve real ATC ASR
> development performance more reliably than replacing DSP with GAN output.

This is supported by the closest published evidence:

- The 2026 synthetic-ATC study found that acoustic simulation was essential,
  but also found that blindly combining every synthetic pipeline did not help.
- Simu-GAN and a follow-up radio-noise study show that one-way CUT-style
  translation can learn UHF/VHF degradations from minutes of unpaired data and
  improve downstream noisy-speech recognition.
- UNA-GAN, CADA-GAN, and the 2026 URSA-GAN work extend that speech-specific
  line with target identity NCE, explicit channel/noise codes, FiLM, and
  controlled stochasticity.
- DENT-DDSP shows that an explicit, controllable radio model is a strong
  alternative. That supports our hybrid DSP-plus-learned-residual design.
- Current synthetic-ASR work consistently finds that downstream real-speech
  WER is the arbiter; perceptual quality and embedding distance are useful
  diagnostics but unreliable winner-selection metrics.

The current implementation is technically capable of MPS smoke runs. A real
two-step training smoke and a separate WavLM-KID smoke completed with finite
values and wrote loadable checkpoints; the targeted channel/FastCUT tests had
103 passes. These are path-validity checks, not trained models or performance
results. The observed M3 timings are also non-authoritative because the Mac GPU
was in concurrent use during measurement.

The MacBook Pro M3 is the default machine for implementation, smoke tests,
short pilots, and successive-halving sweeps. The 5080 is optional acceleration
for promoted long runs and production generation. Nothing in this plan treats
the 5080 as an experiment gate.

Before a serious GAN run, however, we must repair four scientific blockers and
pass one premise check:

1. Holdout receiver clips currently leak into the fitted DSP presets and noise
   bank used to construct the GAN's training input.
2. GAN checkpoints are selected using KID against real and synthetic training
   examples, not independent validation examples.
3. The Domain-A cache can silently reuse stale audio after inputs change.
4. The trainer writes optimizer state but cannot resume it, and an enabled but
   missing inference checkpoint silently falls back to DSP.
5. Calibrated DSP has never been tested downstream against procedural matched;
   MatrixV1 used procedural audio only. Its stored fidelity snapshots are
   currently worse than procedural matched: 0.005116 versus 0.003307 WavLM
   KID, and 0.001283 versus 0.001120 CLAP KID. Before investing in a residual
   on top of calibrated DSP, run a cheap one-seed G1-versus-G3 SFT baseline on
   a shared 500–1,000-content pool. Calibrated DSP must be at least competitive
   downstream. If it is materially worse, fix or abandon that backend before
   training its residual.

Launching a long run before these checks and fixes could give us a valid-looking
checkpoint selected by contaminated metrics. Phase 0 below repairs those
issues directly.

### Scope and claim discipline

1. The full roadmap as originally drafted is roughly a four-to-six-month solo
   program. This revision cuts the critical path to a two-to-three-week go/no-go
   and makes the platform work conditional on a positive screen.
2. With 99 clips concentrated in a few same-day capture sessions and only a
   427-row remaining public tail, every result the current program can produce
   is development/feasibility evidence. Confirmatory language is reserved for
   a future prospectively collected, session-disjoint test.
3. What the model learns from unpaired data is an **unpaired target-audio
   residual**, not a verified channel transform. Domain B differs from Domain A
   in speaker, accent, phraseology, and pacing as well as channel. Channel
   attribution requires the diagnostics in §6.3 and, eventually, repeated
   known probes or near-parallel recordings across receivers (§5.4).

## 1. Questions this plan answers

### Do we currently have a trained GAN that can generate audio?

No. We have:

- a locally cached Kokoro TTS model that generates clean speech;
- procedural and calibrated DSP radio-channel generators;
- a 4.4-million-parameter bounded residual generator and a working trainer;
- inference wiring that will load `G_ema.pt` after one is trained; and
- multiple saved Whisper ASR checkpoints from the latest experiment matrix.

We do not have a saved FastCUT/CUT/CycleGAN generator checkpoint in `runs/`.
The residual remains disabled in
[`configs/mode2_default.yaml`](../../configs/mode2_default.yaml), which is the
correct safe behavior until a checkpoint passes the gates in this plan.

### Was the project supposed to use a GAN to train ASR?

Yes, with an important distinction: the GAN is the radio-channel augmenter,
not the recognizer and not the text-to-speech model. The repository's initial
README described grammar to TTS to DSP-or-CycleGAN to labeled pairs to Whisper.
The intervening work built and validated the grammar, DSP calibration,
verification gate, entity metrics, ASR SFT/GRPO recipe, and experiment harness.
That work gave us a usable baseline, but the neural channel branch was never
run to a real checkpoint.

The newer implementation has already moved in a good direction from the
original free CycleGAN: it is one-way, PatchNCE-constrained, and bounded around
calibrated DSP. The next milestone is to make that experiment scientifically
clean and run it.

### Is FastCUT state of the art today?

Not by itself. FastCUT is a fast 2020 variant of CUT. For this project, the
more current and more directly relevant reference points are:

- Simu-GAN for one-way unpaired clean-to-radio/noisy speech translation;
- UNA-GAN for an even smaller-data speech-CUT result;
- the 2023 comparison of GANs for noisy-speech simulation on UHF/VHF data;
- CADA-GAN for channel-conditioned speech translation;
- URSA-GAN as the newest direct channel-plus-noise method, accepted to IEEE
  TASLP, with the caveat that its large pretrained stack is not the right first
  local implementation;
- DENT-DDSP for controllable, data-efficient radio simulation; and
- the 2026 ATC synthetic-data framework for the complete TTS/VC/channel-to-ASR
  pipeline.

Accordingly, the first model should retain FastCUT's computational advantages
while correcting our implementation toward the speech-specific Simu-GAN
formulation. A channel-conditioned FiLM variant is a second-stage candidate,
not the starting point.

## 2. Current repository and artifact audit

### 2.1 What is already implemented

| Capability | Current implementation | Assessment |
|---|---|---|
| Clean speech | Kokoro multi-voice TTS | Operational; keep as the content source for this milestone. |
| Procedural channel | [`atcgen/channel/chain.py`](../../atcgen/channel/chain.py) | Operational and already used in MatrixV1. |
| Calibrated channel | [`atcgen/channel/learned/backend.py`](../../atcgen/channel/learned/backend.py) | Operational; fitted presets, real noise, and post-effects. |
| Residual generator | [`atcgen/channel/learned/residual.py`](../../atcgen/channel/learned/residual.py) | 4.4M-parameter bounded log-STFT residual ResNet. |
| GAN trainer | [`atcgen/channel/learned/residual_train.py`](../../atcgen/channel/learned/residual_train.py) | PatchNCE, LSGAN, DiffAugment, lazy R1, EMA, multi-scale discriminator, KID. |
| Generation integration | calibrated backend applies optional residual before post-effects | Operational if a valid checkpoint exists. |
| ASR gate and entities | [`atcgen/gate/`](../../atcgen/gate/) and [`atcgen/eval/entity_metrics.py`](../../atcgen/eval/entity_metrics.py) | Useful content-preservation and safety infrastructure. |
| ASR training | [`training/recipe.py`](../../training/recipe.py) | Supports real/synthetic training, but not exact per-generator mixture weights. |
| Experiment matrix | [`scripts/run_matrix.py`](../../scripts/run_matrix.py) | One synthetic pool only; needs paired multi-pipeline support. |
| Legacy CycleGAN | [`atcgen/channel/gan/`](../../atcgen/channel/gan/) | Orphaned from normal generation; code remains as project history only. |

The current residual model works on log-magnitude STFTs with `n_fft=512`, a
128-sample hop, and 256 retained frequency bins. It predicts a `tanh`-bounded
additive residual and reuses the input phase when reconstructing audio. The
default residual bound is 0.35 in the normalized log-magnitude representation.
The generator is deterministic; variability currently comes from TTS, fitted
presets, noise selection, and post-effects rather than a generator latent.

The present trainer is best described as **FastCUT-inspired**, not an exact
reproduction of official FastCUT or Simu-GAN:

- it uses FastCUT's high source PatchNCE weight and no target identity NCE;
- it adds a bounded residual rather than predicting an unconstrained target;
- it uses average-pooled views of one STFT for its “multi-resolution”
  discriminator rather than separate STFT parameterizations;
- its source/key PatchNCE branch is not detached in the same way as common CUT
  implementations;
- it has no latent input, target condition, or channel reconstruction loss;
- it trains on roughly one-second crops and therefore emphasizes local texture
  over long squelch/carrier dynamics; and
- source phase reuse prevents genuinely independent phase/noise generation.

Two details should be corrected rather than treated as experimental choices.
The PatchNCE key/source features should be detached as in the reference CUT
implementation; otherwise both sides of the contrastive target can move
together. The current DiffAugment time roll is circular, which wraps an
utterance tail to its beginning, and its “gain” multiplies normalized
log-magnitudes rather than applying physical waveform gain. Replace these with
zero-padded shared time shifts/crops and true pre-STFT gain.

Those are testable design choices, not automatically defects. The plan keeps a
corrected version of this small model as the first baseline because it is easy
to run locally and structurally limits semantic damage.

Audit verification status:

- 103 targeted channel/FastCUT tests pass;
- the full suite has 628 passes and one unrelated reproducibility assertion
  failure caused by the time-varying `lineage.built_at` field, not different
  generated audio;
- a real-asset two-step MPS run completed with finite GAN, NCE,
  discriminator, and R1 values and wrote all expected checkpoint files; and
- a separate MPS smoke exercised WavLM KID and selected a loadable `G_ema.pt`.

Those temporary smoke artifacts prove execution paths only. They are not
project checkpoints and were not placed in `runs/cut_v1/`.

### 2.2 Current local data

The local calibration manifest contains 99 clips and 450.75 seconds of audio:

| Station | Clips |
|---|---:|
| `KSDL_TOWER` | 50 |
| `KSLE_GROUND` | 8 |
| `KSLE_TOWER` | 16 |
| `SEATTLE_CENTER` | 25 |
| **Total** | **99** |

The current manifest calls 84 clips `train` and 15 `holdout`. There are 96
accepted fitted presets—81 train-derived and 15 holdout-derived—and 155
harvested noise segments—128 train-derived and 27 holdout-derived. The clean
TTS assets include 200 matched and 200 wide cached sources plus smaller smoke
variants.

This is enough for feasibility experiments: Simu-GAN reports useful radio
translation from minutes of target audio. It is not enough for a strong claim
about receiver/session generalization. Most clips per station come from the
same date and contiguous capture period. The random clip-level holdout contains
nearby portions of the same receiver sessions.

Existing augmented 200-clip fidelity snapshots also do not establish calibrated
DSP as the current winner:

| Stored comparison | Procedural matched | Calibrated DSP | Lower is better |
|---|---:|---:|---|
| WavLM KID | 0.003307 | 0.005116 | procedural matched |
| CLAP KID | 0.001120 | 0.001283 | procedural matched |

Earlier no-voice-augmentation measurements favored calibrated DSP. The change
in ordering is largely associated with the shared pitch/tempo frontend,
reinforcing why the new experiment must use identical cached clean/front-end
audio for every branch and evaluate out of sample.

### 2.3 Leakage and evaluation failures to fix

#### The holdout is already in the training transform

[`channel_fit.fit_corpus`](../../atcgen/channel/learned/channel_fit.py) fits all
manifest rows without filtering the split. All 15 nominal holdout clips are
represented in the accepted preset file. Noise harvesting likewise ignores
the split: 27 of the 155 saved noise segments come from nominal holdout clips.
The GAN's Domain B loader correctly selects 84 training clips, but its Domain A
audio is rendered using presets and noise derived from both sides.

The implication is broader than “one option is missing.” Channel validation
cannot be trusted until presets, noise banks, GAN training, and checkpoint
selection all share the same split policy.

#### KID currently selects on training data

The trainer passes the same real Domain-B training paths to `KidTracker` and
translates probes from the Domain-A training cache. “Best KID” is therefore an
in-sample score. With roughly 100 reference clips, one noisy KID estimate is
also too unstable to be the sole checkpoint decision.

#### The current fidelity reference is in-sample

The general channel-fidelity harness also compares against the same local
corpus used to fit the calibrated channel. The stored WavLM real-versus-synth
classifier is nearly saturated. It is useful as a warning that domains remain
separable, but not as a ship gate.

The stored fidelity snapshots reference a 100-clip real directory, whereas
corpus fitting retained 99 clips after rejecting one silence-only clip at
ingestion. The reference and fitting sets are therefore near-identical but not
exactly identical.

#### The old ASR locked test is spent

The published MatrixV1 results on `test[500:2500]` have already informed this
design. It must now be treated as a reported historical test rather than an
untouched final set. The cached public corpus has 427 remaining rows at
`test[2500:2927]`; its role is `heldout_tail_check`, an underpowered directional
check rather than a new test. A prospectively collected, source/session-disjoint
real ATC test set is required for a generalization claim.

### 2.4 Operational gaps

- `state_latest.pt` saves model and optimizer states, but the trainer has no
  `--resume` path and does not save EMA, early-stopping state, or RNG states.
- Domain-A cache reuse checks filenames only. It does not hash the clean audio,
  presets, noise bank, rendering arguments, or code.
- `residual.enabled: true` with a missing checkpoint warns and silently returns
  DSP-only audio. A purported FastCUT experiment could therefore complete
  without applying FastCUT.
- The typed residual configuration currently defaults `enabled` to true even
  though `mode2_default.yaml` explicitly sets it false. Make disabled the only
  default and require an explicit checkpoint-bearing experiment profile.
- Per-row `gen.channel.steps` metadata records that the residual fired; the
  `lineage` object does not. Neither records the checkpoint content hash, step,
  architecture, selection score, or run identity.
- GAN Domain A uses raw cached Kokoro speech, whereas normal generation can
  apply pitch, tempo, and EQ before the channel. This is a train/inference
  mismatch; pitch augmentation already changes stored KID materially.
- GAN Domain A contains one-hop speech; production can double-hop pilot speech.
- GAN Domain B contains squelch/dropout/codec events while Domain A deliberately
  omits them, yet inference applies those post-effects after the residual. This
  can make the discriminator ask the residual to learn effects that production
  later applies a second time. The learned stage's scope must be explicit and
  its A/B construction must match its insertion point.
- The residual can be applied to noise-only rows even though its training crops
  are speech-based.
- Training uses fixed crops with InstanceNorm while inference processes full
  utterances. Compare fixed-window overlap-add inference or a length-stable
  normalization so the learned transform does not depend on utterance length.
- Repeated synthetic manifests are concatenated in ASR training. Their actual
  exposure is determined by file sizes and gate yield, not requested pipeline
  probabilities.
- Separate dataset builds do not guarantee the same clean waveform across
  channel arms. MPS Kokoro output is not bit-stable enough to use separate TTS
  renders as a controlled paired comparison.

### 2.5 What the latest ASR results tell us

MatrixV1 used Mode 1 matched procedural audio; it did not test calibrated DSP
or the learned residual. The current snapshot in
[`docs/results.md`](../results.md) reports:

| Historical arm | Real spent-test normalized WER |
|---|---:|
| MatrixV1-A0 zero-shot | 133.12% |
| MatrixV1-A1 real SFT | 22.40% |
| MatrixV1-A2 gated synthetic only | 57.28% |
| MatrixV1-A2u ungated synthetic only | 62.68% |
| MatrixV1-A3 75/25 real/synthetic SFT | 22.82% |
| MatrixV1-A4 75/25 mix plus GRPO | 20.35% |

The MatrixV1-A4 result improves on MatrixV1-A1 by 2.056 absolute WER points
with a stored paired-bootstrap 95% interval of 0.322 to 4.233 points. The gate
also clearly helped synthetic-only training. At the same time, synthetic-only
remains far behind real-only, 75/25 SFT did not improve real-only by itself,
and the latest bandit-selected data lost to uniform sampling.

These results support four decisions:

1. Preserve a real-data anchor for the principal experiments.
2. Run fixed, preregistered generator comparisons before reviving adaptive
   generator selection.
3. Evaluate SFT generator effects before adding GRPO, so the ASR optimizer does
   not confound the channel comparison.
4. Treat FastCUT as an additional acoustic-diversity source, not a cure for the
   full synthetic-only gap, which also includes TTS prosody/accent/content.

## 3. What current research says

This section distinguishes direct evidence from extrapolation. “Direct” means
radio/ATC or unpaired noisy-speech translation. “Adjacent” means a related
speech or low-data GAN result. “Proposal” means a hypothesis to test here.

### 3.1 Direct evidence

#### Synthetic ATC in 2026

Bagat et al.'s [Synthetic Audio Generation Framework for Air Traffic Control
Speech Recognition](https://arxiv.org/abs/2606.21340) is the closest complete
analogue to this project. With four transcribed hours of ATCO2, it evaluates
TTS, voice conversion, accent conversion, ATC acoustic simulation, and
real/synthetic mixtures with Whisper-small.

The important results for our design are:

- TTS-only synthetic training improved from 53.88% WER without acoustic
  simulation to 33.77% with it.
- Voice-converted synthetic speech with simulation reached 24.18%.
- Real-only reached 22.69%; the best real-plus-synthetic accent-conversion arm
  reached 21.64%, while real plus simulated TTS reached 21.69%.
- A 50/50 real/synthetic batch composition was used.
- Combining all synthetic pipelines did not improve the best result and reached
  22.91%.
- A Whisper filter rejected roughly 35% of accent-converted candidates above a
  50% WER threshold.

This is evidence for an acoustic simulator, aggressive content verification,
and controlled mixtures. It is evidence against assuming that more generator
families automatically help.

#### Simu-GAN and radio-noise simulation

[Noise-Robust Speech Recognition with 10 Minutes Unparalleled In-Domain
Data](https://arxiv.org/abs/2203.15321) introduces Simu-GAN, a one-way
CUT-derived simulation approach for unpaired clean and RATS radio speech. It
reports a downstream WER change from 68.9% to 60.7% in its target setup.

The later [Study of GANs for Noisy Speech Simulation from Clean
Speech](https://arxiv.org/abs/2305.12460) compares translation systems across
stationary, nonstationary, codec, UHF, and VHF conditions. With about three
minutes per training domain, its Simu-GAN variant gives the strongest reported
UHF/VHF spectral-distribution results, reducing log-spectral distance and
multi-scale spectral loss substantially relative to its simulation baseline.
The study explicitly identifies source-phase reuse as a remaining limitation.

This is the strongest direct reason to run our experiment. It also suggests
that the relevant baseline is speech-specific CUT/Simu-GAN, not generic image
FastCUT copied unchanged.

Simu-GAN also trains ASR on source and simulated paths with a decoder-output
consistency loss. Its reported radio result improves from 68.9% WER for the
source path to 60.7% for the dual-path simulated/source system with consistency
and speed perturbation; a conventional mixup comparison using far more real
noise reached 68.0%. This makes paired-view ASR consistency a particularly
well-matched second-wave experiment here.

#### UNA-GAN

[Unsupervised Noise Adaptation using Data
Simulation](https://arxiv.org/abs/2302.11981), ICASSP 2023, retains source
PatchNCE and target identity PatchNCE for clean-to-noisy speech simulation. It
reports useful adaptation with as little as 1.7 minutes of target-domain audio
and approaches its labeled-target speech-enhancement upper bound on VoiceBank.

The 1.7-minute figure is unverified from the abstract and must be checked
against the full text before it is quoted further.

Its main downstream task is enhancement rather than recognition, so it does
not settle our ASR question. It does make target identity NCE the
speech-specific default that source-only FastCUT must beat.

#### DENT-DDSP

[DENT-DDSP](https://arxiv.org/abs/2208.00987) models VHF/UHF degradations with
an explicit and controllable differentiable signal-processing structure. It
reports learning useful channel behavior from seconds of parallel data and
downstream recognition close to a real-data benchmark in its setup.

This does not invalidate a GAN. It makes calibrated DSP a serious baseline and
supports the residual framing: let interpretable DSP cover known receiver
physics and ask the learned model only to cover its residual error.

#### Channel-conditioned translation

[CADA-GAN](https://arxiv.org/abs/2409.12386) adds a channel encoder,
feature-wise conditioning, and channel reconstruction to speech-domain
translation. Its results and ablations support explicit channel embeddings
when multiple target channels must be represented. Its strongest setup has
parallel multi-microphone information that our current local corpus lacks.

The near-term implication is to add station/preset conditioning only after the
unconditioned residual baseline reveals averaging or station collapse. The
long-term implication is to collect repeated content or shared calibration
probes across receivers so channel conditioning has identifiable supervision.

#### URSA-GAN

[Universal Robust Speech Adaptation for Cross-Domain Speech Recognition and
Enhancement](https://arxiv.org/abs/2602.04307), submitted in February 2026 and
accepted to IEEE TASLP, is the newest direct method found in this review. It
combines pretrained channel and noise encoders, FiLM conditioning throughout a
50.7M-parameter generator,
source and target PatchNCE, channel/noise consistency losses, and moderate
stochastic perturbation of reference embeddings. In its joint channel-plus-
noise recognition condition, CER changes from 32.43 for the unadapted system to
27.19, compared with 29.50 for UNA-GAN and 27.94 for CADA-GAN.

This is promising evidence for factorized channel/noise codes and controlled
style perturbation. It is not an implementation prescription for this repo:
its complete pretrained stack is much larger than our residual model, and it
relies on external channel/noise encoders and more structured supervision than
the current 99 clips provide. The practical transfer is to condition a small
residual on our known preset/station/noise metadata before considering learned
reference encoders.

### 3.2 Foundational method evidence

- [Contrastive Learning for Unpaired Image-to-Image Translation
  (CUT)](https://arxiv.org/abs/2007.15651) replaces a reverse generator and
  cycle loss with patchwise contrastive content preservation. The
  [official implementation](https://github.com/taesungp/contrastive-unpaired-translation)
  defines FastCUT as the faster, more memory-efficient setting with stronger
  source PatchNCE and no target identity NCE.
- [CycleGAN, a Master of Steganography](https://arxiv.org/abs/1712.02950)
  demonstrates that cycle consistency can hide information needed to reconstruct
  a source. A lossy radio mapping has exactly the asymmetry that makes a reverse
  cycle questionable.
- [Differentiable Augmentation](https://arxiv.org/abs/2006.10738) and
  [Adaptive Discriminator Augmentation](https://arxiv.org/abs/2006.06676)
  address discriminator overfitting in limited-data GAN training. Our current
  DiffAugment is reasonable, while ADA is a candidate if discriminator
  memorization remains visible.
- [Demystifying MMD GANs / KID](https://arxiv.org/abs/1801.01401) motivates an
  unbiased feature-distribution estimator, but unbiased does not mean low
  variance with fewer than 100 clips.
- [Projected GANs](https://arxiv.org/abs/2111.01007) show the value of frozen
  pretrained features for data-efficient discrimination. A WavLM-feature
  discriminator is a later candidate, not part of the minimum baseline.

### 3.3 Alternatives considered

**Phase-aware and waveform models.** [PHASEN](https://ojs.aaai.org/index.php/AAAI/article/view/6489)
and [DCCRN](https://www.isca-archive.org/interspeech_2020/hu20g_interspeech.html)
show that complex/phase modeling can improve speech enhancement. The
[complex-valued CycleGAN enhancement study](https://arxiv.org/abs/2109.12591)
also reports incremental gains over magnitude-only/source-phase processing.
These are enhancement results, not evidence that phase freedom improves radio
augmentation for ASR. Phase generation increases timing/content risk, so the
bounded magnitude residual remains the correct first experiment. A bounded
complex-mask ablation is justified only if phase-sensitive diagnostics remain
bad after M0 succeeds.

**True multi-resolution discriminators.** [UnivNet](https://arxiv.org/abs/2106.07889)
and [GAN Vocoder: Multi-Resolution Discriminator Is All You
Need](https://arxiv.org/abs/2103.05236) use separate STFT analyses rather than
average-pooling one spectrogram. If the current discriminator misses artifacts,
replace it with actual multi-STFT or a tightly bounded waveform head before
making the generator larger.

**Diffusion and Schrödinger bridges.** [UNIT-DDPM](https://arxiv.org/abs/2104.05358),
[DDIB](https://arxiv.org/abs/2203.08382),
[CycleDiffusion](https://arxiv.org/abs/2210.05559), and
[unpaired neural Schrödinger bridges](https://openreview.net/forum?id=uQBW7ELXfO)
are newer unpaired-translation families,
but the closest evidence is predominantly image-domain, their inference and
training loops are heavier, and there is no comparably direct tiny-data radio
ASR result. They are not the best next experiment. Revisit them if we obtain a
substantially larger channel corpus or the bounded speech-CUT family fails for
a measured multimodality reason.

**Modern flow-matching TTS.** The 2026 ATC study supports flow-matching TTS and
accent/voice conversion as future clean-speech diversity branches. They solve a
different part of the pipeline from a radio-channel residual. For this
milestone, retaining cached Kokoro keeps the channel experiment controlled;
upgrading TTS simultaneously would destroy that attribution.

### 3.4 Synthetic data and mixture evidence

The following results do not prove FastCUT will help ATC, but they constrain how
we should test it:

- [How to Leverage Synthetic Speech for LLM-Based
  ASR?](https://arxiv.org/abs/2606.29031) finds that realistic acoustic
  irregularities matter more than naturalness alone and that fixed-budget
  real/synthetic mixtures are non-monotonic: replacing too much real data can
  hurt, while additive synthetic data can help.
- Ogun et al.'s [exhaustive TTS/VC augmentation
  study](https://arxiv.org/abs/2503.08954) finds that environmental noise and
  reverberation are important, diversity gains saturate, and lexical/acoustic
  axes need to be varied deliberately rather than collapsed into “more data.”
- [Selecting TTS data for ASR](https://arxiv.org/abs/2306.00998) finds that
  intermediate real/synthetic similarity can be more useful than either the
  closest or most distant examples. This argues for a residual-strength
  continuum and distance bins rather than keeping only the most “real” output.
- [COWERAGE](https://arxiv.org/abs/2203.09829) supports stratified coverage of
  easy and difficult samples rather than selecting only the hardest examples.
- [SCADA](https://www.isca-archive.org/interspeech_2020/wang20aa_interspeech.html)
  supports consistency training across augmented views.
- [Patched multi-condition training](https://www.isca-archive.org/interspeech_2022/pesoparada22_interspeech.html)
  supports a later same-utterance channel-patch mixing experiment.
- [R2S representation alignment](https://www.isca-archive.org/interspeech_2025/tran25_interspeech.html)
  suggests that suppressing real/synthetic domain fingerprints can improve ASR.
- The [2026 Fréchet Speech Distance
  study](https://arxiv.org/abs/2601.21386) supports WavLM Base+ as a relatively
  stable embedding choice, while also concluding that distribution metrics are
  complementary diagnostics rather than replacements for listening or
  downstream evaluation.
- [ATCO2](https://arxiv.org/abs/2211.04054) documents the low-SNR, accented,
  fast-speech target conditions. ATC-specific work on
  [contextual callsign recognition](https://www.isca-archive.org/interspeech_2021/zuluagagomez21_interspeech.html)
  and [callsign boosting](https://www.isca-archive.org/interspeech_2021/kocour21_interspeech.html)
  shows why callsign and critical-entity accuracy must accompany aggregate WER.

### 3.5 Research synthesis

The evidence motivates testing this ordering; it does not establish it:

1. Calibrated DSP is the learned-statistics non-neural anchor to test;
   procedural matched remains the proven downstream baseline.
2. FastCUT and Simu-GAN motivate a compact bounded residual candidate.
3. DSP and DSP-plus-residual should coexist in the training distribution.
4. Content validity is a hard constraint; feature realism is a soft objective.
5. Generator selection occurs on held-out channel data and development ASR;
   confirmatory claims require prospectively collected, session-disjoint real
   ATC data.
6. Station conditioning, stochastic residuals, feature discriminators,
   consistency learning, and adaptive selection are follow-up ablations, not
   ingredients in the first result.

The direct literature establishes the **feasibility** of tiny-data radio
translation, not the marginal downstream value of a residual on top of an
already-calibrated simulator. Simu-GAN's headline ASR gain is bundled with a
dual-path decoder-consistency recognizer. The 2023 UHF/VHF study reports
spectral-distribution fidelity, not an isolated augmentation-ASR effect.
UNA-GAN's downstream task is enhancement. DENT-DDSP used small but parallel
data. CADA-GAN and URSA-GAN rely on pretrained channel/noise encoders and
structured supervision unlike our 99 clips. These results motivate the test;
none supplies its answer.

## 4. Proposed architecture

### 4.1 Factor the pipeline into immutable content and channel views

The current dataset builder renders TTS and a channel together. Replace that
for experiments with two explicit stages:

```text
base manifest
  one row = semantic scenario + transcript/entities + exact cached clean audio
        |
        +-- view: clean
        +-- view: procedural_matched
        +-- view: procedural_wide
        +-- view: calibrated_dsp
        +-- view: calibrated_dsp_fastcut(alpha)
        +-- view: real_noise_replay (optional baseline)
```

Every channel view of a `base_id` must consume the exact same clean WAV. This
eliminates TTS nondeterminism as a confound and enables paired content,
acoustic, listening, and ASR-consistency tests.

For calibrated comparisons, also cache a `channel_draw_id`: G3 DSP and every
G4/G5 residual-strength view must consume the exact same preset, noise bed,
SNR draw, hop count, and pre-residual waveform. The only difference in a
DSP-versus-FastCUT pair is the learned residual. Procedural-versus-calibrated
comparisons remain paired on clean content but are not falsely described as
the same channel draw.

A base-manifest row should contain at least:

```json
{
  "base_id": "...",
  "semantic_group_id": "...",
  "split": "train",
  "audio_clean": "clean/....wav",
  "text": "...",
  "text_display": "...",
  "entities": [],
  "role": "pilot",
  "kind": "...",
  "voice": "...",
  "speed": 1.2,
  "tts_seed": 123,
  "frontend": {"pitch": 0.0, "tempo": 1.0, "eq": 0.0},
  "content_sha256": "..."
}
```

A channel-view row should add:

```json
{
  "base_id": "...",
  "view_id": "...",
  "pipeline": "calibrated_fastcut",
  "audio": "wavs/....wav",
  "channel": {"preset": "...", "station": "...", "hops": 1},
  "residual": {
    "applied": true,
    "alpha": 0.5,
    "checkpoint_sha256": "...",
    "training_step": 10000
  }
}
```

These are direct schema changes for the experiment.

### 4.2 Model M0: corrected bounded FastCUT residual

Keep the existing small generator and DSP-anchored input for the first model:

\[
s = \log(1 + |\operatorname{STFT}(\operatorname{DSP}(x))|)
\]

\[
\hat{s} = \max(0, s + \alpha r_{\max}\tanh(G(s)))
\]

\[
\hat{x} = \operatorname{iSTFT}(\exp(\hat{s})-1, \angle\operatorname{STFT}(\operatorname{DSP}(x)))
\]

where `alpha` is exposed at inference and sampled only within a frozen,
validated interval. Implement `alpha=0` as an explicit translator bypass so it
is the calibrated-DSP endpoint, rather than relying on a potentially lossy
STFT round trip; `alpha=1` is the trained bound. This creates a continuous
experiment rather than a binary GAN toggle.

The first corrected loss is one controlled content-preservation ablation:

- **M0/source-NCE:** source PatchNCE with `lambda_NCE=10`, without a target
  identity term.
- **M0/source+identity-NCE:** the same source PatchNCE plus target-identity
  PatchNCE. Identity inputs are real Domain-B crops passed through G; the
  identity term uses the same NCE layers as the source term and detached keys.
  Freeze one identity weight before the comparison, initialized at the
  UNA-GAN/CUT convention `lambda_idt = lambda_NCE`.

Neither variant reproduces FastCUT, CUT, Simu-GAN, or UNA-GAN. They differ only
in whether the bounded residual has the fully specified target-identity
PatchNCE term above.

Keep **M0-SpeechPatch** as a named follow-up, but run it only if the winning
variant shows a diagnosed patch-sampling failure: silence-dominated anchors or
repeated-structure false negatives in the NCE diagnostics. It stratifies
PatchNCE anchors across speech-active harmonic regions,
high-frequency/consonant regions, transitions, and the noise floor. This
simple ablation captures the relevant idea from
[QS-Attn](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_QS-Attn_Query-Selected_Attention_for_Contrastive_Learning_in_I2I_Translation_CVPR_2022_paper.html)
without importing an image-specific attention stack.

Both retain:

- adversarial target-domain loss;
- bounded residual output;
- DiffAugment on real and fake inputs;
- lazy R1, generator EMA, and early stopping;
- one-way translation only; and
- the same architecture, data, and update budget in the comparison.

In all variants, detach PatchNCE key features and replace circular/log-domain
augmentations with zero-padded time shifts/shared crops and true waveform gain.
Freeze whether post-effects belong before or after the learned stage, then
construct Domain A and B consistently with that decision.

Do not change the discriminator, residual bound, loss family, crop duration,
and conditioning all at once. The purpose of M0 is to obtain an interpretable
baseline and determine whether target identity NCE improves content retention
or simply discourages useful channel translation.

### 4.3 Model M1: station/preset-conditioned residual

Promote this only if M0 shows one of the following on validation:

- station distributions converge toward a common average;
- one station improves while another worsens;
- a generator-ID or station classifier finds strong systematic artifacts; or
- the same input requires visibly different receiver transformations.

M1 adds a small station/channel encoder and FiLM modulation in residual blocks,
inspired by CADA-GAN and URSA-GAN. Conditioning can begin with known metadata and fitted
preset descriptors—station, passband, spectral slope, SNR, AGC/compression,
noise family—rather than attempting to infer a rich channel embedding from 84
clips.

The conditioning target must be sampled explicitly during generation and
recorded in the manifest. A model that sees a station label but produces no
measurable per-station difference has not earned its complexity.

### 4.4 Later model candidates

These remain hypotheses until M0 downstream results exist:

- **Stochastic residual:** add a low-dimensional latent or noise map so the
  same DSP input can produce several plausible residuals. Require diversity
  without content loss and compare against the much cheaper `alpha` continuum.
  URSA-GAN suggests that modest perturbation of an explicit channel/noise code
  is safer than unconstrained generator noise.
- **Frozen-feature discriminator:** discriminate WavLM feature maps in addition
  to local STFT patches. Useful if D memorizes speaker/content rather than
  channel texture; risky because pretrained speech features may also reward
  linguistic changes.
- **Long-context branch:** add a lower-rate discriminator or temporal envelope
  loss for squelch, AGC, carrier drift, and long bursts. Keep explicit
  post-effects outside the GAN until evidence shows the residual should model
  them.
- **Complex/phase model:** predict complex masks or use a waveform generator.
  This addresses the documented source-phase limitation but is a larger search
  space and a greater content risk.
- **Residual factorization:** separate stationary coloration from transient
  interference and nonlinearity. This is attractive for control and diagnosis,
  but should follow a residual attribution study.

### 4.5 What not to do first

- Do not revive the old full CycleGAN as the primary model. Its reverse mapping
  spends compute on an unused direction and creates incentives to preserve
  information that a radio channel should destroy.
- Do not train FastCUT directly from pristine TTS to real audio as the only
  path. That asks it to relearn known radio physics and increases the content
  and mode-collapse burden.
- Do not choose a generator by GAN loss, one KID number, or a small listening
  sample.
- Do not mix all available generator pipelines and call a larger corpus a fair
  comparison.
- Do not optimize the GAN to make the current Whisper student fail. Hardness is
  a data-selection property inside a teacher-verified validity region, not a
  generator content objective.

## 5. Data and split design

### 5.1 Channel data has its own split system

The target design has explicit, immutable channel partitions before fitting
presets or harvesting noise:

| Partition | May fit presets/noise? | May train GAN? | May select checkpoint? | May report final fidelity? |
|---|---:|---:|---:|---:|
| `channel_train` | yes | yes | no | no |
| `channel_val` | no | no | yes | no |
| `channel_test` | no | no | no | yes, once |

The unit of independence is the capture session/receiver condition, not the
clipped transmission. The current corpus contains only a handful of these
conditions, so clip counts overstate the effective sample size.

The existing corpus supports **development-only** blocked folds:
`channel_train` and `channel_val`, grouped by station plus capture time-block.
Rebuild those folds from source clips before fitting any artifact. A
`channel_test` partition does not exist yet and cannot be conjured from this
corpus; create it only by prospectively collecting at least one new capture
session/receiver. Until then, make no receiver/session-generalization claim.
Because station counts are unbalanced, report both macro-station and pooled
development metrics.

Leave-one-station-out is an exploratory OOD stress test only. It must not drive
checkpoint selection, and it does not estimate station generalization: station
is confounded with service type, geography, speakers, and session. Removing a
station from Domain B also asks the translator to hit a target it never saw.

### 5.2 Everything derived from a split inherits that split

The following artifacts must be built from `channel_train` only:

- fitted presets;
- noise beds and noise statistics;
- DSP parameter envelopes;
- GAN Domain B;
- GAN Domain A rendered from those presets/noise beds; and
- any learned normalization or feature statistics.

Validation real clips remain raw references. Validation synthetic probes are
rendered from an independent clean-TTS validation set, but may use only
train-derived DSP artifacts. This measures whether the learned transform
generalizes to new clean content and real receiver recordings.

### 5.3 Semantic ASR splits stay independent

Split semantic scenarios before rendering channel views. All views of one
`semantic_group_id` must remain in the same ASR partition. No validation/test
transcript template, callsign instance, or paired clean waveform should enter
GAN-ASR training through another channel arm.

Use the current public split registry for development only after reclassifying
the previously reported locked test as spent. Reserve the 427-row
`heldout_tail_check` for its underpowered directional role. A future
generalization claim requires a prospectively collected real target-airport
test with sources and sessions wholly isolated from channel fitting, gate
tuning, generator selection, ASR model selection, and bandit rewards.

### 5.4 Data collection priorities

The single highest-value acquisition for **channel science** is repeated known
probes or near-parallel clean/receiver recordings across sessions. These make
channel attribution and conditioning identifiable through CADA-style
supervision. New independent sessions are the highest-value acquisition for
**validation**.

If we add real channel data, prioritize diversity in this order:

1. new receiver/capture sessions from the same stations;
2. new stations and service types—ground, tower, approach, center;
3. receiver/antenna/frequency changes;
4. SNR and interference regimes;
5. long idle/noise/squelch spans; and
6. repeated calibration probes or parallel same-content recordings where
   legally and operationally possible.

More adjacent clips from the same feed help optimization but do less for
validation. Repeated known probes across channels would make CADA-style channel
conditioning and channel reconstruction much more identifiable.

## 6. Trainer and checkpoint-selection design

### 6.1 Required trainer hardening

**Required for the go/no-go**

- explicit train/validation splits with independent clean Domain-A manifests;
- candidate evaluation outside the training loaders;
- detached PatchNCE keys;
- physical augmentations: zero-padded shared shifts/crops and pre-STFT gain;
- basic resume of G, D, NCE heads, EMA, optimizers, and step;
- a resolved-configuration dump;
- SHA-256 of corpus, manifests, presets, noise, and cache inputs, with
  fail-closed mismatch handling;
- periodic immutable candidate checkpoints;
- checkpoint identity in inference metadata;
- non-finite aborts;
- fixed paired auditions and residual diagnostics;
- inference-time `alpha` with a true `alpha=0` translator bypass; and
- crop-consistent overlap-add inference.

**Deferred until a promoted run**

- bitwise-exact resume including RNG, patience, and lexicographic-selection
  state;
- a generalized cache-invalidation framework; rebuilding from scratch is
  acceptable at go/no-go scale;
- the full FSD/speech-MMD suite;
- nearest-neighbor memorization audits;
- classifier diagnostics as anything more than reporting; and
- separate-process WavLM evaluation. Serial evaluation of saved candidates
  about every 2,000 steps with cached real embeddings is sufficient.

### 6.2 Checkpoint evaluation is a lexicographic decision

No single scalar captures both channel realism and content safety. The full
candidate report uses the following panels; items deferred in §6.1 are omitted
from the go/no-go report:

**Hard validity/content constraints**

- zero NaN/Inf, empty, corrupt, or duration-shifted outputs;
- clipping and silence within frozen bounds;
- teacher ASR WER delta from calibrated-DSP input;
- callsign and critical-entity preservation;
- waveform/time alignment and correlation diagnostics;
- magnitude bins driven to zero and residual-clamp saturation;
- noise-only hallucination rate; and
- nearest-neighbor similarity to real training clips as a memorization audit.

**Channel fidelity/coverage objectives**

- WavLM Base+ KID, FSD, and speech-MMD with repeated resamples and confidence
  intervals where applicable;
- real-real floor and train-versus-validation reference distances;
- LTAS/passband/spectral-slope error;
- SNR, modulation spectrum, dynamic range, AGC, and clipping distributions;
- station-stratified metrics and macro averages;
- real-versus-synthetic classifier accuracy as a diagnostic; and
- diversity across `alpha`, presets, stations, and random inputs.

**Human checks**

- small blind A/B/ABX comparisons for radio-channel similarity;
- separate intelligibility judgments; and
- balanced examples across stations, SNR, role, short numeric transmissions,
  and long clearances.

Promotion follows one predeclared lexicographic rule:

1. **Hard validity gates:** require finite, nonempty, duration-stable audio;
   clipping and silence inside frozen bounds; and prespecified content
   noninferiority margins for teacher-WER delta and callsign/entity
   preservation. Calibrate those margins on DSP and real-real baselines and
   freeze them before scoring any candidate.
2. **Held-out score:** among passers, minimize one predeclared score: fold-level
   WavLM Base+ KID (or MMD) on fixed validation probes, averaged across blocked
   folds.
3. **Tie-break:** if candidates are within one standard error of the best,
   select the earliest checkpoint.

Everything else in the lists above—spectral panels, classifier accuracy,
diversity, saturation, and listening—is reported as a diagnostic and cannot
add promotion degrees of freedom. With roughly 15 validation clips, KID
resampling does not manufacture power. KID is a noisy selector and never an
early-stopping oracle; the hard gates and earliest-checkpoint tie-break
therefore carry the weight.

**Teacher circularity guard.** Freeze the gate teacher before generating any
arm. Teacher scores act only as content-safety constraints, never as realism
objectives. An independent recognizer or forced-alignment audit must
cross-check a stratified sample of accepted **and** rejected outputs, and
rejection reasons must be reported per pipeline.

### 6.3 Residual attribution diagnostics

Record what the GAN actually changes:

- residual energy by frequency and time;
- percentage of cells at `+-r_max` and percentage floored to zero;
- before/after LTAS, SNR, crest factor, modulation, and active-span measures;
- changes conditional on preset and station;
- one-hop versus two-hop behavior;
- speech versus noise-only behavior;
- source-to-output teacher tokens and entity changes; and
- output variation across `alpha`.

If most gains are reproduced by a simple static EQ/noise adjustment fitted from
the residual average, prefer that interpretable DSP change. FastCUT earns a
place only for structured variation the calibrated channel does not already
capture.

## 7. Mac-first training program

The existing full path works on MPS. The M3 should be used aggressively for
correctness, pilot training, and successive halving. Since the user's GPU was
busy during the exploratory smoke measurements, those wall times must not be
used to predict an idle Mac or 5080 run.

### 7.1 Hardware policy

| Work | Default device | Rule |
|---|---|---|
| Unit/integration tests | CPU/MPS | Run locally. |
| 20–100-step smoke | MPS | Required for every material trainer change. |
| 1k–2k pilot | MPS | Default; inspect learning and validation curves. |
| 5k–10k promoted run | MPS | Valid if stable and practical; run serially with other GPU workloads. |
| Multiple finalist seeds / longer run | MPS or 5080 | 5080 accelerates but does not authorize weaker validation. |
| Production corpus generation | 5080 preferred | Throughput choice after checkpoint freeze. |

Do not run Whisper training and GAN training concurrently on the Mac. MPS
memory pressure and concurrent user workloads make timings and stability hard
to interpret. Make jobs resumable and serial by default.

### 7.2 Day-one idle-device timing protocol

Run this benchmark on day one of implementation, with the Mac GPU idle:

1. record hardware, OS, PyTorch, dtype, power mode, and device;
2. pre-render/load the same Domain-A cache;
3. run at least 100 warmup steps;
4. measure at least 500 ordinary steps and several R1 steps separately;
5. synchronize the device around timing boundaries;
6. repeat three times and report median and range;
7. record peak memory; and
8. measure WavLM evaluation separately from GAN optimization.

This benchmark informs scheduling only. It is not a promotion metric.

### 7.3 Collapsed GAN sweep

Use these stages:

| Stage | Purpose | Approximate budget | Promotion rule |
|---|---|---:|---|
| S0 | integration smoke | 20–100 updates | finite, resumable, correct artifacts and held-out isolation |
| S1 | paired NCE ablation | 1k–2k updates, two paired seeds per arm | §6.2 lexicographic rule |
| S2 | selected formulation | one 5k-update run | §6.2 lexicographic rule and ASR-screen eligibility |
| S3 | conditional confirmation | 10k updates, at least two seeds | only after a positive ASR screen; report direction and station diagnostics |
| S4 | conditional finalist | only if S3 curves and the screen justify it | prefer the 5080 for any 10k+ promoted run |

The only variable in the first sweep is the §4.2 ablation:
`M0/source-NCE` versus `M0/source+identity-NCE`. Run them as paired S1 pilots
of 1,000–2,000 optimizer updates with two paired seeds each, identical data and
configuration otherwise, fixed `r1_gamma=1`, fixed `r_max=0.20`, and the
current architecture and crop policy. Select the winner by the §6.2
lexicographic rule on held-out folds.

The winner receives one S2 5,000-update run. A third seed and any exploration
of `r_max` or R1 occur only after the downstream ASR screen in §8.3 is
positive. Inference-time `alpha` replaces retraining multiple residual bounds.
MPS remains the default for S0–S2.

Express every budget as both optimizer updates and real-audio exposure:
84 clips × crops seen. The discriminator otherwise revisits the same few
recordings thousands of times, which a step count hides.

Use one verified immutable Domain-A cache per data/config identity, shared
across arms. Save checkpoints every 250–500 steps during pilots because a tiny
target corpus may peak early. WavLM Base+ dwarfs the 4.4M-parameter generator,
so never interleave its scoring every 250–500 steps. Score saved candidates
serially about every 2,000 steps with cached real-embedding statistics; only
the WavLM scoring is batched.

## 8. Controlled generator comparison

### 8.1 Channel-view arms

Every arm consumes the same `base_id` pool:

**Premise check (runs first)**

| ID | Pipeline | Purpose |
|---|---|---|
| G1 | procedural matched | Current proven synthetic baseline. |
| G3 | calibrated DSP | Must be at least competitive with G1 downstream before residual training. |

Run G1 versus G3 as the one-seed, shared 500–1,000-content SFT premise check
defined in the executive conclusion.

**Core generator views**

| ID | Pipeline | Purpose |
|---|---|---|
| G1 | procedural matched | Current proven synthetic baseline. |
| G3 | calibrated DSP | Learned-statistics, non-GAN anchor. |
| G5 | calibrated DSP + sampled `alpha` residual | Tests moderate learned variation; its continuum includes G3 and G4. |

G4 remains defined as the fixed-full-strength, `alpha=1` diagnostic view inside
G5's family. It is not a separate core arm.

**Diagnostics and fallbacks**

| ID | Pipeline | Purpose |
|---|---|---|
| G0 | clean Kokoro | Lower-bound channel realism and content upper bound. |
| G2 | procedural wide | Wider hand-authored domain-randomization diagnostic. |
| G8 | training-split real-noise replay | Designated cheap non-GAN challenger; run at screen stage or immediately after a residual no-go. |

ASR mixtures of G1, G3, and G5 are sampling recipes in §9, not generator arms.
The legacy CycleGAN is not decision-relevant and is cut; its code remains only
as project history.

Recipe shares refer to optimizer exposure, not file counts. The sampler must
enforce the requested probabilities after gating.

### 8.2 Fairness controls

- same clean audio, transcripts, entities, voices, speeds, and semantic groups;
- same common-support unique-content budget for primary comparisons;
- same ASR initialization, optimizer updates, batch size, and augmentation;
- same gate thresholds and teachers;
- same real examples and real exposure where applicable;
- pipeline selected after content group, so multiple channel views do not
  multiply one transcript's lexical weight;
- pipeline identity recorded after all filtering;
- results sliced by content group and station/source; and
- no GRPO until SFT identifies the best channel recipe.

For the primary comparison, use paired common-support gating: generate every
compared view for the same ordered base pool and retain a base group only when
every compared view passes the gate. Report per-pipeline yield and a separate
operational all-accepted analysis. Pipeline-dependent gating changes the
surviving content, so equal counts can silently compare different content
distributions.

### 8.3 Cheap ASR screen

Before an expensive full matrix, train short Whisper-tiny SFT probes on
500–1,000 accepted unique content groups per promising pipeline. Evaluate only
on `model_select` and channel-development sets. Drop an arm if it clearly
worsens all of WER, callsign accuracy, entity F1, and critical substitutions or
if its gate yield is impractically low.

This screen is a resource allocator, not publishable evidence. A noisy
Whisper-tiny ranking should not eliminate an arm whose channel metrics and
content constraints are strong unless the downstream degradation is material.

Synthetic-only probes are the **sensitive instrument** for ranking generators.
The MatrixV1-* results showed generator/gate effects exceeding five WER points
synthetic-only training, while real-anchored mixtures compressed the same
differences to fractions of a point. Generator-ranking weight therefore rests
on the synthetic-only screen plus channel-validation diagnostics; mixtures are
reserved for the deployment question.

## 9. ASR mixture experiments

### 9.1 Separate two questions

There are two valid but different experiments:

**Fixed-budget substitution**

- total examples/updates are fixed;
- adding synthetic data displaces real examples;
- answers how to allocate a fixed training budget.

**Additive augmentation**

- every arm receives identical exposure to the full unique real set;
- compare four arms: (i) no extra updates, (ii) extra real repeats, (iii) extra
  G3 updates, and (iv) extra G3/G5-mixture updates;
- the extra-real arm is required to distinguish “synthetic helps” from “more
  optimization helps”;
- answers whether synthetic data adds value when compute can grow.

Do not combine their conclusions. Current research shows they can have
different optima. The 2026 ATC study used an additive design—real data was
kept, with 50/50 real/synthetic batches—so it informs this experiment, not the
fixed-budget substitution experiment.

### 9.2 First SFT matrix

Use the best M0 checkpoint and a frozen gate. Begin with the current 75/25
real/synthetic setting for continuity, but compare the following under exact
sampling:

| Arm | Real share | Synthetic recipe |
|---|---:|---|
| FC-A0 | 100% | none; real-only anchor |
| FC-A1 | 75% | 25% G1 procedural matched anchor |
| FC-A2 | 75% | 25% G3 calibrated DSP |
| FC-A3 | 75% | 25% G5 calibrated + sampled residual |
| FC-A4 | 75% | 12.5% G3 + 12.5% G5 |

Include synthetic-only G1, G3, and G5 arms as diagnostics at small scale, not
as likely winners. Their purpose is to measure the synthetic gap and whether
FastCUT narrows it.

Predeclare one primary and one secondary contrast:

- **Primary:** FC-A4 versus FC-A2: does adding the residual to the mixture beat
  DSP-only at equal budget?
- **Secondary, mechanistic:** FC-A3 versus FC-A2.

FC-A0 and FC-A1 are anchors and references. The real/synthetic ratio sweep is
exploratory only and runs only after the primary contrast exceeds a
predeclared minimum development effect:

- 90/10;
- 75/25; and
- 50/50.

If additive compute is available, run the four-arm experiment in §9.1. The
exact batch sampler should expose two independent controls: real fraction and
conditional per-pipeline weights.

### 9.3 Training schedule ablations

Only after identifying the best data mixture, compare equal-update schedules:

- constant interleaving;
- synthetic warmup followed by a real/DSP anchor phase; and
- residual-strength curriculum from `alpha=0` toward the validated interval,
  ending with real/DSP anchoring.

Then, and only then, add the existing ASR GRPO stage to the winning SFT recipe.
This preserves the ability to attribute improvement to data, training schedule,
or ASR post-training.

### 9.4 Paired consistency experiment

The paired-view dataset enables a second-wave ASR objective:

- compute transcript loss on one sampled view;
- compare token distributions for G3 DSP and G5 DSP-plus-residual views of the
  same clean utterance; and
- add a bounded Jensen–Shannon or token-level KL consistency term.

An optional generator-ID gradient-reversal head can test the R2S hypothesis
that ASR should discard pipeline fingerprints. Both are objective changes and
must be evaluated only after the data-only baseline.

## 10. Evaluation and statistics

### 10.1 Primary downstream outcomes

Predeclare normalized real-ATC WER as the **sole primary efficacy endpoint**.
The following are prespecified **noninferiority safety endpoints**, with
margins calibrated and frozen before any FC- arm is scored:

- exact callsign accuracy and callsign WER;
- per-entity precision/recall/F1;
- clean/general-English regression; and
- hallucination rate on noise-only/empty-transcript inputs.

Additional utility diagnostics are:

- command and numeric-value exact accuracy;
- critical-number substitution rate;
- insertion, deletion, and substitution counts.

Callsign/entity and clean-English regression metrics are safety endpoints, not
additional efficacy objectives. All slices—station/source, seen versus unseen
source, SNR bin, pilot/controller role, duration, entity presence, phraseology,
and accent/geography proxies—are exploratory.

Channel metrics select checkpoints; development ASR selects recipes; a future
independent test estimates the one frozen primary contrast.

### 10.2 Generalization checks

- `model_select` for iteration only;
- `heldout_tail_check`, the 427-row underpowered directional check on a corpus
  that already shaped this design, not a new locked test;
- clean/general English regression set;
- channel-validation references and a future channel-test session;
- noise-only hallucination set; and
- optionally a second ATC corpus or newly collected receiver source.

The current `locked_test` becomes `spent_test_fastcut`. No result from it may
be described as a new untouched test result. A generalization claim requires a
prospectively collected, source/session-disjoint real test set, frozen before
any FC- results are computed.

### 10.3 Statistical protocol

- freeze the generator checkpoint, residual-strength distribution, gate,
  mixture, ASR recipe, and model-selection rule before final evaluation;
- use at least three ASR seeds for finalists;
- use paired bootstrap confidence intervals on aligned utterances;
- resample hierarchically: sessions/sources first, then utterances;
- pair ASR seeds and sampler order across arms, and report per-seed results;
- report macro-source as well as pooled metrics;
- report effect sizes, intervals, and per-seed values rather than only p-values;
- compare every residual recipe directly with calibrated DSP and procedural
  matched, not only real-only; and
- label inference descriptive where too few independent sessions exist for
  session-level resampling.

**Power.** The MatrixV1-A4-versus-MatrixV1-A1 interval, [0.32, 4.23] around
2.056 on 2,000 rows, implies a paired standard error of about 1.0 WER point.
Scaling by `sqrt(2000 / 427)` gives an SE of about 2.2 and a 95% half-width
near 4.2 points on `heldout_tail_check`: an approximately six-point minimum
detectable effect at 80% power. That is far above the plausible sub-point
effect of a channel residual inside a 75/25 mixture.

Consequently, the go/no-go decision rests on `model_select`, channel-validation
diagnostics, and seed-consistent direction—not tail significance. Report the
tail descriptively. A sized future test of roughly 3,700 or more independent
utterances for a two-point effect, and more under session clustering, is the
only path to a confirmatory claim.

### 10.4 Promotion and stop rules

A GAN checkpoint advances to ASR screening only if it passes every hard gate
and wins the held-out score and earliest-checkpoint tie-break in the §6.2
lexicographic rule. Station slices, memorization checks, and blind review remain
reported diagnostics and do not change that selection rule.

A residual data recipe advances to a full ASR run only if its development probe
is competitive with calibrated DSP and procedural matched without entity or
hallucination regression.

Success at this program's scale means that FC-A4 wins the predeclared primary
contrast against FC-A2 on `model_select`, with seed-consistent direction and no
safety regression, plus a non-contradicting `heldout_tail_check`. This remains
development/feasibility evidence. Publication-grade or confirmatory language
waits for the future prospectively collected test set.

If it fails, calibrated DSP remains the production generator. A negative
result is still useful if it identifies whether the failure came from content
damage, data scarcity, station averaging, phase limitations, or lack of
complementary channel coverage.

## 11. Novel experiments worth keeping in reserve

### Residual-strength continuum

Sample `alpha` over a validated interval including zero. Analyze utility by
distance bin rather than selecting only maximum translation strength. This is
cheap, exposes overcorrection, and follows synthetic-data evidence favoring
intermediate domain similarity.

### Leave-one-generator-out fingerprint audit

Train an ASR encoder on all but one synthetic pipeline, then test both the
held-out synthetic view and real ATC. If gains exist only on seen generator
families, the model is learning artifacts rather than channel invariance.

### Generator classifier

Train a lightweight classifier over frozen ASR or WavLM features to identify
clean, procedural, calibrated, and FastCUT sources. High accuracy is not alone
a failure, but high generator separability plus weak real-ASR performance is a
strong warning. Measure rather than optimize this score blindly.

### Channel PatchMix

Splice time patches from paired G3 and G5 views of the same utterance, retaining
alignment and transcript. This tests robustness inspired by patched
multi-condition training. Label it as augmentation, not realistic simulation.

### Station-conditioned FiLM

Condition residual blocks on station/preset descriptors and require
reconstruction or classification of the intended condition. Compare macro-
station validation and cross-station transfer against unconditioned M0.

### Noise-residual factorization

Estimate the average/static residual separately from transient variation.
Reimplement any stable average as DSP, leaving the GAN to model localized
events. This makes the hybrid progressively more interpretable.

### Target-noise replay

Remix clean or DSP speech with receiver noise separated only from
`channel_train`. Compare it directly with GAN output. If replay wins, use it;
the goal is better ASR data, not defending a model family.

### Multiple good GAN seeds as a mixture

If independently trained seeds produce measurably complementary residual
families without content loss, sample across two or three fixed checkpoints.
This is cheaper than adding a stochastic generator, but must beat a single
checkpoint plus `alpha` sampling under equal exposure.

### Lexical-acoustic factorial

Cross common versus rare/entity-rich semantic groups with G1, G3, and G5. This
tests whether channel augmentation disproportionately helps or hurts dense
numeric and callsign speech and keeps lexical-coverage gains separate from
acoustic gains.

## 12. Implementation roadmap

> **Minimal critical path: go/no-go, target 2–3 weeks**
>
> 1. Day one: run the idle-device timing benchmark (§7.2).
> 2. Run the premise check: a one-seed G1-versus-G3 SFT probe on a shared
>    500–1,000-content pool. Abandon or fix G3 first if it loses materially.
> 3. Repair leakage only (Phase 0 core): create grouped `channel_train` and
>    `channel_val`; refit presets and reharvest noise from `channel_train`
>    only; delete contaminated artifacts; run the automated leakage audit.
> 4. Build a minimal paired cache with `base_id`, clean-WAV hash,
>    `channel_draw_id`, and transform hashes. Render G3 once and derive
>    residual views from its pre-residual waveform. Rebuild rather than
>    invalidate at this scale.
> 5. Implement only the trainer must-have tier (§6.1), then complete a
>    100-example end-to-end vertical slice: train → translate → gate → ASR
>    smoke. Build no further infrastructure first.
> 6. Run the paired S1 ablation (§7.3), one S2 5k run, and the §6.2
>    lexicographic selection.
> 7. Run the cheap ASR screen (§8.3): synthetic-only probes plus FC-A2 versus
>    FC-A4 mixture probes on `model_select`. Record a decision: **go**, funding
>    Phases 4–6 and the full matrix; or **no-go**, adopting the G8 replay
>    comparison and shifting research budget to the VC/accent branch.
>
> VC/accent feasibility work starts after this screen regardless of outcome.
> It does not wait for Phase 6: it is the higher-ceiling lever for the
> synthetic-only gap (§2.5).

The phases below describe the **conditional full program**, entered only on a
“go.” The minimal portions of Phases 0–3 above are completed to reach that
decision. In full, the program is roughly 38–69 development days, or four to
six calendar months for one person.

### Phase 0 — make the experiment valid (1–2 weeks)

**Goal:** no training/evaluation leakage and no silent experimental fallbacks.

Change:

- [`atcgen/channel/learned/channel_fit.py`](../../atcgen/channel/learned/channel_fit.py):
  filter an explicit input split/group.
- [`atcgen/dataset/noise_harvest.py`](../../atcgen/dataset/noise_harvest.py):
  propagate and filter split/session metadata.
- [`atcgen/channel/learned/preset.py`](../../atcgen/channel/learned/preset.py):
  support explicit filtered loading and verify expected partitions.
- [`atcgen/channel/learned/backend.py`](../../atcgen/channel/learned/backend.py):
  fail closed when an enabled residual checkpoint is missing; define one-hop,
  two-hop, and noise-only behavior; record checkpoint identity.
- [`atcgen/config.py`](../../atcgen/config.py): make the residual default
  disabled and replace ambiguous paths/options directly.
- local corpus tooling: derive grouped channel train/validation manifests
  before fitting artifacts; a test manifest requires prospective data.

Tests:

- validation clip IDs never appear in preset/noise/GAN training lineage;
- an enabled missing checkpoint raises an error;
- old contaminated artifacts are rejected by partition/hash checks; and
- one-hop, double-hop, and noise-only rules are explicit.

**Exit:** an automated leakage audit reports zero forbidden source IDs.

### Phase 1 — immutable paired channel views (1.5–2.5 weeks)

**Goal:** isolate channel effects from TTS and semantic differences.

Implement a clean-base builder and a channel fan-out builder. Store exact
frontend metadata and hashes. Generate G0–G5 and G8 paired development views
and a small blind-audition pack.

Tests:

- same `base_id` means byte-identical clean input to every channel branch;
- all views preserve transcript/entities/semantic group;
- changing any source/config/checkpoint invalidates the stage cache; and
- view lineage names the exact source and transform hashes.

**Exit:** paired DSP and FastCUT-ready manifests can be reproduced without
rerunning Kokoro.

### Phase 2 — harden and align the trainer (3–5 weeks)

**Goal:** trustworthy local pilot runs.

Change
[`atcgen/channel/learned/residual_train.py`](../../atcgen/channel/learned/residual_train.py)
and [`residual.py`](../../atcgen/channel/learned/residual.py) to add:

**Go/no-go tier:**

- independent train/validation A and B;
- basic resume for G, D, NCE heads, EMA, optimizers, and step;
- resolved configuration and fail-closed input hashes;
- immutable candidates and checkpoint identity in inference metadata;
- `M0/source-NCE` and `M0/source+identity-NCE` content-loss modes;
- detached PatchNCE keys and physical augmentations;
- inference-time `alpha` with a true zero bypass;
- crop-consistent overlap-add inference;
- non-finite aborts and fixed residual/content diagnostics; and
- held-out lexicographic evaluation.

**Deferred tier:**

- bitwise-exact resume with RNG, patience, and lexicographic-selection state;
- generalized cache invalidation rather than rebuild-from-scratch;
- FSD/speech-MMD and nearest-neighbor memorization suites;
- classifier tooling beyond reporting;
- separate-process WavLM evaluation; and
- M0-SpeechPatch, only after the diagnosed NCE failure defined in §4.2.

Tests:

- basic resumed toy runs restore all required go/no-go state;
- held-out evaluator cannot read training IDs;
- EMA and optimizer state round-trip;
- MPS full-size step and KID path pass;
- CUDA smoke passes when the 5080 is available; and
- `alpha=0` is calibrated DSP while `alpha=1` is the full bounded model.

**Exit:** S0 and S1 complete on the M3 with green guards and auditable
artifacts.

### Phase 3 — GAN pilot and selection (1–3 weeks elapsed)

**Goal:** select zero or one M0 formulation honestly.

The go/no-go portion runs the paired S1 NCE ablation and one S2 5k candidate.
After a positive ASR screen only, run S3 confirmation and any justified R1 or
residual-bound exploration. Save candidates every 250–500 steps, inspect fixed
paired audio, and score WavLM serially as specified in §7.3.

**Exit:** either a frozen promotable checkpoint with SHA-256 and validation
report, or a documented negative result. Do not enable residual generation
globally merely because training completed.

### Phase 4 — exact multi-pipeline ASR sampler (1–2 weeks)

**Goal:** make the data mixture the controlled variable.

Change [`training/recipe.py`](../../training/recipe.py) so a batch sampler:

1. selects real versus synthetic by an exact configured probability;
2. selects a semantic/base group;
3. selects a pipeline view by conditional weights; and
4. logs realized examples, hours, content groups, and pipeline shares.

Reuse the existing seams rather than introducing new frameworks:
`config_hash`, resolved configuration, and lineage in
[`atcgen/config.py`](../../atcgen/config.py) and
[`atcgen/dataset/build.py`](../../atcgen/dataset/build.py); audio hashing in
[`atcgen/dataset/local_corpus.py`](../../atcgen/dataset/local_corpus.py);
`training.recipe.mixed_pool` extended with pipeline-aware rows and
realized-exposure logging; lazy multi-manifest pools in
[`training/grpo.py`](../../training/grpo.py); and
[`scripts/run_matrix.py`](../../scripts/run_matrix.py) retrofitted with content
fingerprints rather than a second orchestrator. Every stage uses immutable
fingerprints rather than file-existence or row-count reuse.

**Exit:** a dry-run audit proves requested and realized mixture shares match,
and all arms see equal unique content and updates.

### Phase 5 — ASR screen and full matrix (2–4 weeks elapsed)

**Goal:** answer whether the residual adds downstream value.

Run the full FC- SFT matrix and three seeds for finalists. Run the exploratory
ratio sweep only if the primary contrast clears its frozen minimum development
effect. Freeze the winning SFT data recipe, then evaluate the existing GRPO
stage only on that frozen recipe and its strongest DSP control.

**Exit:** paired development evidence selects one frozen recipe; the
`heldout_tail_check` remains descriptive.

### Phase 6 — final evaluation and operationalization (1 week plus data)

**Goal:** one defensible result and a reproducible production generator.

Collect and freeze a prospectively sourced, session-disjoint real test, then
evaluate the one frozen primary contrast once and publish the full metric panel
and negative slices. Only then use confirmatory language. Production generation
should fail if the checkpoint hash or expected input artifacts differ from the
frozen run.

**Exit:** either ship DSP plus the residual, ship a controlled DSP/residual
mixture, or retain DSP with a documented reason.

## 13. Proposed run artifact layout

Use a clear run layout and delete stale pilots when they are no longer useful:

```text
runs/channel_data_fastcut/
  split_manifest.jsonl
  leakage_audit.json
  train/
    presets.jsonl
    noise/
  val/
    real_manifest.jsonl
  test/
    real_manifest.jsonl

runs/clean_base_fastcut/
  manifest.jsonl
  clean/
  resolved_config.yaml
  hashes.json

runs/fastcut_m0_<run-id>/
  resolved_config.yaml
  inputs.json
  domain_a_cache.jsonl
  checkpoints/
    step_000500.pt
  state_latest.pt
  metrics.jsonl
  validation_report.json
  auditions/
  summary.json

runs/channel_matrix_fastcut/
  views/<pipeline>/manifest.jsonl
  mixture_specs/
  arms/
  eval/
  summary_model_select.json
  summary_heldout_tail_check.json
```

Every run input should identify content hashes, not merely mutable paths.

## 14. Risks and mitigations

| Risk | Evidence/sign | Mitigation |
|---|---|---|
| Discriminator learns speaker/content differences | tiny unpaired Domain B with different speech | DSP anchor, DiffAugment/possibly ADA, identity NCE comparison, held-out content guards, frozen-feature probes. |
| Residual is not channel | discriminator rewards speaker, accent, or prosody differences indistinguishable from channel under unpaired training | bounded residual, identity-NCE arm, §6.3 attribution, probe/parallel-data acquisition, and claims renamed as unpaired target-audio residuals. |
| Station averaging | macro-station metrics diverge | station-stratified selection; M1 conditioning only if needed. |
| Speech deletion/corruption | bins floored to zero, teacher/entity regression | bounded residual, saturation audit, target identity NCE, hard gate. |
| Source-phase limitation | cannot add independent radio noise/phase | retain explicit noise/post-effects; compare real-noise replay; defer complex model. |
| GAN memorization | nearest real neighbors too close; train/val gap | grouped splits, train-only artifacts, nearest-neighbor audit, early stop, more sessions. |
| KID noise or metric gaming | conflicting metrics/no downstream gain | frozen hard gates, one held-out score, earliest-checkpoint tie-break, and real ASR screen. |
| Underpowered endpoints | plausible sub-point effects versus roughly six-point tail MDE | `model_select`-based decision, descriptive tail, and a future sized test. |
| Premise failure | calibrated DSP loses to procedural downstream | G1-versus-G3 precheck before residual investment. |
| Generator fingerprints | high pipeline classifier and weak real WER | mixtures, paired consistency, optional R2S-style invariance. |
| Mixture dilution | all-pipeline arm underperforms | exact primary contrast and conditional exploratory ratio sweep. |
| Stale artifacts | existence-based reuse | immutable stage fingerprints and fail-closed checks. |
| MPS contention/instability | variable speed or memory pressure | serial resumable jobs, idle-device benchmark, 5080 optional for promoted runs. |
| Target mismatch | own SDR stations differ from public ASR eval | report source slices, collect target-source test/calibration, avoid universal claims. |
| Too little channel data | unstable seeds/folds | grouped cross-validation, acquire new sessions, prefer DSP if learned effect is unstable. |

## 15. Decision ledger

### Adopt now

- Run the G1-versus-G3 premise check before residual investment.
- Repair leakage and validate grouped channel train/validation folds.
- Render paired views with the minimal cache and shared pre-residual waveform.
- Run the single `M0/source-NCE` versus `M0/source+identity-NCE` ablation, with
  the identity term fully specified in §4.2.
- Select checkpoints by the §6.2 lexicographic rule.
- Run the bounded go/no-go ASR screen and record the decision.
- Describe all current-scale results as development/feasibility evidence and
  the learned output as an unpaired target-audio residual.

### Defer until evidence asks for it

- station/preset-conditioned FiLM;
- stochastic residual latent;
- frozen-feature discriminator;
- long-context or complex-phase generator;
- channel-patch consistency and generator-adversarial ASR objectives;
- adaptive bandit allocation; and
- the full G-matrix and mixture program, conditional on a “go.”

### Reject as the main path

- unbounded TTS-to-real CycleGAN;
- GAN-only synthetic corpus without a DSP/real anchor;
- one-metric checkpoint selection;
- unpaired generator comparisons that rerender TTS;
- equal-quota top-up gating;
- raw-manifest-size mixture ratios;
- long training before leakage repair;
- the 427-row tail as a locked test; and
- treating the 5080 as a prerequisite for experimentation.

## 16. Recommended immediate next milestone

Execute the §12 minimal critical path and stop at its decision boundary. Its
acceptance criteria are:

1. the automated leakage audit reports zero forbidden source IDs;
2. the paired S1 `M0/source-NCE` versus `M0/source+identity-NCE` ablation is
   complete under identical settings and paired seeds;
3. one 5k candidate remains inside every content guard, or a documented
   negative result explains why none does;
4. the ASR-screen verdict records per-seed synthetic-only and FC-A2-versus-
   FC-A4 numbers; and
5. the go/no-go decision and VC/accent-branch kickoff are recorded.

This milestone either funds the conditional full comparison program or ends
residual investment cleanly while retaining G8 as the cheap channel challenger.

## 17. Primary-source bibliography

### Channel translation and limited-data GANs

- Park et al., [Contrastive Learning for Unpaired Image-to-Image
  Translation](https://arxiv.org/abs/2007.15651), ECCV 2020; official
  [CUT/FastCUT code](https://github.com/taesungp/contrastive-unpaired-translation).
- Chu et al., [CycleGAN, a Master of
  Steganography](https://arxiv.org/abs/1712.02950), 2017.
- Zhao et al., [Differentiable Augmentation for Data-Efficient GAN
  Training](https://arxiv.org/abs/2006.10738), NeurIPS 2020.
- Karras et al., [Training Generative Adversarial Networks with Limited
  Data](https://arxiv.org/abs/2006.06676), NeurIPS 2020.
- Sauer et al., [Projected GANs Converge Faster](https://arxiv.org/abs/2111.01007),
  NeurIPS 2021.
- Bińkowski et al., [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401),
  ICLR 2018.
- Chen et al., [Noise-Robust Speech Recognition with 10 Minutes Unparalleled
  In-Domain Data](https://arxiv.org/abs/2203.15321), ICASSP 2022.
- Maben et al., [Study of GANs for Noisy Speech Simulation from Clean
  Speech](https://arxiv.org/abs/2305.12460), 2023.
- Chen et al., [Unsupervised Noise Adaptation using Data
  Simulation](https://arxiv.org/abs/2302.11981), ICASSP 2023.
- Guo et al., [DENT-DDSP](https://arxiv.org/abs/2208.00987), 2022.
- Wang et al., [CADA-GAN](https://arxiv.org/abs/2409.12386), ICASSP 2025;
  official [implementation](https://github.com/JethroWangSir/CADA-GAN).
- Wang et al., [Universal Robust Speech Adaptation for Cross-Domain Speech
  Recognition and Enhancement](https://arxiv.org/abs/2602.04307), accepted to
  IEEE TASLP, 2026.
- Hu et al., [QS-Attn](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_QS-Attn_Query-Selected_Attention_for_Contrastive_Learning_in_I2I_Translation_CVPR_2022_paper.html),
  CVPR 2022.
- Fu et al., [PHASEN](https://ojs.aaai.org/index.php/AAAI/article/view/6489),
  AAAI 2020.
- Hu et al., [DCCRN](https://www.isca-archive.org/interspeech_2020/hu20g_interspeech.html),
  Interspeech 2020.
- Jang et al., [UnivNet](https://arxiv.org/abs/2106.07889), 2021, and
  [GAN Vocoder: Multi-Resolution Discriminator Is All You
  Need](https://arxiv.org/abs/2103.05236), 2021.
- Sasaki et al., [UNIT-DDPM](https://arxiv.org/abs/2104.05358), 2021.
- Su et al., [DDIB](https://arxiv.org/abs/2203.08382), ICLR 2023.
- Kim et al., [Unpaired Image-to-Image Translation via Neural Schrödinger
  Bridge](https://openreview.net/forum?id=uQBW7ELXfO), ICLR 2024.

### Synthetic speech for ASR and ATC

- Bagat et al., [Synthetic Audio Generation Framework for Air Traffic Control
  Speech Recognition](https://arxiv.org/abs/2606.21340), Interspeech 2026;
  official [code](https://gitlab.inria.fr/rbagat/atc_generation).
- Zuluaga-Gomez et al., [ATCO2 corpus](https://arxiv.org/abs/2211.04054), 2022.
- Ogun et al., [An exhaustive study of TTS/VC synthetic-data augmentation for
  ASR](https://arxiv.org/abs/2503.08954), 2025.
- [How to Leverage Synthetic Speech for LLM-Based
  ASR?](https://arxiv.org/abs/2606.29031), 2026.
- Minixhofer et al., [Scaling behavior of synthetic speech for
  ASR](https://www.isca-archive.org/interspeech_2025/minixhofer25_interspeech.html),
  Interspeech 2025.
- Liu et al., [Selecting TTS data for ASR](https://arxiv.org/abs/2306.00998),
  2023.
- Bartelds et al., [COWERAGE](https://arxiv.org/abs/2203.09829), 2022.
- Tsunoo et al., [TTS, CycleGAN, and pseudo-label data
  augmentation](https://www.isca-archive.org/interspeech_2021/tsunoo21_interspeech.html),
  Interspeech 2021.
- Wang et al., [SCADA consistency
  training](https://www.isca-archive.org/interspeech_2020/wang20aa_interspeech.html),
  Interspeech 2020.
- Pérez-Parada et al., [Patched multi-condition
  training](https://www.isca-archive.org/interspeech_2022/pesoparada22_interspeech.html),
  Interspeech 2022.
- Tran et al., [R2S real-to-synthetic representation
  alignment](https://www.isca-archive.org/interspeech_2025/tran25_interspeech.html),
  Interspeech 2025.
- Kim et al., [A Systematic Study of Fréchet Speech
  Distance](https://arxiv.org/abs/2601.21386), 2026.
- Zuluaga-Gomez et al., [Contextual semi-supervised learning for ATC
  callsigns](https://www.isca-archive.org/interspeech_2021/zuluagagomez21_interspeech.html),
  Interspeech 2021.
- Kocour et al., [Callsign boosting for air-traffic-control
  ASR](https://www.isca-archive.org/interspeech_2021/kocour21_interspeech.html),
  Interspeech 2021.

## Final proposal

Proceed, but as a bounded two-to-three-week go/no-go. First verify that
calibrated DSP is competitive with procedural matched, repair leakage, run the
single paired identity-NCE ablation, select one 5k candidate lexicographically,
and test it with the sensitive synthetic-only and FC-A2-versus-FC-A4 screens.
Fund the full comparison program only on a positive result.

The likely product remains a frozen, auditable channel component whose
moderate residuals are sampled beside DSP, with every utterance content-checked
before ASR training. VC/accent is the higher-ceiling lever for the
synthetic-only gap and starts after the first screen regardless of its outcome.
At the current data scale, every claim is development/feasibility evidence;
confirmatory claims wait for prospectively collected, session-disjoint real
data.
