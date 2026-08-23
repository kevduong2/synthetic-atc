# Literature Survey: Synthetic Speech / TTS-Based Data Augmentation for ASR Training (state of the art as of mid-2026)

Scope: papers 2023–2026, emphasis on 2025–2026. All claims below are sourced; quantitative numbers are as reported by each paper. One paper (Bagat et al. 2026) is directly on ATC and should anchor the design doc.

## 1. Effectiveness of TTS-generated audio for ASR fine-tuning, and the synthetic-to-real gap

**Claim: TTS/VC augmentation delivers double-digit relative WER reductions when speech attributes are jointly augmented, but naive mixing of synthetic and real data often underperforms.**
- Citation: "An Exhaustive Evaluation of TTS- and VC-based Data Augmentation for ASR," Ogun, Colotte, Vincent (Inria), arXiv:2503.08954 (2025).
- Jointly augmenting speech attributes (duration/speaking rate, speaker, etc.) with flow-based TTS/VC reduced Conformer-Transducer WER by 11% relative on Common Voice and up to 35% relative on LibriSpeech vs. real-only training. Pitch augmentation and VC-based speaker augmentation were ineffective in their setup. They explicitly note that because synthetic speech has lower diversity, "naively combining synthetic and real data often does not yield the best results."

**Claim: The synthetic-to-real gap is architecture-dependent; end-to-end attention models suffer the most.**
- Citation: "On the Effect of Purely Synthetic Training Data for Different ASR Architectures," Hilmes, Rossenbach, Schlüter (RWTH Aachen), arXiv:2407.17997 (2024).
- On LibriSpeech dev-clean, training on purely synthetic data roughly doubled WER for an attention encoder-decoder (7.5% → 14.1%) while GMM-HMM degraded far less (8.1% → 10.0%). Also found that low-quality vocoding (Griffin-Lim) barely hurts ASR utility, and TTS training loss poorly predicts synthetic data's ASR value — relevant when choosing a TTS: perceptual quality is not the target metric.

**Claim: Purely synthetic fine-tuning works but plateaus well above real-data performance; small amounts of real data close much of the gap.**
- Citation: "Towards Improved Speech Recognition through Optimized Synthetic Data Generation," Perrin & Boulianne (CRIM), arXiv:2508.21631 (2025). Québec French, ESPnet transformer + Whisper-medium (LoRA).
- 360h fully synthetic: 26.2–27.1% WER vs. 13.6–14.2% for 85.7h real. Adding just 10h real to 350h synthetic: 24.5–25.2%; 60h real + 710h synthetic: 20.4–20.5%. Their optimization pipeline (TTS fine-tuning on target dialect, generator-verifier filtering using an ASR to reject bad synthesis above a WER threshold, sampling-temperature tuning to 0.65) cut dev WER from 30.8% to 17.5% — showing the synthesis pipeline itself has large headroom for optimization.

**Claim: Optimal real:synthetic mixing ratios cluster around modest synthetic fractions; too much synthetic degrades performance.**
- Citation: "How to Leverage Synthetic Speech for LLM-Based ASR Systems?" Labrak et al., arXiv:2606.29031 (2026): a synthetic proportion of **10–30% is optimal** for LLM-based ASR; beyond 30% performance degrades measurably. With representation-level tricks (layer-wise weighted pooling) plus RIR augmentation they matched all-real baselines using only 25% real speech (13.6h), and achieved 8.01% WER vs. 8.68% baseline (7.7% relative gain).
- Citation: "Impact of Text Origin and Real-Synthetic Data Ratio in TTS-Augmented Low-Resource ASR," IEEE (2025/2026), ieeexplore.ieee.org/document/11252042: reports a **1:1 real:synthetic ratio** performing strongly across in-domain and out-of-domain test sets in low-resource settings. (Paywalled; ratio claim from abstract only.)
- Citation: "An Analysis of the Effectiveness of Synthetic Speech Data for ASR Fine-tuning in Selected Indic Languages," Pulikodan et al., arXiv:2606.17662 (2026): intermediate mixing ratios best; purely synthetic fine-tuning degraded Whisper performance; synthetic data with mismatched acoustic characteristics actively hurt.
- Note the spread (10–30% vs 1:1): the optimum is task- and volume-dependent; treat the ratio as a hyperparameter to sweep, not a constant to copy.

**Claim: The gap can also be mitigated at the parameter level via task arithmetic.**
- Citation: "Task Arithmetic can Mitigate Synthetic-to-Real Gap in ASR" (SYN2REAL task vector), Su, Farn, Sun, Chen, Lee, EMNLP 2024, arXiv:2406.02925.
- Subtracting a "synthetic-speech direction" task vector after fine-tuning on TTS data gave an average 10.03% WER improvement over baselines on SLURP. A cheap post-hoc option if mixed training is impractical.

### Failure modes
- **Whisper hallucination**: "Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio," Barański et al., ICASSP 2025 (arXiv:2501.11378). Non-speech/noise segments reliably induce a recurring set of hallucinated phrases; a "bag of hallucinations" post-filter reduces WER. Related: "Listen Like a Teacher" (arXiv:2511.14219) mitigates hallucination via adaptive layer attention + distillation, and follow-up work found ~3 of 20 decoder attention heads cause >75% of hallucinations in large-v3, with targeted fine-tuning cutting hallucination on non-speech inputs from ~100% to ~15%. Implication: noise-only and noise-heavy segments (radio squelch, carrier noise) are exactly Whisper's hallucination trigger — include noise-only "hallucination control" samples with empty/background-only transcripts in fine-tuning data.
- **Overfitting to TTS artifacts / low diversity**: Ogun et al. (2503.08954) and the OOV literature (arXiv:2011.11564) both flag TTS artifacts and low acoustic/speaker diversity as the mechanism by which synthetic-heavy training hurts; Perrin & Boulianne's verifier-filtering exists because TTS mispronounces/skips text, silently corrupting labels.
- **Label corruption from TTS errors**: generator-verifier filtering (transcribe synthetic audio with an existing ASR, reject above a WER threshold) is the standard countermeasure (arXiv:2508.21631).

## 2. Best practices: diversity and mixing strategies

**Voice/speaker diversity is the single most-cited requirement.**
- "Improving Code-Switching Speech Recognition with TTS Data Augmentation," Yeo, Hu, Gopal, Peng, Liu, Chng, arXiv:2601.00935 (2026), using CosyVoice2: 100h real + 100h random-speaker synthetic reduced Mixed Error Rate 12.1% → 10.1%; synthetic audio that merely reuses training-set voices contributed less than audio introducing new timbres/prosody.
- "Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup" (LatentVoiceMix), Bian, Lin, Cheng, arXiv:2511.20534 (2025): mixing speaker-timbre latents (convex blends, Beta(0.5,0.5)) in a Diff-HierVC voice-conversion model beat waveform and spectrogram augmentation — Whisper fine-tuned on Wolof: 0.202 WER vs 0.242 (spectrogram) and 0.217 (waveform). Key finding: mixed timbres cluster within the real-speaker distribution, while waveform-level perturbation produces outlier timbres.

**Which attributes to vary**: Ogun et al. found speaking-rate/duration and joint attribute augmentation effective, pitch augmentation ineffective. Labrak et al. recommend pitch-shift and high-pass filtering as the priority signal-level perturbations for their setting — so evidence on pitch is mixed; rate/tempo variation is more consistently supported.

**Accent variation**: "Few-Shot Synthetic Accented Speech for ASR Fine-Tuning: What Helps and When?" Halychanskyi et al., arXiv:2604.27273 (2026, wav2vec 2.0): accent diversity helps generalization; gains scale with synthetic volume with diminishing returns; excessive synthetic-to-real ratio degrades performance. Directly relevant to ATC: Bagat et al. found generating L1-to-L2 accented speech (making native speech sound non-native) beat accent normalization.

**Text diversity matters and is cheap**: "Text Generation with Speech Synthesis for ASR Data Augmentation," Huang, Keren, et al., arXiv:2305.16333 (2023): LLM-generated text + TTS gave 9–15% relative WER improvement, with neural text generation beating traditional text augmentation. Caveat from Perrin & Boulianne: LLM-generated text alone (no real audio anchor) performed terribly (50.9–54.2% WER) until even 10h of real audio was added.

**Conversational/multi-speaker structure**: "Generating Data with Text-to-Speech and Large-Language Models for Conversational Speech Recognition," Cornell, Darefsky, Duan, Watanabe, SynData4GenAI 2024, arXiv:2408.09215: LLM-scripted dialogues + conversational multi-speaker TTS significantly outperformed classical multi-speaker simulation when fine-tuning Whisper for telephone/distant settings. Also "Mind the Gap" (arXiv:2605.15442) on the residual gap between simulated conversational mixtures and real interactions.

## 3. Whisper-specific fine-tuning with synthetic data

Direct evidence: Perrin & Boulianne (Whisper-medium + LoRA; Whisper benefited but the synthetic-vs-real ranking held), Pulikodan et al. (Whisper on Indic languages; intermediate ratios best, pure synthetic harmful), LatentVoiceMix (Whisper on Wolof), Yeo et al. (code-switching), Cornell et al. (conversational Whisper fine-tuning), and — most relevant —

**"Synthetic Audio Generation Framework for Air Traffic Control Speech Recognition," Bagat, Zhang, Yamagishi, Illina, Vincent, arXiv:2606.21340 (2026).** This is essentially prior art for this project:
- Whisper-small on ATCO2: 63.32% WER out-of-the-box; 22.69% fine-tuned on ~1.2h real ATC data.
- Synthetic-only fine-tuning got surprisingly close: VC + ATC acoustic simulation 24.18%; TTS (F5-TTS) + acoustic simulation 33.77% — VC-based generation transferred better than pure TTS.
- Best overall: real + synthetic L1-to-L2 accent-converted data, 21.64% WER.
- Their ATC acoustic simulation: downsample to 8 kHz and back up to 16 kHz, high-pass filter at 200 Hz cutoff, mix in background noise separated from real ATC recordings.
- Design lesson: accent *diversification* (adding non-native accents) beat accent normalization; channel simulation was necessary for synthetic data to be useful at all.

## 4. Closing the TTS-to-real acoustic gap

- **"Messier is better"**: Labrak et al. (2606.29031) found RIR convolution improved downstream ASR precisely by degrading perceptual quality (UTMOS 4.36 → 1.34) — the gain comes from making synthetic audio acoustically irregular like real channel recordings, not from naturalness. This is the clearest mechanistic statement in the recent literature: match the *channel*, not the *studio*.
- **Channel/codec simulation**: Bagat et al.'s 8 kHz round-trip + 200 Hz high-pass + real ATC noise beds (above); telephony codec simulation (a-law, G.722, cellular/VoIP codec groups) as augmentation is established practice ("Audio Codec Simulation based Data Augmentation for Telephony Speech Recognition," Interspeech 2019/2020-era, ResearchGate 339754199).
- **SpecAugment on top of synthetic data** remains the standard baseline for closing train-eval gaps (time/frequency masking on log-mel), and is complementary to source-level TTS diversity — it perturbs the spectrogram, while VC/latent-mixup perturbs the speaker; LatentVoiceMix showed spectrogram-only augmentation is the weakest of the three for low-resource Whisper fine-tuning.
- **Vocoder artifacts**: Hilmes et al. found vocoder quality mattered surprisingly little for ASR utility; do not over-invest in vocoder fidelity relative to channel realism and diversity.
- **Parameter-space correction**: SYN2REAL task vectors (arXiv:2406.02925) as a post-hoc alternative.

## 5. Oversampling rare/critical utterances via synthesis

- **OOV/rare words**: "Using Synthetic Audio to Improve the Recognition of Out-Of-Vocabulary Words in End-To-End ASR," arXiv:2011.11564 (2020, foundational): TTS audio for OOV words improves their recognition; flags TTS artifacts/low diversity as the risk, countered by matching synthetic to real acoustic conditions.
- **Named entities / code-switching**: "Improving Code-Switching and Named Entity Recognition in ASR with Speech Editing based Data Augmentation," Liang et al., Interspeech 2023, arXiv:2306.08588 — speech-editing (splicing synthesized entity spans into real utterances) beats whole-utterance TTS for entity coverage. Yeo et al. (2601.00935) for full-TTS code-switching augmentation.
- **Domain-term pipelines**: the pattern of {domain lexicon → LLM generates diverse carrier sentences → TTS → noise/channel augmentation} is documented in applied work (e.g., AWS Nemotron domain-adaptation writeup, 2025-2026) and in "Improving Synthetic Data Training for Contextual Biasing Models with a Keyword-Aware Cost Function" (arXiv:2509.09197), which weights the loss toward the rare keywords inside synthetic utterances rather than treating all tokens equally.
- **Dysarthric/personalized**: "Personalized Fine-Tuning with Controllable Synthetic Speech from LLM-Generated Transcripts for Dysarthric Speech Recognition," arXiv:2505.12991 (2025) — same recipe applied to speaker-level rarity.

## Practical takeaways for the design doc

1. **Treat channel realism as the highest-leverage component.** The strongest recent evidence (Labrak; Bagat) says gains come from making synthetic audio "messy" like the target channel — for ATC: 8 kHz bandwidth round-trip, high-pass ~200-300 Hz, codec/compression artifacts, and noise beds taken from *real* ATC recordings. Perceptual TTS quality is a non-goal (Hilmes: even bad vocoders are fine).
2. **Never fine-tune Whisper on synthetic-only data if any real data exists; sweep the mixing ratio.** Evidence clusters between ~10-30% synthetic (Labrak) and ~1:1 (IEEE low-resource paper); pure-synthetic degrades (Indic paper). Even 10h of real audio dramatically anchors a large synthetic corpus (Perrin & Boulianne).
3. **Speaker diversity beats per-sample quality.** Use many voices, and prefer methods that create *new plausible* timbres (VC, latent timbre mixup) over waveform perturbation of a few voices. Bagat et al. specifically found VC-derived synthetic ATC audio outperformed TTS-derived.
4. **Diversify accents rather than normalizing them** — L1-to-L2 accent conversion was Bagat et al.'s best single addition for ATC.
5. **Vary speaking rate/tempo; treat pitch augmentation as unproven** (helped in one setup, ineffective in another).
6. **Generate text with an LLM for coverage of rare phraseology, but verify audio labels**: run every synthetic clip through an existing ASR and reject clips above a WER threshold (generator-verifier, Perrin & Boulianne). Consider loss-weighting rare keywords (arXiv:2509.09197) and splicing critical terms into real carrier audio (arXiv:2306.08588) rather than synthesizing whole utterances only.
7. **Defend against Whisper hallucination explicitly**: include noise-only/squelch-only segments with empty transcripts in the fine-tuning mix, and consider a bag-of-hallucinations post-filter (Barański et al., ICASSP 2025). This validates the hallucination-control samples already in the atc-gan pipeline.
8. **Stack augmentations**: source diversity (TTS/VC voices, accents, rates) + channel simulation + SpecAugment are complementary; SpecAugment alone is the weakest lever.
9. **Fallback if mixed training is impractical**: SYN2REAL task-vector arithmetic recovers ~10% WER relative after synthetic-only fine-tuning (EMNLP 2024).
10. **Expect a floor**: even optimized pure-synthetic pipelines plateau ~1.5-2x the WER of real-data training (Perrin & Boulianne); the realistic goal is stretching limited real ATC hours, not replacing them.

Uncertain/weakly sourced items, flagged: the IEEE 1:1-ratio paper was abstract-only (paywalled); the "3 of 20 attention heads cause hallucinations" figure came from a search summary of follow-up hallucination work rather than a fetched paper; the AWS domain-adaptation reference is a vendor blog, not peer-reviewed.

Key URLs: arxiv.org/abs/2606.21340 (ATC framework), arxiv.org/abs/2503.08954, arxiv.org/abs/2508.21631, arxiv.org/abs/2606.29031, arxiv.org/abs/2406.02925, arxiv.org/abs/2501.11378, arxiv.org/abs/2601.00935, arxiv.org/abs/2511.20534, arxiv.org/abs/2604.27273, arxiv.org/abs/2407.17997, arxiv.org/abs/2305.16333, arxiv.org/abs/2408.09215, arxiv.org/abs/2306.08588, arxiv.org/abs/2606.17662, arxiv.org/abs/2509.09197.
