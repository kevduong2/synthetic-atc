# Practical Training Recipes: Small-Data Generative Audio Models and Whisper Fine-Tuning (as of 2026)

Scope: (a) training a channel-translation generative model (CycleGAN-style or diffusion) on ~1,000 short noisy radio clips; (b) fine-tuning Whisper small/medium on ~10k synthetic+real ATC samples. Hardware: one RTX 5080 (16 GB). Claims are cited where sources exist; items marked **[lore]** are practitioner consensus without a single canonical citation.

## 1. Small-data GAN training

**Discriminator overfitting is THE failure mode below ~10k images.** Baseline StyleGAN2 shows severe discriminator overfitting (FID > 30 on 10k images); with augmentation-based regularization, a few thousand images suffice. Two canonical fixes, both directly portable to spectrogram GANs:

- **ADA — adaptive discriminator augmentation** (Karras et al., "Training GANs with Limited Data," NeurIPS 2020, arXiv:2006.06676; official impl. github.com/NVlabs/stylegan2-ada-pytorch). Differentiable pixel/geometric/color augmentations applied to every discriminator input, with augmentation probability p adjusted online from an overfitting heuristic (the fraction of real images getting positive discriminator output, target ~0.6). No loss or architecture changes; works from scratch and when fine-tuning. Matches full-data StyleGAN2 with ~10x fewer images.
- **DiffAugment** (Zhao et al., NeurIPS 2020, arXiv:2006.10738; github.com/mit-han-lab/data-efficient-gans). Simpler fixed-probability variant: apply the same differentiable augmentation (color, translation, cutout) to both real and fake images in both G and D updates. Demonstrated usable results from as few as 100 images. For spectrograms, restrict to augmentations that are physically meaningful: time shifts/masks yes, vertical (frequency) flips no **[lore]**.

**Regularization.** R1 gradient penalty (Mescheder et al., "Which Training Methods for GANs do actually Converge?", ICML 2018, arXiv:1801.04406) is the standard discriminator regularizer; its weight gamma is the single most important hyperparameter to sweep — StyleGAN2-ADA's heuristic is gamma ∝ resolution²/batch size, and DiffAugment used gamma = 1 at 256². Sweep gamma over orders of magnitude (0.01–10) rather than fine-grained **[lore, echoed in StyleGAN2-ADA README]**. Spectral normalization (Miyato et al., ICLR 2018, arXiv:1802.05957) is the common alternative in audio discriminators; GAN vocoders (MelGAN, HiFi-GAN) typically use weight norm in G and spectral/weight norm in D.

**EMA of generator weights is standard and nearly free.** An exponentially averaged copy of G consistently outperforms the live generator (Yazıcı et al., "The Unusual Effectiveness of Averaging in GAN Training," ICLR 2019, arXiv:1806.04498); decay 0.999–0.9999, with 0.9999 best in large-scale studies (also central in diffusion practice — Karras et al., arXiv:2312.02696). Always evaluate/sample from the EMA weights.

**Transfer learning beats from-scratch on 1k images.** FreezeD (Mo et al., arXiv:2002.10964): fine-tune a pretrained GAN with the lower discriminator layers frozen — a simple, strong baseline for small target sets. ADA is explicitly compatible with fine-tuning. StyleGAN2-ADA's docs note transfer learning typically converges in ~1,000 kimg vs ~5,000–25,000 kimg from scratch. "When, Why, and Which Pretrained GANs Are Useful?" (ICLR 2022, arXiv:2202.08937) finds source-dataset coverage/diversity matters more than visual similarity to the target. Caveat: pretrained checkpoints for spectrogram GANs are scarce compared to face models, so for audio this often means pretraining yourself on a larger speech corpus (e.g., LibriSpeech spectrograms) then fine-tuning on the 1k radio clips **[lore]**.

**Batch size and failure detection.** StyleGAN2-ADA defaults to batch 32–64; small batches (8–16) are workable if gamma is retuned **[lore]**. Detection signals: (1) discriminator outputs on real vs fake diverging steadily / D loss → 0 = discriminator overfitting; ADA's r_t → 1 is the same signal quantified; (2) FID or KID tracked every few hundred kimg rising after an early minimum = overfitting or collapse — small datasets hit their best FID early, then degrade (StyleGAN2-ADA README); (3) mode collapse: collapsing sample diversity, oscillating losses. KID is preferable to FID at n≈1,000 real samples since FID is biased at small n (Bińkowski et al., arXiv:1801.01401).

**Audio/spectrogram specifics.** Transposed-conv upsampling produces checkerboard/tonal artifacts — audible in vocoders; prefer nearest-neighbor upsample + conv, or use anti-aliasing designs (Avocodo, arXiv:2206.13404; FA-GAN, Interspeech 2024, arXiv:2407.04575). Multi-resolution STFT discriminators are the single most consistently helpful audio-GAN component ("GAN Vocoder: Multi-Resolution Discriminator Is All You Need," arXiv:2103.05236). If you operate on mel/magnitude spectrograms you sidestep phase during translation but need Griffin-Lim or a neural vocoder (HiFi-GAN) at the output; phase reconstruction is the main quality bottleneck **[lore + FA-GAN]**. For a channel-degradation model (clean→radio), a vocoder fine-tuned on degraded audio may be needed since pretrained vocoders assume clean speech **[lore]**.

## 2. CycleGAN-specific practice

Original recipe (Zhu et al., ICCV 2017, arXiv:1703.10593): LSGAN loss, Adam lr 2e-4 with β1=0.5, constant lr then linear decay (100+100 epochs), λ_cycle = 10, identity loss at 0.5·λ_cycle, and a **replay buffer** of the last 50 generated images fed to D (from SimGAN, Shrivastava et al. 2017) to reduce oscillation. Identity loss (feed a target-domain image to the generator, penalize change) prevents unnecessary spectral tint shifts — for spectrograms this helps preserve linguistic content and overall energy structure.

**When cycle consistency fails:** the clean→radio mapping destroys information (added noise, band-limiting), so the reverse mapping is one-to-many. Under a strict cycle loss the network resolves this by hiding the destroyed information steganographically in imperceptible high-frequency structure ("CycleGAN, a Master of Steganography," Chu et al., NeurIPS 2017 workshop, arXiv:1712.02950). Practical consequences: relax λ_cycle, use a perceptual/feature-level cycle loss, or drop the cycle entirely. In CycleGAN speech-enhancement literature, cycle consistency also causes residual noise retention (Cycle-in-CycleGAN, arXiv:2109.12591). For mel-spectrogram conversion specifically, CycleGAN-VC3 (Kaneko et al., Interspeech 2020, arXiv:2010.11672) found vanilla CycleGAN-VC2 damages time-frequency structure on mels and added TFAN normalization — evidence that image-domain CycleGAN defaults do not transfer unmodified to spectrograms.

**CUT is the more data-efficient successor** (Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation," ECCV 2020, arXiv:2007.15651; github.com/taesungp/contrastive-unpaired-translation). One-sided translation with a patch-based contrastive (PatchNCE) loss instead of the cycle: half the GPU memory, ~2x faster training (FastCUT variant), no reverse generator, and it can train even in single-image regimes. For the asymmetric clean→radio task (the reverse direction is never needed), CUT is a natural fit and avoids the steganography failure by construction. Recommendation: prototype CUT first, keep CycleGAN as the fallback.

## 3. Diffusion on small audio data, and fast inference

- **Fine-tuning a pretrained audio diffusion/flow model (LoRA or full) is the small-data path**; from-scratch diffusion on ~1k clips is feasible for narrow unconditional domains but slower to converge than a fine-tuned GAN, on the order of 1–3 GPU-days at spectrogram resolution **[lore — no canonical small-audio-diffusion benchmark exists]**. EDM2 practice (Karras et al., CVPR 2024, arXiv:2312.02696) applies: EMA decay choice materially affects quality and can be searched post-hoc.
- **Fast inference via distillation is practical on one GPU but is a second training stage.** Easy Consistency Tuning ("Consistency Models Made Easy," ICLR 2025) fine-tunes a pretrained diffusion model into a 1–2 step generator in ~100k iterations. Audio-specific: ConsistencyTTA (Interspeech 2024), FlashAudio (rectified flow + distillation, ACL 2025, arXiv:2410.12266), RFWave (multi-band rectified flow vocoding, ICLR 2025), MeanAudio (arXiv:2508.06098) — the field has converged on rectified-flow/consistency-style few-step audio generation, with 10-step rectified-flow models reaching ~160x realtime on an RTX 4090 (RFWave).
- **However: this project's inference is offline** (generate ~10k training clips once). At 25–50 DDIM/flow steps, generating 10k five-second spectrogram clips is at most hours on the 5080, so distillation is likely not worth its training cost — only pursue it if the channel model must later run inside a real-time loop.

## 4. Whisper fine-tuning recipes (2024–2026)

**Baseline recipe** (Hugging Face fine-tuning blog, Gandhi 2022 — still the reference): seq2seq trainer, lr ~1e-5 (roughly 40x below pretraining), linear decay with ~10% warmup, effective batch ~32–64, 3–5k steps for ~10–15 h of audio **[community-standard]**.

**Full FT vs LoRA.** LoRA (r=8–32 on attention projections, lr ~1e-3, optionally 8-bit base) trains Whisper-large on <16 GB and reaches within ~1 WER point of full FT on domain tasks (Vaibhav Srivastav, fast-whisper-finetuning repo). LoRA also forgets less: a rehearsal-free multilingual study on Whisper found base-language performance essentially preserved under LoRA, further improved by orthogonal-gradient O-LoRA (arXiv:2408.10680) — though forgetting under LoRA is not universally zero and depends on hyperparameters (arXiv:2512.17720). With 10k in-domain samples, full FT of Whisper-small and LoRA (or full FT + checkpointing) of Whisper-medium are both realistic on 16 GB: whisper-medium full FT fits a V100-16GB at per-device batch 2 with 16x gradient accumulation (HF docs). **[Consensus]**: at this data scale, full FT of small and LoRA of medium are the two configurations to compare.

**Freezing the encoder** reduces trainable parameters and overfitting, and is common in low-resource recipes — but it presumes the acoustic domain matches pretraining. ATC's shift is precisely acoustic (VHF channel, band-limiting, noise), so keep the encoder trainable, possibly at lower lr; freeze it only if overfitting is observed **[reasoned recommendation; both variants appear in the literature]**.

**Catastrophic forgetting of general English** matters here mainly for rare proper nouns and non-standard phraseology. Mitigations, in increasing cost: low lr + early stopping; LoRA; rehearsal (mix ~10–20% general English, e.g. LibriSpeech/Common Voice, into fine-tuning) **[standard practice]**.

**SpecAugment during FT**: supported in HF (`apply_spec_augment`); original Whisper applied it in later training. One documented pitfall: masking with value 0 collides with zero-padding and can induce end-of-audio hallucinations (openai/whisper discussion #838). Use it — it is cheap regularization at 10k samples — but validate against hallucination metrics.

**Empty-transcript / noise-only samples**: including noise-only segments with empty transcriptions (Whisper's no-speech path) is the accepted counter to hallucination on silence/static (CrisperWhisper, arXiv:2408.16589; multiple fine-tuning guides). This directly validates the hallucination-control samples already in the atc-gan pipeline. Keep them a minority (~5–10% of batches) **[lore]**.

**Timestamp tokens**: fine-tuning only on non-timestamped targets degrades Whisper's timestamp ability; the original model saw timestamps on ~50% of samples. If alignment/subtitling is ever needed, include timestamped variants for a fraction of samples (openai/whisper discussion #838; CrisperWhisper). If only transcripts are needed, ignore.

**Mixing synthetic and real.** The best-supported findings:
- Sequential curriculum — train on synthetic first, then fine-tune on real — beats joint mixing (Cornell et al., SynData4GenAI 2024; corroborated in an Indic-language Whisper study, arXiv:2606.17662).
- With incremental synthetic augmentation, WER improves monotonically with substantial gains already at +20% synthetic and saturation around +80% (arXiv:2606.17662).
- Real data remains worth far more per hour; synthetic helps most under domain shift, and TTS quality/domain match is critical — an unadapted TTS caused a ~42% relative degradation in one study.
- ATC-specific evidence that the overall program works: fine-tuned Whisper reaches 13.5% WER on ATCO2 and 1.17%/3.88% (random/speaker split) on ATCOSIM vs ~63% zero-shot for whisper-small (Whisper-ATC, van Doorn & Sun, ICRAT 2024, TU Delft); but a European-trained ATC model degrades to ~30% WER on American ATC audio — accent/phraseology match of the synthetic text+voices matters as much as the channel (Research Square 2026, rs-8970162). A 2026 synthetic-ATC-audio framework paper (arXiv:2606.21340) uses exactly the TTS + channel-simulation approach this project takes.
- Batch-level vs dataset-level mixing: no strong published evidence either way; dataset-level shuffling is the default. **[gap in literature]**

## 5. Compute estimates (single RTX 5080, 16 GB — rough)

Reference points are V100/A100 numbers; a 5080 is comfortably faster than a V100 for FP16/BF16 **[lore]**.

| Job | Estimate |
|---|---|
| CycleGAN, 256² spectrograms, 1k clips, ~200 epochs | 6–15 h |
| CUT/FastCUT, same data | roughly half CycleGAN (paper claims ~2x faster, half memory) |
| StyleGAN2-ADA-style from scratch, 256², ~5,000 kimg | ~4–7 days (V100 ref: 4,000 kimg ≈ 6 days) |
| Same via transfer learning, ~1,000 kimg | ~1 day |
| Spectrogram diffusion from scratch, small U-Net | 1–3 days **[lore]** |
| Consistency/flow distillation stage | +0.5–1 day; likely unnecessary (offline generation) |
| Whisper-small full FT, 10k samples, 3–5 epochs | 3–8 h |
| Whisper-medium, LoRA or full FT + grad checkpointing (adds 20–30% step time) | 8–24 h (A100-40GB ref for a much larger run: ~42 h) |
| HiFi-GAN vocoder fine-tune on degraded audio (if needed) | ~1 day **[lore]** |

## Practical takeaways

1. Prototype **CUT/FastCUT before CycleGAN**: one-sided, ~2x cheaper, sidesteps the cycle-consistency information-hiding failure that clean→noisy mappings provoke.
2. If using any GAN: **DiffAugment or ADA + R1 (sweep gamma) + generator EMA (0.999–0.9999)** is the non-negotiable small-data stack; use a multi-resolution STFT discriminator and avoid transposed-conv upsampling.
3. Track **KID (not FID) on the 1k real clips** and discriminator real/fake output gap; expect best checkpoint early — checkpoint often, take the pre-degradation EMA snapshot.
4. Diffusion is viable but slower to iterate; skip distillation — offline generation at 25–50 steps is cheap enough.
5. Whisper: compare **full FT of small vs LoRA of medium**; lr 1e-5 (full) / ~1e-3 (LoRA), warmup + linear decay; keep encoder trainable (the domain shift is acoustic); SpecAugment on with the padding-value caveat.
6. Use the **synthetic→real curriculum** (pretrain on synthetic, finish on real) rather than naive joint mixing; keep noise-only/empty-transcript samples (~5–10%) for hallucination control; mix in ~10–20% general English if forgetting shows up; include timestamped samples only if alignment is a deliverable.
7. Match **accent and phraseology** of synthetic text/voices to the target airspace (US vs European), per the WhisperATC→US 30% WER degradation — this is as important as channel realism.
8. Everything above fits the 5080: worst-case single jobs are ~1 day; the full pipeline (channel model + Whisper FT + ablations) is a 1–2 week compute budget.

Key sources: arXiv 2006.06676, 2006.10738, 1801.04406, 1806.04498, 2002.10964, 2202.08937, 1703.10593, 1712.02950, 2007.15651, 2010.11672, 2103.05236, 2206.13404, 2407.04575, 2312.02696, 2410.12266, 2408.10680, 2408.16589, 2606.17662, 2606.21340; NVlabs/stylegan2-ada-pytorch README; Whisper-ATC (ICRAT 2024); HF Whisper fine-tuning blog; Vaibhavs10/fast-whisper-finetuning; openai/whisper discussion #838.
