# ATC Speech Recognition: Datasets, Benchmarks, and Synthetic-Data Work (as of Aug 2026)

## 1. ATC speech corpora

**ATCO2** (Zuluaga-Gomez et al., "ATCO2 corpus...", arXiv:2211.04054, https://arxiv.org/abs/2211.04054; "Lessons Learned in ATCO2", arXiv:2305.01155). The largest ATC resource: ~5,281 h of VHF audio collected via volunteer receivers (LiveATC-style feeders) + OpenSky Network metadata, with **automatic** (pseudo-label) transcripts (ATCO2-PL set), plus a **4 h manually transcribed test set** with callsign/command/value NER and speaker-role tags. A **1 h test subset is free** (HF: `Jzuluaga/atco2_corpus_1h`); the 4 h set and PL set are sold through ELDA. Audio is noisy VHF; the companion benchmark paper characterizes ATCO2-Test at **SNR ~10–15 dB** and LiveATC-Test at **5–15 dB**. Corpus was explicitly designed to avoid the licensing problems of older ATC sets. GitHub: https://github.com/idiap/atco2-corpus

**UWB-ATCC** (Univ. of West Bohemia, Pilsen). Real Czech-airspace pilot/controller English, manually transcribed with speaker-role labels; free download. Reported ~20 h total; the wav2vec2 benchmark paper uses 10.4 h train / 2.6 h test, and lists it as relatively clean (SNR ≥ 20 dB). HF mirror: `Jzuluaga/uwb_atcc`.

**ATCOSIM** (TU Graz, LREC 2008, https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html). **10.7 h** of non-prompted controller speech from real-time simulations, close-talk headset (clean, 32 kHz), **10 non-native English speakers** (German/Swiss/French). Free. Main caveats: simulated, tiny speaker pool — random-split WERs on it are wildly optimistic (see §2).

**LDC ATCC** (LDC94S14A). Real FAA audio from 3 US airports (DFW, Logan, Washington); sources variously cite ~70 h total (~26 h speech after silence removal); 8 kHz; LDC license fee. The wav2vec2 benchmark uses 23 h train / 2.6 h test, SNR 10–15 dB.

**HIWIRE**. ~28 h, simulated cockpit environment, **prompted** military-style phrases by French, Greek, Italian, Spanish speakers — useful for accent studies, unrealistic phraseology/prosody.

**ATCSpeech** (Yang et al., Interspeech 2020, http://www.interspeech2020.org/uploadfile/pdf/Mon-1-10-1.pdf). Real Chinese-airspace multilingual corpus: 16,111 English utterances (~17.5 h), 28,927 Chinese (~23.2 h), 16,645 mixed (~15.8 h). Access by application, not open.

**jacktol/atc-dataset** (HF, https://huggingface.co/datasets/jacktol/atc-dataset). Community set = ATCO2 1-h subset + UWB-ATCC, cleaned/re-split: **14,795 clips (11.9k train / 2.93k test)**, 0.3–15.3 s clips, MIT-licensed metadata. Page carries a deprecation notice pointing to a newer, higher-quality version.

**ATCCaps** (Li et al., arXiv:2606.22399, June 2026, https://arxiv.org/abs/2606.22399). New call-sign-aware dataset: **202.94 h, 170,385 utterances, 922 normalized callsigns**, built with confidence-aware transcript parsing + **ADS-B-derived callsign metadata**, rule-based filtering, LLM-assisted captions; eval subset anchored on manually annotated ATCO2-test. Their ATC-adapted Whisper baseline: **14.85% WER / 9.44% CER** on ATCCaps eval, **20.02% WER** on UWB-ATCC valid.

**Private sets** frequently seen in papers (not obtainable): NATS (18 h, SNR ≥20 dB) and ISAVIA (14 h) from the ATCO2/HAAWAII projects; LiveATC-Test (1.8 h, SNR 5–15 dB).

## 2. ASR benchmarks: what WER is achievable

**Whisper zero-shot is very poor on ATC; most of the headline gap is transcript-format mismatch.** Whisper-ATC (van Doorn et al., ICRAT 2024, https://research.tudelft.nl/en/publications/whisper-atc-open-models-for-air-traffic-control-automatic-speech-/ ; code https://github.com/jlvdoorn/WhisperATC), Whisper large-v2:

| Condition | ATCO2 | ATCOSIM |
|---|---|---|
| Zero-shot, raw | 71.62% | 79.11% |
| + ATC normalization | 29.05% | 17.98% |
| + domain prompt | 24.03% | 16.74% |
| Fine-tuned (A2-AS) | **13.46%** | **1.17%** |

Across sizes (fine-tuned): ATCOSIM 1.19–3.5% for most sizes; ATCO2 ranges 14.66% (large-v2) to >70% (tiny). A follow-up (Journal of Open Aviation Science 2026, https://journals.open.tudelft.nl/joas/article/download/8477/6530/35422) reports ATCOSIM **1.17% random split vs 3.88% speaker-split** — i.e., random splits on 10-speaker corpora leak speakers.

**jacktol Whisper medium.en** (blog + HF, https://jacktol.net/posts/fine-tuning_whisper_for_atc/): on ATCO2-1h+UWB-ATCC test, pretrained 94.59% → fine-tuned **15.08% WER** (84% relative). Notes severe zero-shot hallucination on short clips.

**wav2vec 2.0 / XLS-R benchmark** (Zuluaga-Gomez et al., ICASSP 2023, arXiv:2203.16822, https://arxiv.org/abs/2203.16822). Fine-tuning w2v2-Large-60k on ~32 h ATC: **5.4% WER (NATS, clean), 7.3% (ISAVIA)** with 4-gram LM; on noisy sets, models fine-tuned on 132 h reach **19.8% (ATCO2-Test)** and **24.9% (LiveATC-Test)** vs Kaldi hybrid baselines of 24.7%/35.8%. E2E gave 20–40% relative WER reduction over hybrids. Paper states operational deployments target **<5% WER**.

**Kaldi/hybrid with ATCO2 pseudo-labels** ("Lessons Learned", arXiv:2305.01155): CNN-TDNNf trained on ATCO2 pseudo-labels reaches **17.9%/24.9%** on public ATC test sets, 6.6–7.6% absolute better than out-of-domain supervised training — pseudo-labeled VHF data works.

**Accent adaptation**: fine-tuning for Southeast-Asian-accented ATC gives **9.82% WER** on a self-built SEA test set (arXiv:2502.20311, https://arxiv.org/abs/2502.20311). Whisper-ATC also reports region-specific fine-tuning improving real-world performance up to 60% relative.

**In-domain SSL**: BEST-RQ pre-training on 4.5k h unlabeled ATC improves offline and streaming ASR (arXiv:2509.12101; Springer 2026, https://link.springer.com/chapter/10.1007/978-3-032-07959-6_1).

Realistic expectations: **~1–4% WER on clean simulated speech (ATCOSIM), ~9–15% on real accented/regional data with fine-tuning, ~13–20% on noisy VHF (ATCO2-type), floor so far ~13% on ATCO2.**

## 3. Synthetic / TTS-generated ATC audio for ASR training

**The rumored Interspeech 2026 paper is real and verified**: Bagat, Zhang, Yamagishi, Illina, Vincent, **"Synthetic Audio Generation Framework for Air Traffic Control Speech Recognition," Interspeech 2026, arXiv:2606.21340** (https://arxiv.org/abs/2606.21340). This is the closest prior art to atc-gan. Pipeline: **F5-TTS** (voice-cloning flow-matching TTS) + **kNN-VC** voice conversion (speaker diversity via L2-ARCTIC references) + **TokAN-based controllable L1→L2 accent conversion**, plus **acoustic channel simulation: downsample to 8 kHz → upsample to 16 kHz, 200 Hz high-pass, background-noise injection using noise separated from real ATCO2 recordings**. Data scale is small (4 h synthetic matching 4 h real ATCO2). Key numbers (Whisper-small on ATCO2 test): out-of-box 63.32%; fine-tuned on real only **22.69%**; best mix (L1→L2 accent-converted + real) **21.64%**; synthetic-only 24.18%; TTS w/o channel simulation 53.88% → with it 33.77% (**37% relative gain from channel simulation alone**). They filter ~35% of accent-converted outputs as TTS hallucinations via Whisper round-trip (>50% WER discard). Ablation conclusion: **accent diversity > speaker diversity; channel simulation is critical**.

**Vu & Wei (GWU), "Generating Realistic ATC Voice Communication using AI-based TTS Models," ICNS 2026** (https://bpb-us-w2.wpmucdn.com/web.seas.gwu.edu/dist/9/15/files/2026/04/Dao-DeepFakeATC-Comms-ICNS26.pdf). Uses open-source **Chatterbox** TTS with speaker conditioning on 1000 ATCOSIM utterances; framed as security/spoofing-resiliency, but contributes an **aviation-aware synthesis eval suite**: WER intelligibility, critical-token F1, speaker-embedding similarity, pitch-contour correlation, DTW log-mel distance. Finding: current TTS nails lexical fidelity and identity but has **systematic prosodic/temporal artifacts** (speaking rate, pauses, pitch contour).

**ASTRA** (arXiv:2606.18319): next-gen ATCO training simulator with autonomous "simpilots" (LLM+TTS voices) — synthetic ATC speech for human training, not ASR training. Predecessor: Idiap's virtual simulation-pilot agent (arXiv:2304.07842).

**Synthetic rare-event text**: SESAR readback-error work generates synthetic readback-error datasets by entity-substitution templating from real transcripts (Ahrenhold et al., SESAR Innovation Days 2022, https://www.sesarju.eu/sites/default/files/documents/sid/2022/paper_3.pdf).

**No published work found combining TTS with GAN-based channel/style transfer (CycleGAN) for ATC ASR** — the Bagat pipeline uses DSP channel simulation, not learned channel transfer. That appears to be open ground.

## 4. ATC-specific challenges

- **Callsigns dominate errors.** Contextual boosting of active callsigns from air-surveillance (ADS-B) data: Kocour et al., Interspeech 2021 (https://www.isca-archive.org/interspeech_2021/kocour21_interspeech.html): **+4.7% absolute WER improvement, callsign accuracy +27.1% absolute to 82.9%**; Nigmatulina et al., ICASSP 2022 two-step FST+NLP boosting (arXiv:2202.03725); callsign error 6.2%→2.8% with surveillance data (arXiv:2108.12156). ATCCaps confirms residual errors concentrate on long callsigns, locations, dense phraseology.
- **Code words / phraseology**: ICAO pronunciations (niner, tree, fife, NATO alphabet) plus real-world deviations from standard phraseology; digit-heavy entities (FLs, headings, frequencies) are long-tail and acoustically confusable.
- **Accents**: every major corpus is accent-skewed (ATCOSIM European L2, ATCSpeech Chinese, SEA dataset); accent adaptation consistently worth large relative gains (arXiv:2502.20311; Bagat 2026 shows accent diversity beats speaker diversity in synthesis).
- **Channel**: VHF AM radio, ~300–3400 Hz effective bandwidth, 8 kHz-equivalent, SNR 5–20 dB; push-to-talk clicks and truncation. Recent work does PTT-event identification to improve speech activity detection (ResearchGate: "Enhancing Speech Activity Detection in ATC via Push-to-Talk Event Identification", 2025).
- **Hallucination on noise**: Whisper hallucinates on non-speech audio at a **40.3% rate** in a systematic study ("Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio", ICASSP 2025, arXiv:2501.11378); mitigations include a "bag of hallucinations" post-filter and hidden-representation steering (arXiv:2606.07473). jacktol's ATC work independently observed repeated-token hallucination on short/noisy clips. Directly relevant to noise-only PTT segments; ATC-specific hallucination studies are scarce — training on noise-only negatives (as in this repo's hallucination-control samples) was not found published for ATC specifically.
- **Speaking rate**: ATC speech is notoriously fast; the GWU paper quantifies TTS speaking-rate/pause deviations from ATC norms as a detectable artifact — i.e., naive TTS is too slow/too pausy.

## 5. Text normalization and WER conventions

- Whisper-ATC's open normalizer (https://github.com/jlvdoorn/WhisperATC/blob/main/Evaluate/Normalizer.py) extends Whisper's EnglishTextNormalizer with ATC rules: lowercase, digits→spoken words, expand frequencies/runways/altitudes. Normalization alone cut zero-shot WER from ~72–79% to ~18–29% — **always report whether WER is raw or normalized**.
- ATCO2 defines annotation conventions per ICAO rules/ontologies (verbatim spoken numbers, greetings handling, callsign word forms) and normalizes gold vs automatic transcripts before scoring (arXiv:2211.04054, arXiv:2305.01155).
- The Airbus ATC 2018 challenge (arXiv:1810.12614) established joint WER + callsign-detection evaluation; newer work adds callsign accuracy (CSA), critical-token F1 (GWU), and callsign-aware seen/unseen splits (ATCCaps) since aggregate WER under-measures operational risk.

## 6. Emergency / unusual phraseology

Very thin literature — this is a genuine gap. What exists: synthetic readback-error generation by entity substitution (SESAR SID 2022, above); SCOPE, a lightweight LLM framework for readback monitoring (arXiv:2605.29543); ATCCaps' unseen-callsign generalization analysis. **No paper found on ASR robustness for MAYDAY/PAN-PAN or emergency phraseology recognition specifically, nor any emergency-phraseology test set** — flagged as unverified-absent (searched, nothing found). Synthetic generation of emergency traffic would be novel and defensible on rare-event grounds, mirroring the readback-error templating precedent.

## Practical takeaways

1. **Prior art to cite and beat**: Bagat et al. Interspeech 2026 is the direct competitor/reference — TTS+VC+accent conversion+DSP channel sim, only 4 h synthetic, best result 21.64% vs 22.69% real-only on ATCO2/Whisper-small. Their channel simulation is simple DSP; a learned channel model plus scale (hundreds of hours, pluggable text) is the differentiator. Their ~35% TTS-hallucination discard rate and Whisper-round-trip filtering (>50% WER reject) is a QC recipe worth adopting.
2. **Channel simulation is the single highest-leverage realism component** (37% relative WER gain in their ablation); accent diversity is second; speaker diversity alone barely helps.
3. **Evaluation setup**: test on ATCO2-test (1 h free, 4 h ELDA) + UWB-ATCC + ATCOSIM speaker-split; use/extend the WhisperATC normalizer; report raw and normalized WER plus callsign accuracy. Targets: beat 13.5% (ATCO2, fine-tuned large-v2) / ~15% (jacktol mix) with less or no real data.
4. **Free training/eval data**: UWB-ATCC (~20 h), ATCOSIM (10.7 h), ATCO2-1h, jacktol HF mix (MIT, but note deprecation notice), ATCCaps (202.94 h — check license). ELDA purchase unlocks ATCO2 4 h + 5,281 h PL.
5. **Noise-only/hallucination-control negatives are well-motivated** by the 40.3% non-speech hallucination rate but unpublished for ATC — publishable angle, as is emergency-phraseology coverage.

All claims above are sourced as cited; the two items that could not be verified are marked in §6 (no emergency-phraseology ASR work found) and §3 (no GAN-channel-transfer ATC work found).
