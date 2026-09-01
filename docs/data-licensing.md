# Data provenance and licensing

This document records the provenance and license status of real-audio and text
sources touched by the project. Generated datasets are intended for commercial
use. The status below was verified by a research sweep on 2026-09-01; items
marked **ACTION REQUIRED** still need human confirmation.

## Sources currently used

### `reference-data-for-v1-run/updated_kixd_clips/`

- 7,374 KIXD (New Century AirCenter) tower clips from August 1–8, 2025.
- 16 kHz mono PCM16 with human transcripts from the asr project V2.1.2 corpus.
- Produced through our own capture and annotation pipeline; receiver-source
  confirmation is still pending.
- **ACTION REQUIRED:** Confirm that the capture source was our own receiver and
  was not derived from LiveATC. LiveATC's terms prohibit third-party and
  commercial use: "Audio streams may not be used in any third-party products."
  A human must verify the current terms at <https://www.liveatc.net/legal/>;
  Cloudflare blocks automated fetches.

### `data/real/calibration/`

- KSDL tower audio from our own SDR captures.
- Cleared for project use.

### `jacktol/atc-dataset` (Hugging Face)

- Used only as an internal evaluation and generalization signal. It must never
  enter training mixes or shipped artifacts.
- Its MIT tag is **VOID** for the bundled data. The dataset card identifies
  ATCO2-1h and UWB-ATCC as its sources. The ATCO2 end-user agreement sections 3
  and 4 allow internal research use only and prohibit redistribution and
  derivative products. UWB-ATCC is CC BY-NC-SA 4.0, which requires
  non-commercial use and share-alike distribution.
- These restrictions extend to derived models, including
  `jacktol/whisper-medium.en-fine-tuned-for-ATC`, and re-hosted combinations,
  including `Tabys/ATC_combined`, `Shiry/ATC_combined`,
  `jacktol/ATC-ASR-Dataset`, and `Jzuluaga/atco2_corpus_1h`.
- Plan: replace this evaluation source with own-capture or cleared corpora
  listed below.

### `synthetic_generation_deployed_airports_v2.0.1.jsonl`

- Internally supplied, colleague-generated text scenes covering six airports.

## Cleared for commercial use (candidates, not yet integrated)

### ATCOSIM (TU Graz and EUROCONTROL)

- 10.7 hours of clean, close-talk ATC simulation speech.
- The license explicitly permits research and development use "also in a
  commercial environment." Redistribution is limited to one's own
  organisation.
- Best use: dry-speech input for voice conversion and real phraseology text.
- Source: <https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html>

### TartanAviation (CMU AirLab)

- 3,374 hours of real US-tower VHF from self-recorded receivers: KAGC on 121.1
  MHz and KBTP CTAF on 123.05 MHz, sampled at 44.1 kHz and synchronized with
  ADS-B.
- The repository code is BSD-3-Clause and the paper is CC BY 4.0, but the data
  license is unstated.
- **ACTION REQUIRED:** Email the CMU AirLab authors and obtain written
  confirmation of the data license before use.
- Best use after clearance: noise beds, channel-calibration reference, and
  untranscribed adaptation audio.
- Source: <https://github.com/castacks/TartanAviation>

### ATCO2 full corpus (ELRA-S0484)

- Paid corpus. ATCO2 states that commercial licensing is available.
- Obtain the applicable commercial terms in writing before use.

## Not usable commercially

- **ATCO2 free 1-hour set:** The end-user agreement restricts use to bona fide
  internal research and prohibits derivative products and marketing.
- **UWB-ATCC:** CC BY-NC-SA 4.0; non-commercial and share-alike restrictions
  apply.
- **LiveATC.net audio:** The published terms prohibit third-party and commercial
  use, pending the human verification noted above.
- **LDC ATCC (LDC94S14A):** Commercial rights are available only through
  for-profit LDC membership at acquisition time.
- **DLR youtube-atc:** Inherits the YouTube Terms of Service and contains
  simulator VoIP rather than VHF channel audio.

## Policy

Every new real-audio source must be added to this file with a license verdict
before it enters calibration, training, or evaluation. Restricted data may be
used evaluation-only for internal development, but it must never enter training
mixes, shipped models, or published datasets.
