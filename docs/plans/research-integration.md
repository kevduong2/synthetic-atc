# 07 — Research-findings integration plan

Maps `docs/research-findings.md` (the Aug 2026 engineering proposal) onto this
codebase at proof-of-concept scale. Scope discipline: everything here runs on
the Mac (MPS); items that genuinely need the 5080 or licensed data are recorded
as deferred, not stubbed.

## What the proposal settles that we adopt now

| Proposal | Local realization |
|---|---|
| §4.1 scenario-service: structured entity ground truth + deterministic validator | `atcgen/entities.py` (shared schema) + upgraded `atcgen/text/` grammar emitting `entities`, display+spoken forms, European/ICAO phraseology anchored on real-train vocab, phonetic-respelling knob |
| §4.4 verification-gate (D8: reject, never relabel) | new `atcgen/gate/`: multi-teacher consensus, word/entity fidelity, audio validity, gold/silver/adversarial/rejected tiers, lineage in manifest |
| §4.8 evaluation-platform entity panel | `atcgen/eval/entity_metrics.py` + harness/CLI integration: WER+S/D/I, callsign acc, slot F1 per entity type, critical-number substitution rate, hallucination metrics, slices |
| §4.6 L2 recipe: SFT → synth → GRPO with anti-hallucination penalties | `training/` staged recipe + `training/grpo.py` for whisper-tiny.en (KL to SFT ckpt, repetition/length/hallucination penalties from day one) |
| §4.7 L3 bandit with hardness window | `atcgen/rl/bandit.py`: Thompson-sampling recipe bandit over (scenario class, voice, rate, SNR band, channel condition, difficulty), hardness window `WER_teacher < τ1`, `τ2 < WER_student < τ3` |
| §4.3 channel-twin details | bandpass re-applied after augmentation steps, loudness normalization, capped domain-randomization envelope from calibration stats (KSDL/KSLE measurements) |
| D10 real anchor | scenario vocab (airlines/waypoints/stations) harvested from jacktol train split only |
| D11/§4.8 split discipline | `atcgen/dataset/splits.py`: deterministic disjoint slices — real-train / reward-val / model-selection / locked-test; locked-test touched once |

## Deviations (PoC-scale, deliberate)

- **Generator class (D1/D2, L1 GRPO):** stays Kokoro TTS + DSP/CUT channel.
  Flow-matching TTS training + generator GRPO need the 5080 + cleared
  pretraining data (D13 blocks public F5 weights). The *outer* config loop
  (`atcgen/rl/`) and the new L3 bandit are the RL we can run honestly here.
- **Judge diversity (D4):** student = whisper-tiny.en; teachers =
  whisper-base.en + faster-whisper-small (int8) + wav2vec2-base-960h (CTC —
  the architecturally-distinct one). True Canary-class judges deferred.
- **VC/accent branch (D6):** deferred to a follow-up wave (kNN-VC toward real
  ATC speakers); noted as the biggest published lever (24.2% vs 33.8% synth-only).
- **Split leakage:** jacktol is utterance-segmented; speaker/callsign overlap
  across its train/test is possible. Prohibited in production (§4.8), accepted
  and documented for the PoC. User's own transcribed set replaces it later.

## Shared contract: `atcgen/entities.py`

Single module consumed by scenario validator, gate, and eval. Spoken-form-first
(transcripts in this project are verbatim spoken words, ATCO2 style).

- `Entity(type, value, spoken, critical)` — `type` in {callsign, runway,
  heading, altitude, flight_level, frequency, speed, squawk, altimeter,
  waypoint, atis}; `value` canonical (e.g. `"CSA123"`, `"24L"`, `"FL350"`,
  `"127.825"`); `critical=True` for numeric safety entities.
- `spoken_number_words(...)` / parsing helpers: bidirectional spoken↔canonical
  digits incl. niner/tree/fife, decimal, group forms ("one twenty seven").
- `extract_entities(text, airlines=...) -> list[Entity]` — best-effort parser
  over spoken-form text (used on teacher/model hypotheses and real refs).
- `score_entities(ref, hyp) -> EntityScore` — per-type precision/recall/F1,
  exact callsign accuracy, critical-number substitution rate (a critical
  entity present in both but with different value).
- Grammar emits ground-truth entities directly (no parsing needed on synth
  refs); parser exists for hypotheses and real transcripts.

## Manifest row extensions (dataset builder)

Add: `entities` (JSON list), `display` text, `tier` (gold/silver/adversarial/
rejected), `gate` (teacher WERs, entity verdicts, audio checks, reasons),
`lineage` (config hash, seed, profile, text source, channel params). Pre-release:
break the schema freely (AGENTS.md).

## Experiment protocol (the validation run)

Student whisper-tiny.en, normalization frozen (`training/normalize.py`), eval
via the entity panel on **locked-test = jacktol test[500:2500]** (test[0:500]
was spent by rl_verify in a prior session). Splits: real-train train[0:8000],
reward-val train[8000:9000], model-selection train[9000:10000].

Budget-matched optimizer steps across arms:

| Arm | Training |
|---|---|
| A0 | zero-shot |
| A1 | SFT on real-train |
| A2 | SFT on gold-gated synthetic only |
| A2u | SFT on the same synthetic pool, ungated (gate-value ablation, WAVe mirror) |
| A3 | SFT on 75% real / 25% gold synthetic |
| A4 | A3 checkpoint + GRPO (reward −WER + penalties, KL to A3) |

Exit bars (PoC-scaled from §7): A2 within ~1.5–5 abs WER of A1 (≤1.5 = published
parity bar met); A3 ≤ A1; A4 < A3 with no hallucination-rate regression;
A2 < A2u (gate earns its cost). Entity panel reported for every arm; release
logic follows §4.8 (entity metrics gate, not aggregate WER alone).
