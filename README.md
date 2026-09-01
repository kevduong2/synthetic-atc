# atc-gan — synthetic ATC radio audio for ASR training

Generates label-correct, channel-realistic air-traffic-control radio audio and
uses it to train ASR models for noisy VHF transmissions. The design follows the
engineering proposal in `docs/research-findings.md` (flow-matching-era synthetic
data platform, three separated RL loops, verification-gated training data);
`docs/plans/research-integration.md` maps that proposal onto this codebase at
proof-of-concept scale and records the deliberate deviations.

## Pipeline

```
scenario service          speech            radio-channel twin        verification gate
ICAO grammar + entity  →  Kokoro TTS +   →  DSP chain (procedural  →  multi-teacher consensus
ground truth + EU/US      voice augment     or per-clip calibrated)   + entity fidelity
phraseology, validated                      + optional CUT residual   → gold/silver/adversarial
        ↑                                                                     ↓
   L3 recipe bandit  ←———  eval platform (entity panel, D11 splits)  ←—  ASR training
   (Thompson + hardness    WER/SDI, callsign acc, critical-number        SFT → mix → GRPO
   window, counterfactuals) substitution rate, slices                    (whisper student)
```

Key modules:

- `atcgen/text/` + `atcgen/entities.py` — scenario grammar with structured
  entity ground truth (callsign/runway/heading/altitude/FL/frequency/…),
  deterministic validator, vocabulary anchored on real transcripts
  (`scripts/harvest_vocab.py`).
- `atcgen/channel/` — parameterized DSP channel twin (bandpass re-applied at
  the receiver-filter and delivery points, squelch, AGC, codec, noise beds
  from real recordings), per-clip calibrated mode, capped domain-randomization
  envelope (`configs/real_envelope.json`), EBU R128 loudness.
- `atcgen/gate/` — every sample passes word- and entity-level verification by
  frozen teachers (whisper-base.en + wav2vec2 CTC) before it may train
  anything; reject, never relabel. Tiers: gold / silver / adversarial (≤5% of
  any mix) / rejected, with full lineage.
- `atcgen/eval/` + `training/evaluate.py` — channel-fidelity tiers (stats,
  KID, probe) and the downstream entity panel; `atcgen/dataset/splits.py`
  pins the disjoint real splits (locked_test is read once per arm).
- `training/` — student recipe: SFT on real → continued SFT on gated synthetic
  mix → GRPO with hallucination/repetition/length penalties and KL to the SFT
  checkpoint (`training/recipe.py`, `training/grpo.py`).
- `atcgen/rl/` — outer config search (CEM vs. random control) and the L3
  Thompson-sampling recipe bandit with the teacher-bounded hardness window.

## Documentation

Start with the [documentation index](docs/README.md). The plain-Markdown
architecture overview is in [docs/architecture.md](docs/architecture.md),
with the illustrated version in [the systems manual](docs/systems-manual.html).
Use [docs/cli-reference.md](docs/cli-reference.md) for commands and
[docs/results.md](docs/results.md) for the current evidence snapshot.
[Data provenance and license status](docs/data-licensing.md) covers every real-audio and text source used or considered by the project.

## Quickstart

```bash
uv sync                                  # deps (Mac/MPS supported)
uv run pytest -q                         # test suite

# generate + gate a dataset
uv run python scripts/generate_dataset.py --config configs/mode1_matched.yaml \
    --n-samples 500 --out runs/demo --seed 7 --text "grammar:region=eu"
uv run python scripts/gate_dataset.py --dataset runs/demo

# the full validation matrix (A0 zero-shot … A4 mix+GRPO)
uv run python scripts/run_matrix.py --out runs/matrix_v1

# L3 recipe bandit
uv run python scripts/rl_recipe_bandit.py --pulls 30 --out runs/bandit_v1
```

Real proof-of-concept corpus: `jacktol/atc-dataset` (ATCO2-1h + UWB-ATCC).
Channel calibration references: `data/real/calibration/` (own SDR captures).

## Status

Pre-release; nothing is pinned and schemas may change freely (see `AGENTS.md`).
Experiment results live in `runs/` (gitignored); the current validation-matrix
summary is written to `runs/matrix_v1/summary_locked_test.json` by
`scripts/run_matrix.py`, and `docs/results.md` snapshots the current
experimental evidence in-repo.
