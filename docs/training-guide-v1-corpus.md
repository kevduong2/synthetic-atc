# Training with the V1.0.0 corpus

This guide is for consumers of `data/corpus/V1.0.0`. The important rule is to
filter `gate_tier` before registering the corpus with an ASR trainer: the CSVs
are complete audit exports, not preselected training sets. The gate measures
whether the post-channel waveform supports its label; it is not a general
quality or realism score ([gate semantics](gate.md#verification-gate)).

## Tier meanings and corpus inventory

| `gate_tier` | Meaning | Train | Test |
| --- | --- | ---: | ---: |
| `gold` | Speech with no rejection reason, best-teacher WER $\leq 0.25$, and critical-slot recall $\geq 0.5$. | 19,278 | 416 |
| `silver` | Speech with no rejection reason and best-teacher WER $\leq 0.50$. There is no additional recall floor after the gold check, but a critical substitution still rejects the row. | 29,366 | 610 |
| `adversarial` | Hard but label-supported speech: $0.50 < \mathrm{WER} \leq 0.90$, critical-slot recall $\geq 0.5$, and no rejection reason. | 42,897 | 868 |
| `rejected` | Any audio failure, repeated segment, critical substitution, WER $>0.90$, or hard clip below the critical-recall floor. | 61,121 | 1,220 |
| `noise` | Empty-reference noise control. It has no WER; with consensus enabled, all teachers must emit more than 2 normalized words for it to fail as speech-bearing. | 4,800 | 0 |

Threshold definitions are from [gate.md](gate.md#tiers-and-thresholds); exact
split counts are from [the P4b export report](../lab/reports/prod-p4b-tiers.md#results).

## What enters training

- Use `gold` and `silver` as the default synthetic speech pool. Keep the tiers
  visible in training logs so results can be broken out by label confidence.
- Start with no `adversarial` rows. If a controlled hard-example experiment
  earns their use, keep them at **no more than 5% of the full training mix**, not
  5% on top of it. These rows sit near the teacher failure boundary; overweighting
  them increases label-noise exposure and makes hardness-driven objectives easier
  to reward-hack ([gate cap](gate.md#tiers-and-thresholds),
  [label-corruption and reward-hacking risks](research-findings.md#8-risks--mitigations)).
- Never train on `rejected`. Retain those rows for generator/channel diagnostics,
  rejection-reason analysis, and yield regressions; the gate intentionally
  rejects rather than relabels them ([gate contract](gate.md#verification-gate)).
- Use `noise` only as an explicit empty-target anti-hallucination stratum. It is
  train-only and must be scored with false-speech/hallucination metrics, not
  included in speech WER ([export semantics](../scripts/export_corpus_csv.py)).

## Starting mixture and ASR registration

Start at **75% real and 25% selected synthetic**, inside the supported operating
range of **70--80% real and 20--30% synthetic**. Keep a replay buffer of useful
real examples and finish each cycle with real-only calibration before selecting
the checkpoint ([mixture and pipeline guidance](research-findings.md#46-l2--asr-training-platform)).
The staged research recipe is real SFT, synthetic adaptation, then ASR GRPO;
Whisper's standard Hugging Face PEFT LoRA path is not compatible with its
log-mel encoder, so use the sibling trainer's supported fine-tuning path rather
than assuming generic LoRA works ([pipeline note](research-findings.md#46-l2--asr-training-platform)).

Create trainer-facing CSVs from the audit exports. This baseline excludes
adversarial and rejected speech; keep noise separate until its sampling rate is
chosen against a real-silence hallucination metric.

```python
from pathlib import Path

import pandas as pd

corpus = Path("data/corpus/V1.0.0")
train = pd.read_csv(corpus / "corpus_train.csv", keep_default_na=False)

speech_pool = train[train["gate_tier"].isin(["gold", "silver"])].copy()
noise_pool = train[train["gate_tier"].eq("noise")].copy()

assert not speech_pool["gate_tier"].isin(["adversarial", "rejected"]).any()
speech_pool[["audio", "text", "suspect"]].to_csv(
    "synthetic_corpus_train.csv", index=False
)
```

Apply the same `gold`/`silver` filter to `corpus_test.csv` for the trainer-facing
synthetic test CSV. The sibling ASR trainer concatenates real and synthetic rows;
it has no ratio knob. For a real fraction $r$ and $N_{real}$ real rows, sample
$N_{real}(1/r-1)$ synthetic rows before export. At the **75%** starting point,
that is one synthetic row per three real rows
([ASR registration and mixing](../.github/skills/asr-feedback-loop/SKILL.md#2-register-it-in-optimizationconfigyaml)).

Place or link the filtered CSVs and audio under
`reference-data-for-v1-run/asr/resources/synthetic_clips/V1.0.0/`, then set all
three fields in the sibling repository's `optimization/config.yaml`:

```yaml
data:
  synthetic:
    train_csv: resources/synthetic_clips/V1.0.0/synthetic_corpus_train.csv
    test_csv: resources/synthetic_clips/V1.0.0/synthetic_corpus_test.csv
    clips_dir: resources/synthetic_clips/V1.0.0/audio
```

The ASR loader requires all three paths or none. It resolves audio by basename,
so copied files must have unique basenames
([ASR CSV contract](../.github/skills/asr-feedback-loop/SKILL.md#1-export-synthetic-audio-for-the-asr-trainer)).

## Test split and locked evaluation

`corpus_test.csv` is a synthetic held-out split grouped by transcript and
stratified by airport. It is useful for synthetic regression checks and
train/test leakage control; it is **not** evidence of real-audio generalization
and cannot replace real validation or final evaluation
([export split semantics](../scripts/export_corpus_csv.py)).

Use real development/model-selection data during iteration. Read
`data/real/kixd/kixd_locked_day.csv` exactly once, on the final trained model,
as the last pre-ship check; it is the held-out day `20250808` with **337 rows**
([locked-day discipline](runbook-v1-3080.md#5-frozen-config-values-decided-overnight-2026-09-01)).

## V1 production provenance and caveats

- The waveform already includes the low-pass step at **3.8 kHz** and the
  subsequent **1 kHz-region** correction (`peaking_eq` centered at **1,100 Hz**, 
  **+7.0 dB**, $Q=1.7$). Do not apply those corrections again
  ([production addendum](results.md#fidelity-history)).
- Final channel fidelity passed D3'' against the real cohort: in-band maximum
  LTAS gap **1.43 dB** and matched WavLM KID
  **0.004331 +/- 0.000938**
  ([production addendum](results.md#fidelity-history)).
- Export QA found **157,462 train rows**, **3,114 test rows**, no null tiers,
  exact tier reconciliation, and matching manifest hashes; the full suite was
  **787 passed, 3 skipped**
  ([P4b QA evidence](../lab/reports/prod-p4b-tiers.md#director-summary)).

These are corpus-construction and development-fidelity results, not evidence
that V1 improves a downstream ASR. Phrase trainer outcomes as development
evidence until the final model receives its single locked-day evaluation
([claim discipline](../.github/skills/lab-protocol/SKILL.md#9-claim-discipline-wording)).