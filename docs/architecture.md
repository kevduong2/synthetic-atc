# Architecture

atc-gan is a synthetic-data loop for training ASR on noisy air-traffic-control
radio. A scenario grammar creates the spoken transmission and its structured
ground truth. Kokoro renders the speech. A physically ordered channel twin
turns the clean render into a radio transmission. A two-teacher gate checks
whether the label survived before the row can enter a training pool. The
student is trained on a controlled real/synthetic mixture, evaluated on
disjoint real splits, and used to steer the next generation recipe.

This page is the plain-Markdown companion to the [illustrated systems
manual](systems-manual.html). The code is authoritative for thresholds,
schemas, and CLI details. The [research proposal](research-findings.md) gives
the evidence and design decisions behind the system; the [PoC integration
plan](plans/research-integration.md) maps those decisions onto this repo.

## End-to-end circuit

The scenario layer is label-first. `atcgen/text/` builds phraseology for EU
and US scenarios from slots such as callsign, runway, heading, altitude,
flight level, frequency, speed, squawk, waypoint, and ATIS. Each slot creates
an `Entity` while the utterance is assembled. The resulting row carries the
spoken form, the transcript used as the ASR label, a display form, and the
canonical entity value. Critical numeric and identity entities are marked for
separate verification. The grammar validator checks the emitted entities and
their spoken forms; it does not make generated audio the source of truth.

The dataset builder sends the spoken form to Kokoro TTS. The current speech
stage uses a curated Kokoro voice pool and a sampled delivery rate. Voice and
rate augmentation may run before the channel. The TTS output is normally at
24 kHz; the channel backend returns the radio waveform at the configured
16 kHz output rate. The planned voice-conversion/accent branch is not part of
the current path.

The channel is a twin of the transmission path, not one undifferentiated
effect. Talker-side coloration and push-to-talk truncation happen once. Path
effects are applied per hop, with a second relay hop for selected pilot rows.
Receiver-side co-channel mixing, AGC, squelch, clicks, and delivery effects
then run in receiver order. Bandpass is re-applied at receiver boundaries
where the physical filter would act again. Procedural profiles and calibrated
per-clip presets share the backend interface. The applied draws are recorded
in the channel record and copied into row lineage.

The gate receives the degraded waveform that the student would train on. It
does not gate the clean TTS render. Frozen `whisper-base.en` and
`wav2vec2-base-960h` transcribe the clip independently. The first is a
language-modelled encoder-decoder; the second is a CTC teacher from a
different architecture family. The gate combines word error, audio checks,
and entity-level evidence. It rejects a row whose label cannot be supported;
it never rewrites the label to match a teacher hypothesis.

The gate writes one of four tiers: `gold`, `silver`, `adversarial`, or
`rejected`. Gold is the near-clean evidence tier. Silver permits a higher
teacher WER while retaining broadly correct content. Adversarial rows are
hard clips whose critical entities remain provable; `select_tiers` limits
them to 5% of a requested training mix. Rejected rows stay in the gated
manifest with teacher hypotheses, entity verdicts, reasons, audio checks, and
lineage, but never train or reward a model.

The selected tiers form the synthetic side of the student buffer. The current
student is `whisper-tiny.en`. The training recipe is supervised fine-tuning,
then a 75/25 real-to-gated-synthetic mixture, then student GRPO for the
`mix_grpo` arm. The GRPO stage samples transcript groups and rewards lower
ATC-normalized WER while penalizing repetition, length deviation, and
hallucination. A KL term anchors the policy to the SFT checkpoint. Noise-only
rows provide explicit cases for the hallucination penalty.

Evaluation uses real audio, not synthetic audio, when the question is whether
synthetic data helps the student. The PoC split registry separates real-train,
reward-validation, model-selection, and locked-test roles. The current
jacktol protocol uses train slices for real training and development, while
the locked test is `test[500:2500]`; `test[0:500]` was spent by an earlier
verification run. Each arm reads the locked test once. It is not used to
choose a recipe, tune a checkpoint, or feed a generation objective.

The evaluation report includes aggregate WER, but release decisions use the
entity panel: callsign accuracy, per-entity slot F1, critical-number
substitution rate, hallucination measures, and slices. A synthetic pool is
therefore useful only when it improves the student on disjoint real replay
without weakening the safety-entity panel.

```text
  born-labeled scenario
  grammar + entities + display
             |
             v
       Kokoro TTS
       voice/rate augment
             |
             v
  physically ordered channel twin
  talker -> path/hops -> receiver -> 16 kHz
             |
             v
  two-teacher verification gate
  whisper-base.en + wav2vec2-base-960h
             |
       +-----+-------------------+
       |                         |
       v                         v
  gold / silver /          rejected + reasons
  adversarial <= 5%       (manifest only; no training)
       |
       v
  75% real + 25% gated synthetic
       |
       v
  student SFT -> student GRPO
       |
       v
  disjoint real splits -> entity panel -> generation knobs
```

The feedback arrow changes generation knobs, not labels, teacher weights, or
student evaluation references. L3 chooses recipe dimensions such as scenario
class, voice, rate, SNR band, channel condition, and difficulty. The outer
config loop searches channel and generation settings with CEM and keeps a
random-search control. Both paths feed candidates back through the same
generation and verification stages.

```text
                 +----------------------------------------------+
                 |              locked_test                      |
                 |       read once per arm; outside loops        |
                 +----------------------+-----------------------+
                                        |
  real-train + gated synth              | report only
             |                          v
             v                  +---------------+
       student training ------> | entity panel  |
             ^                  +-------+-------+
             |                          |
             |                          v
             |                    L3 steering
             |              bandit + outer CEM
             |                          |
             +------ generation knobs-+

  L1: generator GRPO       deferred; GPU + cleared weights
  L2: student GRPO         live; validated on real development slices
  L3: recipe bandit         live; teacher-bounded and counterfactual-audited
```

## Three separated RL loops

The proposal places reinforcement learning at three different interfaces.
They remain separate because they optimize different objects, consume
different evidence, and have different failure modes. A single adversarial
loop would let a generator exploit a student weakness, corrupt the label, and
then receive credit for the resulting student failure.

| Loop | Optimizes | Status |
| --- | --- | --- |
| L1 | Flow-matching speech generator | Deferred to GPU work; Kokoro plus DSP is the stand-in. |
| L2 | ASR student | Live and validated; WER, anti-degeneracy penalties, and KL to SFT. |
| L3 | Recipe selection | Live; Thompson sampling, teacher bound, counterfactual audit. |

L3's hardness window is a data-selection rule. A candidate must remain
teacher-trustworthy, then may be useful because the current student finds it
hard. Student hardness never becomes a generator reward. The outer CEM loop
is also a config search over generation/channel knobs, not a replacement for
L2 or L3 and not a path around the gate.

## Invariants

These are architectural constraints, not tuning preferences.

1. The judge/teacher ASR is not the student ASR. Teachers are frozen, and one
   teacher is architecturally distinct from the Whisper student.
2. The gate runs before any sample can train or reward anything. Failed rows
   are rejected, never relabeled.
3. The adversarial tier is capped at 5% of any training mix.
4. Student-hardness signals never reach a generator objective. Hardness is
   confined to L3 and bounded by teacher fidelity.
5. Real splits are disjoint by role. The locked test is outside every loop and
   is read once per arm.
6. Entity-panel metrics gate releases, not aggregate WER alone.
7. LiveATC audio is not in any training path, and CC-BY-NC weights are not
   used as product checkpoints.
8. Every row carries lineage: config hash, seed, profile, text source, code
   revision when available, and the generation/channel record. Gated rows
   also carry teacher verdicts and reasons.
9. When the question is student benefit, synthetic data is evaluated mixed
   with real replay. Synthetic-only results do not answer that question.

The split rule does not make the current PoC corpus a production-grade split:
jacktol is utterance-segmented, so speaker or callsign overlap across its
train and test portions is possible. That limitation is recorded below and
in `docs/known-issues.md`.

## Module map

| Path | Responsibility |
| --- | --- |
| `atcgen/text/` + `atcgen/entities.py` | Grammar, born labels, entity parsing, and scoring. |
| `atcgen/tts/` | Kokoro, voice/rate variation, and the pre-channel augmentation seam. |
| `atcgen/channel/` | DSP, calibrated presets, CUT residual, noise, ordering, and loudness. |
| `atcgen/gate/` | Teachers, audio/entity checks, tiers, retiering, and the cap. |
| `atcgen/dataset/` | Builds, manifests, lineage, real corpus, splits, and noise harvest. |
| `atcgen/eval/` | Entity panel, WER/S/D/I, channel metrics, diagnostics, and harnesses. |
| `atcgen/rl/` | L3 bandit, hardness window, CEM loop, reward harness, and fine-tuning. |
| `training/` | Student SFT, 75/25 pools, GRPO, normalization, and evaluation. |
| `scripts/` | Generation, gating, matrix, CEM, bandit, harvest, and eval CLIs. |
| `configs/` | Generation/channel profiles, calibrated settings, and ablations. |

## Current deviations and deferred items

The following are current PoC boundaries, not alternate paths that the system
silently uses.

- **VC/accent branch.** Voice or accent conversion of real clips is deferred.
  The current speech branch is Kokoro plus the configured augmentation seam.
- **FM-TTS and L1.** Flow-matching TTS and generator GRPO are deferred to GPU
  work. Public F5-TTS weights are CC-BY-NC, so they cannot be used as product
  weights; a cleared retraining or separately licensed weight path is needed.
- **Stronger gate teachers.** The current two-teacher gate is useful but weak
  on callsigns. Callsign verification is currently teacher-bounded at about
  0.16 recall on degraded audio. A stronger judge panel is deferred; the
  parser is not a substitute for teacher evidence.
- **Stale outer-loop artifacts.** The best config in `runs/rl_v1` predates the
  channel-physics re-fit. It is not a current baseline for the config loop.
- **jacktol leakage caveat.** Its utterance segmentation permits speaker or
  callsign overlap across train and test. This is accepted for the PoC and is
  prohibited for a production split design.

For generation, gating, training/evaluation, RL operation, commands, and the
current evidence snapshot, use the [documentation index](README.md). The
[known issues](known-issues.md) page carries operational and metrological
caveats. Open the [systems manual](systems-manual.html) in a browser for the
illustrated walkthrough of the same architecture.
