# 06 — Outer-loop RL: tuning the generator against downstream ASR reward

Status: implemented (`atcgen/rl/`, `scripts/rl_loop.py`, `scripts/rl_verify.py`);
first search running 2026-08-24. Two research sweeps (RL-for-audio-generation
SOTA; reward-loop design) inform every decision below; citations are arXiv IDs
from those sweeps.

## §1 What the loop is

An outer optimization loop that treats the synthetic-ATC generator's config as
a policy and *downstream ASR improvement* as the reward:

```
candidate config  ──►  generate n_synth clips (fixed text pool, fixed gen seed)
                       ──►  fine-tune whisper-tiny.en (ft_steps, fixed ft seed)
                       ──►  transcribe fixed real ATC dev slice
reward = WER_zero-shot(dev) − WER_after(dev)      (ATC-normalized, paired)
```

The reward is the *actual objective*, not a proxy: WER of the fine-tuned model
on real, human-transcribed ATC audio (jacktol/atc-dataset = ATCO2-1h +
UWB-ATCC). This sidesteps the worst reward-hacking modes documented for
ASR-as-judge-of-TTS loops (easy-to-transcribe synthetic audio scores *worse*
here, because the fine-tuned model then fails on hard real audio).

Novelty: the literature sweep found **no published closed loop tuning
channel-simulation parameters against an ASR-derived reward** (Simu-GAN
2203.15321, TS-RIR 2103.16804, Learning-to-Simulate 1810.02513 bracket the
space without closing it), and no whisper-tiny ATC fine-tune at all.

## §2 Why this design (research-backed decisions)

- **Search the DSP channel knobs, not the generative stage.** Bagat et al.
  (2606.21340, Interspeech 2026): the DSP chain was worth 37% relative WER
  synth-only; learned VC/accent stages added far less. Our 19-knob space
  (`default_atc_space()`) mutates exactly that surface.
- **Rank-based optimizer at tens-of-evals budget.** BO > CMA-ES > policy
  gradient at ≤150 evaluations (2104.10201, 2303.00890, 1803.07055).
  Implemented: CEM (default; uses only reward ranking), REINFORCE-Gaussian,
  and random search (the mandatory control — TrivialAugment 2103.10158 and
  G-Augment 2210.10879 both show searched policies often fail to beat
  cheap/no search). CMA-ES deliberately absent. Upgrade path: TuRBO/SAASBO-
  style BO + ASHA multi-fidelity if the search continues on the 5080.
- **Paired everything (common random numbers).** All candidates share one
  text pool (`text_pool.jsonl`, seeded), one generator seed (`GEN_SEED`), one
  fine-tune seed, one dev slice. Bisani & Ney (ICASSP 2004): paired WER deltas
  have ~3× smaller CIs than independent measurement.
- **Reward as delta on a fixed dev set** — variance reduction plus an honest
  anchor: trial 0 is always the hand-tuned base profile evaluated by the same
  harness (`seed_default_first`).
- **Hard bounds are the anti-collapse mechanism.** Extremal-Goodhart drift
  toward clean/easy audio (1910.07113 ADR, 2311.01885 DORAEMON) is blocked
  structurally: codec prob ≥ 0.3, squelch prob ≥ 0.3, SNR hi ≤ 35 dB,
  clean_arm ≤ 0.10, speed lo ≤ 1.2. The optimizer cannot leave the
  radio-degraded regime.
- **Frozen normalization.** WER-normalization choice alone moves ATC numbers
  1–3+ absolute points (2211.04054, 2409.02449); the same WhisperATC
  checkpoint reads 13.46% vs 37.62% across two papers from split/normalizer/
  prompting alone. `training/normalize.py` is frozen for the whole loop and
  all reported numbers are `wer.atc_normalized` from `build_report`.
- **Blind test slice.** The search queries only `train[0:400]`. The
  generalization claim comes from `scripts/rl_verify.py` on `test[0:500]`,
  touched once, with paired bootstrap (2000 reps) between arms — Ladder-style
  protection (1502.04585) against adaptive dev overfitting at our ~30-query
  budget.

## §3 Measured operating point (Mac / MPS, 2026-08-24)

| stage | cost |
|---|---|
| generate 200–300 clips | ~2.5–4 min (~1.4 clips/s) |
| whisper-tiny.en fine-tune, 300 steps b=8 | ~3.5 min (0.58 s/step) |
| transcribe 400-utterance dev slice | ~20 s (batched greedy) |
| **one trial** | **~7 min** |

Zero-shot whisper-tiny.en on the dev slice: **119.7% WER** (ATC-normalized) —
it emits conversational English against accented VHF speech. The anchor trial
(hand-tuned `mode1_matched`): fine-tune reward **+0.433** (WER → 76.4%).
Effect sizes between candidates are multiple WER points on a paired
400-utterance slice, i.e. well above the ~1-point/~0.5-noise regime the
research warns about for frontier ATC results; the reward SNR here is healthy.

## §4 Known biases and their mitigations

- **Short-horizon bias** (1803.02021): a k-step inner loop favors
  fast-to-learn (easy, low-diversity) data — a bias, not noise. Mitigations:
  the inner fine-tune *saturates* (train loss ≈ 0 by step 300, ~12 epochs),
  so candidates are separated by what their data teaches, not how fast;
  verification re-tests the winner at 2× steps and 2–3× data on a disjoint
  slice with a *fresh* text-pool seed.
- **Whisper-judges-Whisper** (2607.08256): reward and eval share one model
  family. Weaker here than in TTS-verifier loops (our references are human
  transcripts of real audio), but a cross-family check (wav2vec2/XLS-R WER on
  the same slices) is the cheapest hardening if results move to the 5080.
- **Search-space myopia** (G-Augment lesson): re-tuning knob ranges inside a
  fixed chain may buy little; the larger wins historically come from adding
  *operations*. If the searched config fails to beat the anchor
  significantly, the next move is widening the space (new primitives,
  accent/L2 voices — 2606.21340 found accent diversity the only augmentation
  beating real-only), not a fancier optimizer.
- **The random control can win.** The literature says this is the single most
  likely outcome. `--optimizer random` at equal budget is part of the
  protocol; if it matches CEM, ship the uniform policy and record the
  negative result (unpublished for ATC either way).

## §5 Artifacts and protocol

- `runs/rl_v1/` — first search: CEM, 19 knobs, 5×6 + anchor, n_synth=300,
  ft_steps=300, dev `train[0:400]`, text pool 600 (seed 1234). Resumable:
  rerun the same command after any crash.
- `runs/rl_v1/trials.jsonl` — every evaluation (vector, knob values, reward,
  full Tier-3 report, timings); `best.json` + `best_config.yaml` track the
  incumbent. This is the cached (config → WER) table the NAS literature says
  to keep for honest future comparisons.
- `scripts/rl_verify.py --run runs/rl_v1` — final A/B/C: zero-shot vs
  base-config vs searched-config, blind `test[0:500]`, n_synth=600,
  ft_steps=600, fresh text seed 4321, paired bootstrap on identical
  utterances, Tier-3 slices (callsign accuracy, hallucination rate on
  noise-only — a number never published for VHF audio).
- Tier-3 slice metrics guard against aggregate-WER myopia (2603.05267):
  callsign WER/accuracy and hallucination rate ride along in every trial's
  `metrics.report`.

## §6 Phase 2 (5080, not yet built)

Where actual RL earns its name: a **contextual per-utterance policy** —
a small network mapping utterance features (digit density, callsign
complexity, role, category) to per-sample channel parameters, trained with
group-relative policy gradient (GRPO-style: render the same transcript G≈8
times under sampled parameters, standardize reward within the group).
Precedents: Learning-to-Simulate 1810.02513, Meta-Sim2 2008.09092, DVRL
1909.11671, REINFORCE-SpecAugment 2312.08641 (whose low-resource,
domain-mismatched setting matches ATC). Requires a cheap frozen-ASR reward
(NLL + diversity + KL anchor to the phase-1 config), with periodic
true-reward validation — the honest limitation being that "audio the ASR
finds informative" ≠ "audio that improves the ASR after fine-tuning" until
measured. Also worth testing there: gradient-alignment (LESS 2402.04333 /
Deep AutoAugment 2203.06172 style) as an f2 pre-screen — measure Kendall τ
against ~20 true-reward trials from `trials.jsonl` before trusting it.
