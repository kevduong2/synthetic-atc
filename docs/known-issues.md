# Known issues

Operational and metrological caveats that are not visible from any single
module's docstring. Dated entries are field measurements from the sessions of
2026-08-24/25; undated statements are derivable from the committed code.
Pre-release: fix these by changing the code freely (`AGENTS.md`), then delete
the entry.

## 1. Kokoro TTS is nondeterministic on MPS (~−19 dB at identical seeds)

Two renders of the same text/voice/seed on the MPS backend differ by roughly
**−19 dB** of rendering noise relative to the signal (measured 2026-08-25
during matrix field-testing).

Why it matters: the outer-loop reward harness (`atcgen/rl/reward.py`) is built
on common random numbers — a pinned text pool and a fixed harness override of
`config.seed` — precisely so that two candidate configs differ only in their
knobs, not in draw luck. MPS render noise re-introduces draw luck *downstream
of the seed*: identical configs no longer produce identical audio, so identical
configs no longer produce identical rewards. Consequences:

- Reward deltas smaller than the resulting WER noise floor are not
  attributable to config differences. Before trusting a small delta, estimate
  the floor by evaluating the same config twice.
- "Same seed" does not mean "same dataset" for any dataset built on MPS.
  Reproducibility claims should hash text + drawn channel parameters (the
  manifest lineage), never waveforms.

Workarounds: average over more samples per trial, or render on CPU (slower;
CPU determinism has not been verified either — measure before relying on it).

## 2. MPS memory pressure can kernel-panic the machine

Two kernel panics on 2026-08-25 while running the validation matrix on the
36 GB machine. MPS **wires** (unpageable) memory; more than two concurrent
whisper-tiny trainers push wired memory past physical RAM and the machine
panics rather than swaps.

Mitigations:

- Run arms sequentially: `scripts/run_matrix.py --arm-workers 1
  --eval-workers 1` (the current `runs/matrix_v1` invocation).
- Cap the MPS allocator:
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.30`.
- Treat a live matrix/bandit run as exclusive: don't load models or start
  other memory-heavy work on the machine until it finishes.

## 3. Fidelity metrics: probe saturated, calibration set leaks into the reference

Two related caveats on the Tier 2 numbers from `atcgen/eval/harness.py`:

**The WavLM probe is saturated at the current reference size.** With n≈100
real clips, the real-vs-synthetic probe reads 0.98–1.00 balanced accuracy for
*everything* — including clean ungated TTS — at every WavLM layer, against a
~0.54 real-vs-real null floor. 768-dim features against ~100 clips per class
is a regime where a classifier separates almost anything (see the null-floor
discussion in `atcgen/eval/probe.py`), so the probe cannot rank channel work
until the ~1k labeled real set arrives. **KID (plus Tier 1
LTAS L1) are the iteration metrics** until then; the probe stays in reports as
a trend, not a gate.

**The harness reference set is the channel's own training data.** The
`--ref` directory is `data/real/calibration/` — the same 100 clips
`atcgen/channel/learned/channel_fit.py` fits Mode 2 per-clip presets from, and
the source of the per-station noise beds (`runs/calib_v1/noise` is harvested
from `runs/calib_v1/clips`, which is built from the same captures). Every
calibrated-mode fidelity number is therefore **in-sample for the channel
parameters**: it measures fit, not generalization. In particular the headline
Mode 2 vs Mode 1 comparison (LTAS L1 1.22 dB vs 6.84; WavLM KID 0.00257 vs
0.00295) is tilted toward Mode 2 by construction. Treat Mode 2 numbers as
upper bounds and hold out a disjoint real reference before a mode comparison
becomes load-bearing.

Scope: this caveat covers the *channel-fidelity* tiers only. The downstream
metrics — student WER and the entity panel — are evaluated on
`jacktol/atc-dataset` splits, which are disjoint from the calibration
captures and unaffected.

## 4. Smaller items

- **GRPO grad clipping saturates.** Raw grad norms run 13–28 against a clip of
  1.0, so the clip is the de-facto learning rate (`training/grpo.py`). Pick
  clip/lr deliberately before reading much into arm A4.
- **`runs/rl_v1` best_config is stale.** It was searched against the
  pre-bandpass-re-application chain physics; don't reuse it as a seed or a
  baseline.
- **Hallucination penalty is untested on silence.** The GRPO pool has no
  noise-only rows yet, so the anti-hallucination penalty has nothing to fire
  on during training.
- **Frame-energy SNR estimator saturates** once squelch gating is realistic
  (noted in `configs/mode1_matched.yaml`) — measured SNR stops tracking the
  configured SNR band.
- **Pitch augment costs 2.6× on KID** (resampling artifacts land in the WavLM
  low layers). Profiles leave it unchanged; on/off is an open owner decision.
- **Entity parser recall on real transcripts is ~72.7%**, so entity metrics
  computed against *parsed real references* undercount; synthetic references
  are exact (grammar-emitted) and unaffected.
- **Wide profile bandpass edge.** After bandpass re-application, the wide
  profile's `bandpass.high` low edge of 2300 Hz is harsher than intended;
  consider raising toward 2800.
- **jacktol split leakage.** The corpus is utterance-segmented, so
  speaker/callsign overlap across its train/test is possible. Accepted and
  documented for the PoC (`docs/plans/research-integration.md`); prohibited in
  production split design.
