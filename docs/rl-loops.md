# RL loops

The project has three separate RL control points. Their rewards answer
different questions, so they are not collapsed into one adversarial objective.
The separation prevents student hardness from becoming a generator incentive
to corrupt labels, and prevents a single metric from being hacked while other
fidelity or safety metrics regress. The design context is in
[research-findings.md](research-findings.md).
The illustrated system overview is [systems-manual.html](systems-manual.html).

## The three loops

- **L1, generator alignment:** optimize a flow-matching speech generator with
  generator-side intelligibility, entity, realism, and speaker/accent rewards
  plus an anchor. L1 is deferred until GPU hardware and license-cleared
  flow-matching TTS weights are available. The current generator is Kokoro TTS
  followed by the parameterized DSP channel twin, which is the L1 stand-in.
  See [generation.md](generation.md).
- **L2, student training:** train Whisper on real and gated synthetic audio,
  then apply student GRPO. Its reward is ATC-normalized WER plus repetition,
  length, and hallucination controls. The implementation is documented in
  [training-and-eval.md](training-and-eval.md).
- **L3, recipe selection:** choose which named synthetic recipe to generate
  next. Its proxy reward is a teacher-bounded student-hardness window; its
  non-proxy check is a selected-versus-uniform counterfactual on real
  `reward_val` audio.

The outer config loop is a separate search over generator knobs. It changes a
configuration, then lets the fixed student reward harness measure downstream
utility; it does not turn student error into a generator loss.

## Outer config loop

`atcgen.rl.loop.run_loop()` supplies the ask/evaluate/tell and persistence
logic. `atcgen.rl.space.SearchSpace` maps a unit-cube vector to a deep copy of
the YAML-style generator config. `default_atc_space()` declares 19 knobs over
noise, bandpass, receiver artifacts, talker controls, and batch composition.
The hand-tuned profile is inverted into the cube by `default_vector()` so it
can be evaluated as the anchor.

`atcgen.rl.policy` provides three optimizers:

- `CrossEntropyMethod` (the default in `scripts/rl_loop.py`) refits a diagonal
  Gaussian to the top elite vectors each batch.
- `RandomSearch` uniformly samples the same cube and is the control.
- `ReinforceGaussian` is an available score-function alternative.

The CEM/random comparison is deliberate. CEM uses reward ranking, while the
random optimizer supplies a no-learning control with the same candidate-space
interface. Optimizer state includes its NumPy generator state so a resumed
run continues its proposal stream.

`atcgen.rl.reward.TrueRewardHarness` implements the expensive reward. It
creates one seeded grammar text pool and reuses it for every candidate, forces
the generator config seed to `GEN_SEED = 20260824`, renders a fresh synthetic
batch, fine-tunes a fresh `openai/whisper-tiny.en` copy with
`finetune_lite`, and evaluates on one cached real dev slice. Its
`RewardResult.reward` is baseline normalized WER minus post-fine-tune
normalized WER; positive means the candidate helped. The shared text pool,
generator seed, fine-tune seed, and cached baseline keep the comparison tied
to the candidate knobs.

`atcgen.rl.types` is the contract between the optimizer and harness. A
`RewardResult` carries reward, post-training WER, baseline WER, optional
hallucination rate, a `proxy` flag, and extra metrics. `atcgen.rl.stats` adds
paired WER bootstrap over aligned utterances. Its
`paired_bootstrap(refs, hyps_a, hyps_b)` docstring defines

```text
delta = WER_a - WER_b
```

Therefore a positive delta means side **b** has lower WER and wins; a negative
delta means side **a** wins. The result contains `delta`, a percentile
confidence interval, and a two-sided p-value.

### Persistence and failure handling

`run_loop()` appends each completed trial to `trials.jsonl` and checkpoints
loop bookkeeping plus optimizer state in `optimizer_state.json`. `best.json`
and `best_config.yaml` are updated only by a successful non-proxy reward. A
restart reads the existing trial log and checkpoint, preserves trial
numbering, and continues from the next un-evaluated search work. The hand-tuned
configuration is trial 0 when `seed_default_first` is enabled, which is the
default in the CLI. The anchor is logged as a reference and is not passed to
the optimizer's `tell()` update.

If a candidate raises during generation, fine-tuning, or evaluation, the loop
appends an error row with its vector and exception text. It drops that vector
from the batch before calling `optimizer.tell()`; it does not insert a
sentinel reward that would teach the search policy that the candidate region
is intrinsically bad.

The current `runs/rl_v1/best_config.yaml` is stale: that search predates the
channel-physics re-fit and its bandpass re-application. Re-run the outer loop
before quoting it as a current best configuration. The stale-run caveat is
also recorded in [known-issues.md](known-issues.md) and the result index is in
[results.md](results.md).

## L3 recipe bandit

`atcgen.rl.bandit.RecipeBandit` uses Thompson sampling over the named recipes
in `atcgen.rl.recipes`. Each pull renders one batch, obtains normalized
per-sample WER from a frozen teacher and the current student, routes the rows,
updates one Beta posterior, and appends the pull and state artifacts.

The 12 recipe buckets are:

- `eu_routine`: European phraseology anchor.
- `eu_fast_speech`: European phraseology with speed `1.30--1.55`.
- `eu_readback_errors`: readback-error/correction exchanges.
- `eu_confusable_callsigns`: callsigns differing by one digit.
- `us_routine`: FAA tower/approach phraseology.
- `mixed_phonetic_respell`: phonetic respelling with both regions.
- `low_snr`: injected SNR `0--14` dB.
- `high_snr_clean`: easy SNR `15--27` dB with receiver artifacts backed off.
- `dense_numerics`: numeric entity kinds such as headings, levels, speeds,
  squawks, and frequencies.
- `noise_heavy_channel`: crackle, dropouts, co-channel, hum, fading, and
  heterodyne artifacts.
- `narrowband_codec`: low-bitrate codec, extra resampling, and tightened
  bandpass.
- `rare_and_emergency`: rare vocabulary and emergency categories.

`atcgen.rl.recipes.Recipe.apply()` deep-copies the base config and applies
named dotted overrides. A recipe can also select a grammar region, category,
or utterance-kind filter. `FilteredTextSource` uses rejection sampling and
returns the last draw after its maximum tries, recording accepted and rejected
draws rather than crashing a starving bucket.

### The hardness window

`HardnessWindow` uses strict comparisons:

```text
WER_teacher < tau1
tau2 < WER_student < tau3
```

The default thresholds are `tau1=0.8`, `tau2=0.4`, and `tau3=1.2`.

`tau1` is a structural teacher-trust bound. If the frozen teacher cannot read
the rendered audio under this bound, the sample is dropped. Student hardness
never reaches a generator objective: it only decides which teacher-trusted
rows are targeted for L3 selection. This prevents a generator from learning
that garbling a number is a way to earn reward.

Rows route as follows:

- `selected`: teacher-trusted and inside the student hardness window.
- `spillover`: teacher-trusted but `too_easy` or `too_hard`; it remains
  available as non-targeted, teacher-trusted data.
- `dropped`: no reference or `teacher_untrusted`; these rows do not enter a
  training buffer.

`tau2` is defined against the current student's error distribution and must be
recalibrated for every refreshed student checkpoint. The recorded comparison
is concrete: `tau2=0.40` for the zero-shot student left only 3.3% in-window
against the A4 student; measuring that student's clean-arm median and changing
to `tau2=0.15` restored 19.4%. Leaving the old threshold in place makes the
window describe the previous checkpoint rather than the current one.

`BetaPosteriors` starts each bucket at Beta(1, 1). A pull contributes its
in-window count as successes and the other rows as failures. Thompson sampling
draws one beta sample per bucket and takes the largest; posterior width supplies
exploration. `AsrPullEngine` keeps the teacher architecturally separate from
the student, shares one log-mel extraction between their decodes, and permits
the caller to replace the student with `set_student()` between rounds.

The bandit writes `pulls.jsonl`, `state.json`, per-pull configs and synthetic
data, `selected/manifest.jsonl`, `spillover/manifest.jsonl`, and scheduled
`counterfactuals.jsonl`. Samples are appended before the pull row; on resume,
buffer rows whose pull never reached `pulls.jsonl` are truncated so a retry
does not double-count them. The posterior and pull counters are restored from
`state.json`, or replayed from `pulls.jsonl` when only the log exists.

### Counterfactual audit

`AsrCounterfactual` is the non-proxy audit. For each round it uses the same
frozen initialization, the same fine-tune steps and batch size, and equal-size
selected and uniform arms. The selected arm samples the targeted buffer. The
uniform arm freshly generates an equal share from every recipe and applies the
same teacher trust gate; it is not built from spillover because spillover is
already shaped by the bandit's choices. Both fine-tuned models are scored on
the same first `eval_n` rows of the `reward_val` split.

The returned field is `delta_wer_selected_vs_uniform = wer_uniform -
wer_selected`. Positive means selection beat uniform; negative means uniform
had lower WER. The normal schedule is every eight pulls and once at the end;
the CLI can disable it with `--counterfactual-every 0`.

The honest PoC result is negative. The naive counterfactual was about -2.5 WER
points; the replay-corrected run was -0.6 points. The exact replay artifact
records `wer_selected=0.2202247`, `wer_uniform=0.2144819`, and
`delta_selected_vs_uniform=-0.0057428` in
`runs/bandit_v2/counterfactual_replay.json`.
The selected policy therefore did not clear the counterfactual win bar.

The diagnostic was still useful. On the current synthetic distribution, the A4
student is saturated: its median synthetic WER is 0.096 versus 0.561 for the
frozen teacher, so student hardness is no longer a reliable proxy for marginal
training value. The posterior did localize `us_routine` as the weak slice for
the EU-trained student. See [results.md](results.md) for the result summary.

## Command-line entry points

Full flag tables belong in [cli-reference.md](cli-reference.md). The three
current entry points have these roles:

- `scripts/rl_loop.py` runs the outer config search. The short form is
  `uv run python scripts/rl_loop.py --out runs/rl_v2`. It defaults to
  `configs/mode1_matched.yaml`, CEM, four iterations of four candidates,
  optimizer seed 0, trial-0 default seeding, 200 synthetic clips per trial,
  300 fine-tune steps, dev indices `0:200`, a 400-row text pool, and automatic
  device selection. Use `--optimizer random` for the control,
  `--no-seed-default` to omit the anchor, or `--no-resume` to restart.
- `scripts/rl_recipe_bandit.py` runs L3. The short form is
  `uv run python scripts/rl_recipe_bandit.py --out runs/bandit_v2`.
  Defaults are 30 pulls, 60 clips per pull, zero-shot tiny as student,
  `openai/whisper-base.en` as teacher, thresholds `0.8/0.4/1.2`,
  counterfactuals every 8 pulls, 300 fine-tune steps per counterfactual,
  150 clips per arm, 400 `reward_val` evaluation rows, and automatic device
  selection. `--student` changes the checkpoint, so recalibrate `--tau2`.
- `scripts/rl_verify.py` runs the blind zero-shot/base/best A/B verification.
  It reads `best_config.yaml` from `--run`, defaults to test indices `0:500`,
  renders 600 clips, fine-tunes 600 steps with batch 8 and learning rate
  `1e-5`, uses a fresh 1,200-row text pool with seed 4321, and writes
  pairwise bootstrap results to `<run>/verify/verify_report.json`. Use
  `--save-models` to save the two fine-tuned checkpoints.

All three scripts write their own run artifacts and can be rerun against their
run directory. Keep the `reward_val`, `model_select`, and locked-test policies
from [training-and-eval.md](training-and-eval.md) when choosing slices.
