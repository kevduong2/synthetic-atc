---
name: generator-config
description: How atc-gan generator profiles (configs/*.yaml) work and how to make a safe experimental variant. Mode 1 procedural vs mode 2 calibrated, --set dot-path overrides, config.resolved.yaml, calibration artifact paths, the FastCUT residual checkpoint, the frozen V1 values that must not change, the mode2_safe and talker_only search spaces, and how to add a fixed arm to scripts/rl_power_check.py with its test. Use before editing any config or defining an experiment arm.
---

# Generator config

## 1. Anatomy of a profile

```
mode: procedural | calibrated      # mode 1: hand-written channel.chain; mode 2: fitted presets per clip
seed, output (sample_rate, loudness_mode/db), tts (voices, speed), voice_augment (pitch_semitones, tempo, eq_tilt_db)
dataset (noise_only_frac, pilot_double_hop_prob, category_quotas)
channel: {chain: [...steps by primitive name...], clean_arm_prob}        # mode 1 only
calibrated: {calibration: {corpus_dir, presets, noise_bank, snr_jitter_db, cross_station_prob},
             residual: {enabled, checkpoint, apply_prob, alpha, residual_scale_max},
             post_effects: {squelch, dropouts, codec}, expansion: {...}}   # mode 2 only
qc: {enabled, max_retries, asr_roundtrip}
```

Distributions are dicts: `{uniform: [lo, hi]}`, optionally with `prob:` for
augmentations (`{prob: 0.5, uniform: [-2, 2]}`). Every render writes
`config.resolved.yaml` next to `manifest.jsonl`; read it to confirm what ran.

Live profiles: `configs/mode2_fastcut_kixd.yaml` (V1 production shape; KIXD
calibration; residual off until a checkpoint exists),
`configs/mode1_matched_kixd.yaml` (procedural control, required for the
`degraded` arm), `configs/mode2_fastcut.yaml` / `mode1_matched.yaml` (four-station
predecessors), `configs/ablation_pitch_off.yaml`. `runs/` paths inside them
exist only after the calibration chain in the file header has run.

## 2. Overrides instead of forks

```
uv run python scripts/generate_dataset.py --config configs/mode2_fastcut_kixd.yaml --n-samples 150 --out runs/<id>/render --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set calibrated.post_effects.dropouts.prob=0.30 --set dataset.noise_only_frac=0
```

`--set PATH=VALUE` (repeatable) edits any dot-path leaf; prefer it over copying
a profile so two renders share one channel definition. Text sources:
`grammar:region=eu` (built-in grammar), `path.jsonl` (sampled with
replacement), `sequential:path.jsonl` (in order, exactly once; refuses
`--n-samples` beyond file length). Per-clip voice/speed land under the manifest
row's nested `gen` key.

Fork a profile only when the variant will be reused by several commands (e.g.
`configs/mode2_kixd_resid.yaml` with `calibrated.residual.enabled: true` and
`checkpoint: runs/fastcut_kixd_cuda/G_selected.pt`). Copy, edit, keep the
header comment honest about what changed and why. Residual loading is strict:
`enabled: true` with a missing checkpoint fails the run, by design. Use
`G_selected.pt` (the checkpoint the lexicographic rule chose), never `G_ema.pt`.

## 3. Frozen V1 values (do not move inside a mission)

| knob | frozen value | evidence |
|---|---|---|
| `tts.speed` | uniform [1.0, 1.4] | aug_off costs +2.8 WER pts, 4/4 seeds |
| `voice_augment.pitch_semitones.prob` | 0.5 | pitch_off ≈ +0.9 pts, 8/10 seeds |
| `voice_augment.tempo.prob` / `eq_tilt_db.prob` | 0.3 / 0.4 | bundle effect, not separable |
| channel | mode 2 calibrated + FastCUT residual (`source+identity`, scale cap 0.20, `G_selected.pt`) | go/no-go wave |
| channel search | none by WER (reward is blind); LTAS + matched KID govern | power check |

A pre-registered win produces a V1.1 recommendation in the report, not an edit.

## 4. Search spaces and arm bounds

`atcgen/rl/space.py`: `talker_only_space` (speed edges, pitch/tempo probs;
both modes), `mode2_safe_space` (calibrated only; 15 knobs: `snr_jitter_db`
edges lo∈[−9,−1] hi∈[1,9], `post_effects.squelch.prob` [0.3,1], `gated_floor_prob`
[0,0.2], `dropouts.prob` [0,0.4], `codec.prob` [0.3,1], `codec.quality` lo
edge [0.5,0.85], `residual.apply_prob` [0,1], `residual_scale_max` [0.05,0.35],
talker and batch-composition knobs), `default_atc_space` (mode 1 chain knobs).
Fixed arms should stay inside these bounds; `build_arm` does not clamp.

## 4b. Multi-station calibration (production config)

`calibrated.calibration.station_mix` decides which station's channel each clip
gets; unset, it follows preset counts, so a KIXD-heavy preset pool makes every
airport's audio sound like KIXD. The production config sets it explicitly
(uniform over the stations in `presets_stats.json`, names spelled exactly as
there; a name with no presets raises at load). `cross_station_prob` (0.1,
frozen) is a no-op with one station and live with several. Presets come from
`channel_fit --per-station N`; plain `--limit` head-truncates a file grouped by
station. See `docs/runbook-v1-3080.md` §1c.

## 5. Adding a fixed arm to `scripts/rl_power_check.py`

Arms are entries in the `ARMS` dict: a list of `(knob, value)` mutations
applied to the base profile. Knob constructors (from `atcgen.rl.space`):

```python
dist_bound_knob(name, "calibrated.calibration.snr_jitter_db", 0, -9.0, -1.0)   # edge 0/1 of a {uniform: [lo, hi]}
dist_prob_knob(name, "voice_augment.pitch_semitones")                            # the prob: of an augmentation block
scalar_knob(name, "calibrated.post_effects.dropouts.prob", 0.0, 0.40, kind="prob")  # a plain leaf (import it)
chain_param_knob / chain_prob_knob                                               # mode 1 channel.chain steps only
```

Example, the "harder noise floor" arm from a spec:

```python
"snr_lo_down": [(dist_bound_knob("snr_jitter_lo", "calibrated.calibration.snr_jitter_db", 0, -9.0, -1.0), -9.0)],
"post_fx_up":  [(scalar_knob("squelch_prob", "calibrated.post_effects.squelch.prob", 0.3, 1.0, kind="prob"), 0.95),
                (scalar_knob("dropouts_prob", "calibrated.post_effects.dropouts.prob", 0.0, 0.4, kind="prob"), 0.30)],
```

Then: extend the parametrized arm test in `tests/test_rl_power_check.py`
(search for `ARMS`) so the new arm resolves and renders on the intended mode,
keep the mode guard in mind (`degraded` and any `channel.chain` arm require a
procedural base config; the script errors otherwise), run `uv run pytest -q
tests/test_rl_power_check.py`, and only then launch. Additive changes only; the
existing arms keep their names because reports cite them.
