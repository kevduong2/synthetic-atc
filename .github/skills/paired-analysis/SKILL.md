---
name: paired-analysis
description: Computing and reporting the statistics the atc-gan lab accepts. Paired-by-seed deltas from dev_rows.jsonl with paired_report.py and rescore_bounded.py, direction counts and paired t, the 2xSE power gate, matched KID via make_matched_sets.py plus atcgen.eval.embed_dist, LTAS via ltas_check.py, and the table formats for reports and docs/results.md. Use after any power-check, arm or fidelity run, and when auditing a report.
---

# Paired analysis

## 1. Where the raw numbers live

A power-check or RL run under `runs/<run>/` has `harness/baseline/*_rows.jsonl`
(zero-shot per-row errors on the dev slice) and `trials/<arm>_s<seed>/dev_rows.jsonl`
(per-row `errors`, `ref_words`, `capped` after fine-tuning). Everything below
is recomputed from those rows; `summary.json` is never quoted
(`runs/power_check_kixd/summary.json` mixes two metrics).

Reward = bounded zero-shot WER − bounded post-fine-tune WER; positive means the
arm's synthetic data helped. Bounded: each row's errors are capped at its
reference length.

## 2. Commands

```
uv run python scripts/analysis/paired_report.py runs/<run>        # the table you report
uv run python scripts/analysis/rescore_bounded.py runs/<run>      # unbounded vs bounded per cell (metric-drift check only)
uv run python scripts/analysis/analyze_dev_rows.py runs/<run>     # per-source / capture-hour breakdown, per-cluster direction
```

`paired_report.py` prints, per arm, the reward at each shared seed and then
the paired comparison against `base`: n, mean, sd, se, t, df, direction count
and the per-seed diffs. In that table **positive = base better = the arm
hurts**. Copy that table into the report unchanged.

## 3. Decision arithmetic (what the rule in the spec means)

For seeds i = 1..n shared by arm and base: d_i = R_base(i) − R_arm(i).
mean = Σd_i/n; SE = sd(d)/√n; t = mean/SE with df = n−1; direction = number of
d_i with the sign of the mean. A spec's "visible" or "wins" rule is:

- |mean| ≥ 2 × SE **and** direction ≥ 3/4 (or n/n when n = 3);
- for a channel arm additionally: matched KID of its render not worse than base
  by more than 1 SE (a WER win that costs fidelity is a note, not a change).

Two seeds can only gate, never claim. Quote the 95% CI (mean ± t_{0.975,df}·SE)
when n ≥ 4. Check additivity when an effect was split: the halves' paired
means should sum to the whole within one SE.

## 4. Matched KID (the only KID the lab quotes)

```
uv run python scripts/analysis/make_matched_sets.py --out runs/<id>/kidsets --real-dir runs/calib_kixd/clips --syn base=runs/<id>/render_base/wavs --syn arm=runs/<id>/render_arm/wavs
```

It draws a fixed 1,000-clip reference subset (seed 0; reproduces the full-set
KID), writes raw and matched copies of every set (energy-trim to the active
region at −35 dB from peak, RMS-normalize to −26 dB), prints duration,
zero-sample fraction and RMS per set, and then prints the `embed_dist`
commands to run:

```
uv run python -m atcgen.eval.embed_dist runs/<id>/kidsets/arm_matched runs/<id>/kidsets/ref_matched --device cuda --out runs/<id>/kidsets/kid_arm_matched.json
```

Quote `<tag>_matched` vs `ref_matched` only; report the subset spread from the
JSON as the SE. Same clip count on both synthetic sides. Raw-vs-raw numbers
may appear once, labelled, to show the padding/level confound.

## 5. LTAS

```
uv run python scripts/analysis/ltas_check.py runs/<id>/render_base/wavs runs/<id>/render_arm/wavs --label base --label arm --json runs/<id>/ltas.json
```

Reports dB relative to curve peak at the measurement frequencies of the real
KIXD reference (100/200/400/1k/2k/3k/4k Hz; real values in `docs/results.md`).
The in-band 1–3 kHz gap has a ~2 dB measurement floor. The known open gaps:
+13–24 dB at 4 kHz and +13.5 dB at 100 Hz for mode 2.

## 6. Table formats

Report and `docs/results.md` addendum, paired table:

```
| Arm | n | Paired delta vs base | SE | t (df) | Direction | Per-seed diffs |
|---|---:|---:|---:|---:|---:|---|
| aug_off | 4 | +0.0283 | 0.0098 | 2.90 (3) | 4/4 hurt | +0.021, +0.031, +0.029, +0.032 |
```

Fidelity table:

```
| Render | Matched KID | ± | LTAS 100 Hz | 1 kHz | 3 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
```

Decision line: `D1 (rule: |mean| ≥ 2 SE and 3/3 agree): mean=+0.002, SE=0.009, 1/3 → FAIL (reward blind to channel)`.

## 7. Pitfalls

- The runner's printed "separation" is unpaired; do not quote it.
- Different seed sets across arms break pairing; only shared seeds count.
- A trial run with a different `--dev-indices` or dev file is a different
  experiment; check the dev-composition line in each cell's log.
- KID on un-matched audio, on fewer than ~150 clips per side, or against a
  reference that overlaps calibration clips, is not reportable.
- Mixed metrics in one `--out` (bounded vs unbounded): rescore from rows.
