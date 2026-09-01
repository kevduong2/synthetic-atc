# V1 production runbook — RTX 3080 (2026-09-01)

End-to-end: recalibrate on the full multi-airport clip set → train the FastCUT
residual → render the two-view corpus → gate → export `corpus_train.csv` /
`corpus_test.csv`. Written during the overnight experiment session; the frozen
config decisions and their evidence are in `docs/results.md` (2026-09-01
addendum, pending) and the morning report.

**Status: FROZEN 2026-09-01 ~05:50 — evidence in `docs/results.md` addendum.**

## 0. One-time prep

```bash
uv sync
uv run pytest -q                          # expect all green
uv run python scripts/bench_devices.py --device cuda   # ~5 min; replaces every
                                          # MPS-extrapolated estimate below
```

Drop the FULL airport-diverse clip set (complete un-truncated zip) into a
directory of wavs named `<STATION>_YYYYMMDD_HHMMSS.wav` — the station prefix
and timestamp drive per-station splits and capture-block folds. Verify the
extraction is complete this time (`unzip -t`), and check per-station counts:

```bash
ls <clips_dir> | sed -E 's/_[0-9]{8}_[0-9]{6}\.wav//' | sort | uniq -c
```

## 1. Recalibration (full diverse set)

Balanced calibration: if station counts are highly skewed, cap per-station
clips (`--limit` in stage 4 samples across stations; verify presets_stats.json
per-station counts afterwards and re-run with an explicit per-station cap if
one station dominates).

```bash
# 1a. probe TTS (Domain A) — scene text, then RESAMPLE TO 16 kHz (required:
#     build_paired_views writes Kokoro-native 24 kHz, channel_fit needs 16 kHz)
uv run python scripts/build_paired_views.py base \
    --out runs/gan_a_base_v1 --n 200 --seed 0 \
    --config configs/mode1_matched_kixd.yaml --text data/text/scenes_v2.0.1.jsonl
uv run python scripts/build_paired_views.py base \
    --out runs/gan_val_base_v1 --n 64 --seed 1 \
    --config configs/mode1_matched_kixd.yaml --text data/text/scenes_v2.0.1.jsonl
uv run python - <<'PY'
import glob, soundfile as sf, numpy as np
from atcgen.channel.primitives import resample, TARGET_SR
for d in ("runs/gan_a_base_v1/clean", "runs/gan_val_base_v1/clean"):
    for p in glob.glob(d + "/*.wav"):
        wav, sr = sf.read(p, dtype="float32")
        if sr != TARGET_SR:
            sf.write(p, resample(np.asarray(wav, np.float32), sr, TARGET_SR), TARGET_SR)
PY

# 1b. ingest → folds → noise beds → presets
uv run python -m atcgen.dataset.local_corpus <clips_dir> runs/calib_v2
uv run python -m atcgen.dataset.channel_splits \
    --corpus runs/calib_v2/corpus.jsonl --out runs/channel_data_v2
uv run python -m atcgen.dataset.noise_harvest \
    runs/channel_data_v2/corpus.jsonl runs/channel_data_v2/train/noise --split channel_train
uv run python -m atcgen.channel.learned.channel_fit \
    runs/channel_data_v2/corpus.jsonl runs/channel_data_v2/train/presets.jsonl \
    --probe-dir runs/gan_a_base_v1/clean --split channel_train \
    --limit 600 --device cuda
```

**1c. Create the production config (REQUIRED — §2–§4 reference it):**

```bash
cp configs/mode2_fastcut_kixd.yaml configs/mode2_v1.yaml
# then edit configs/mode2_v1.yaml:
#   calibrated.calibration.presets  -> runs/channel_data_v2/train/presets.jsonl
#   calibrated.calibration.noise_bank -> runs/channel_data_v2/train/noise
#   (talker values are already the frozen ones — do not touch; see §5)
```

Manifest-reading note: the per-preset `passband_hz` field looks degenerate for
many presets (widths < 50 Hz); it does not reflect delivered bandwidth. The
fitted EQ lives in `band_edges_hz`/`band_gains_db` and rendered audio is
full-band (dry-run LTAS reproduced the 1.4 dB in-band gap on a different text
corpus).

## 2. FastCUT residual training (frozen decisions from the go/no-go wave)

Frozen: `source+identity` NCE, `--residual-scale-max 0.20`, selection rule
`lexicographic_v1.1_fold_paired_tiebreak` (the trainer applies it; use
`G_selected.pt`, not `G_ema.pt`).

```bash
uv run python -m atcgen.channel.learned.residual_train \
  --corpus runs/channel_data_v2/corpus.jsonl \
  --split channel_train --val-split channel_val \
  --tts-dir runs/gan_a_base_v1/clean --val-tts-dir runs/gan_val_base_v1/clean \
  --presets runs/channel_data_v2/train/presets.jsonl \
  --noise-bank runs/channel_data_v2/train/noise \
  --out runs/fastcut_v1 --device cuda \
  --steps 5000 --batch-size 12 --crop-frames 128 --lr 2e-4 \
  --base 48 --n-res 6 --scales 1 2 4 --num-patches 256 \
  --nce-mode source+identity --lambda-nce 10.0 --lambda-gan 1.0 \
  --r1-gamma 1.0 --r1-every 16 --ema-decay 0.9995 \
  --residual-scale-max 0.20 --a-renders 4 \
  --eval-every 500 --eval-clips 64 --save-every 500 --seed 0
```

~89 min at MPS speed; expect meaningfully less on the 3080 (bench first).
Point `configs/mode2_v1.yaml` residual checkpoint at
`runs/fastcut_v1/G_selected.pt` and set `calibrated.residual.enabled: true`.

## 3. Render the corpus (two-view policy, 155,776 + noise-only)

Rationale: sampling with replacement at 100k draws covers only ~56k of the
77,888 texts; the two-view schedule covers every text exactly twice with fresh
voice/speed/channel draws, `base_id`-paired for later consistency training.

```bash
# main render — exact two-view coverage (noise fraction OFF here)
uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml \
    --n-samples 155776 --out runs/train_v1 --seed 7 \
    --text sequential:data/text/scenes_v2.0.1_2view.jsonl \
    --set dataset.noise_only_frac=0

# hallucination-control noise clips (~3%), same channel definition
uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml \
    --n-samples 4800 --out runs/train_v1_noise --seed 8 \
    --set dataset.noise_only_frac=1.0
```

Rough wall-clock: ~0.38 s/clip on MPS → ~16.5 h; a 3080 typically lands at
3–5× that TTS throughput → ~4–6 h for the main render (bench first). The loop
is single-process — if the bench says you need parallelism, shard by seed
(`--seed 7/17/27`, separate `--out`, split the 2view file) and export all
shards together in §4.

## 4. Gate + export

```bash
uv run python scripts/gate_dataset.py --dataset runs/train_v1 --device cuda
uv run python scripts/export_corpus_csv.py \
    --dataset runs/train_v1 --dataset runs/train_v1_noise \
    --out data/corpus/V1.0.0 --version V1.0.0 --include-noise-only
```

Output: `data/corpus/V1.0.0/corpus_train.csv` + `corpus_test.csv` +
`manifest.json` (sha256), matching the asr repo's V2.1.2 schema
(`audio,text,suspect`, absolute paths, RFC4180). Noise-only rows go to train
only; the test split is cut by transcript group, stratified by airport.

Notes: `--version` is strictly `V<int>.<int>.<int>` (a suffix like
`V1.0.0-dry` raises); `--test-frac` defaults to 0.02 (~3,100 test rows at full
scale — intentional, not a broken split).

**Gate-yield expectation (measured on the overnight dry run):** the calibrated
config is deliberately harder for the frozen teacher pool than the procedural
one — dry-run tier rates were ~26% teacher-rejected / 18% gold (vs 11%/34% for
mode 1; text-pool differences confound part of that gap). This matches the
prior FastCUT wave's ~40% G5 yield, which still produced the best downstream
WER — realistic-but-hard audio is the point, and training should use the gated
tier mix per the recipe, not gold-only. If the gold+silver count matters for a
downstream consumer, the knob is the teacher WER ceiling in the gate config —
an owner decision, made after seeing full-scale gate stats, not by rendering
extra views.

Recommended ASR training usage (evidence: fc_combo + literature): mix real
labeled data with the synthetic at roughly 50–75% real weight per batch, and
finish with a short real-only phase. Do not train synthetic-only when real
labels exist. Per-airport balance: the corpus is airport-balanced by
construction; keep resampling (convergence) and loss-weighting (label noise)
as separate levers if you rebalance further.

## 5. Frozen config values (decided overnight 2026-09-01)

**Talker (frozen at the shipping baseline — do not change):** `tts.speed`
[1.0, 1.4], `voice_augment.pitch_prob` 0.5, `tempo_prob` 0.3, `eq_tilt_prob`
0.4. Evidence: disabling all talker augmentation costs +2.8 WER points (4/4
paired seeds, t=2.90/3df); no individual component is separable at the
overnight budget, and "pitch off for KID fidelity" is explicitly NOT adopted —
confirmed by a powered 10-paired-seed follow-up: removing pitch costs ≈0.9 WER
points (8/10 seeds harmful; see results addendum).

**Channel:** Mode 2 calibrated per §1, FastCUT residual per §2. No WER-searched
channel changes — the overnight power check showed the fine-tune reward is
blind to even large channel manipulations at feasible budgets; channel fidelity
is governed by LTAS + level-matched KID instead.

**Fidelity metrics (post-calibration checks tomorrow):**
- Quote KID only on level-matched, energy-trimmed comparisons (raw KID is
  ~35–40% padding/level contamination; tool:
  the overnight session's `make_matched_sets.py` pattern — trim to active
  region, RMS-normalize both sides, then `atcgen.eval.embed_dist`).
- LTAS check vs the per-station real curves: the in-band (1–3 kHz) gap must be
  ≤ ~2 dB (measurement floor). Known open gap: both modes leak +13–24 dB at
  4 kHz and mode 2 runs +13.5 dB hot at 100 Hz vs real; verify whether the
  trained residual closes this — if not, add a steeper final band edge and
  re-check before rendering.

**Reward machinery (for any future WER-based selection):** bounded per-row WER
(errors capped at reference length), day/session-disjoint dev splits, ≥4 paired
seeds per comparison, paired stats only. `runs/power_check_kixd/summary.json`
is mixed-metric — use the rescoring tools, not that file.

**Deliberately unspent:** `data/real/kixd/kixd_locked_day.csv` (day 20250808,
337 rows) has never been read by any selection. Spend it once, on the final
trained model, as the last pre-ship check.

## 6. Licensing

See `docs/data-licensing.md`. Action items for Kevin: verify KIXD capture
provenance vs LiveATC terms; email CMU AirLab re TartanAviation data license.
`jacktol/atc-dataset` remains eval-only — never in training mixes.
