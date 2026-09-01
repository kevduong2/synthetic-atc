# V1 production runbook — RTX 3080 (2026-09-01)

End-to-end on the Windows/3080 box: recalibrate on the full multi-airport clip
set → train the FastCUT residual → render the two-view corpus → gate → export
`corpus_train.csv` / `corpus_test.csv`. The frozen config decisions and their
evidence are in `docs/results.md` (2026-09-01 addendum) and the morning report.
Every command below is PowerShell (one line, or backtick-continued; no `\`,
no heredocs). Run it as lab mission `lab/missions/prod-v1.md` if the Copilot
team is driving; long jobs go through `scripts/lab/jobs.py launch --gpu` so
they survive the tool call and the lab-assistant can watch them.

**Status: FROZEN 2026-09-01 ~05:50 — evidence in `docs/results.md` addendum.**

## 0. One-time prep

**0.1 Payload from the Mac** (gitignored; same relative layout under the repo
root). The production run rebuilds its own calibration, so it needs less than
the experiment window does:

| Path | Needed for |
|---|---|
| `data/` (110 MB) | `data/text/scenes_v2.0.1.jsonl` (probe text), `scenes_v2.0.1_2view.jsonl` (render), real-audio manifests |
| the FULL airport clip archive | §1 recalibration — see `docs/data-handoff.md` for the delivery contract |
| `data/real/calibration/` (own-SDR KSDL/KSLE clips) | merge into the §1 input dir so those stations get presets too |
| `runs/calib_kixd/`, `runs/channel_data_kixd/` (optional) | only to compare the new KIXD presets against the overnight ones |

**0.2 Environment**

```powershell
uv python install 3.11
uv sync                                  # torch/torchaudio from the cu126 index on Windows; driver >= 560
uv run python -c "import torch, soundfile; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), soundfile.__libsndfile_version__)"
uv run pytest -q                         # expect ~780 passed
uv run python scripts/bench_devices.py --device cuda --gan --wavlm --tts --dsp --sft --out runs/bench/cuda.json
```

`cuda.is_available()` must be `True`; libsndfile should read 1.2.2. The bench
needs at least one section flag (it errors otherwise) and replaces every
MPS-extrapolated estimate below — read its s/render and s/step before planning
the day. If real-audio manifests were copied from the Mac, rewrite their
absolute paths once (`scripts/lab/relocate.py`, see the `gpu-jobs` skill);
nothing built fresh on this machine needs it.

**0.3 Clips.** Extract the complete un-truncated archive into one directory of
wavs named `<STATION>_YYYYMMDD_HHMMSS.wav` — the station prefix drives
per-station splits and the timestamp drives capture-block folds. Test the
archive first (the last one was cut at 1.2 GB), then check per-station counts:

```powershell
uv run python -m zipfile -t airport_clips_v2.zip
Expand-Archive airport_clips_v2.zip -DestinationPath reference-data-for-v1-run/airport_clips_v2
Copy-Item data/real/calibration/*.wav reference-data-for-v1-run/airport_clips_v2/     # own-SDR stations, if not already in the archive
Get-ChildItem reference-data-for-v1-run/airport_clips_v2 -Filter *.wav | ForEach-Object { $_.BaseName -replace '_\d{8}_\d{6}$','' } | Group-Object | Select-Object Count,Name
```

Every expected station (KEUG, KOJC, S50, KSLE, KIXD, KSDL) must appear with a
consistent `ICAO_FACILITY` spelling; a file that does not match the pattern
lands in station `unknown`. Below, `<clips_dir>` is that directory.

## 1. Recalibration (full diverse set)

Balance is applied here, not by deleting data. Two seeded per-station caps
bound the work: `local_corpus --per-station 1500` keeps at most 1,500 source
files per station before any decoding (the delivered archive holds ~209k
clips; calibration needs hundreds per station, and 1,500 leaves the 1,000-clip
KID reference subset and honest capture-block folds with room to spare), and
`channel_fit --per-station 150` fits presets from at most 150 of those. Plain
`--limit` on `channel_fit` is a head truncation of a file grouped by station —
never use it alone on this corpus. The full archive stays on disk untouched;
raising a cap later is a rerun of §1, not a re-delivery.

```powershell
# 1a. probe TTS (Domain A) — scene text; build_paired_views writes Kokoro-native
#     24 kHz and channel_fit requires 16 kHz, so resample in place afterwards
uv run python scripts/build_paired_views.py base --out runs/gan_a_base_v1 --n 200 --seed 0 --config configs/mode1_matched_kixd.yaml --text data/text/scenes_v2.0.1.jsonl
uv run python scripts/build_paired_views.py base --out runs/gan_val_base_v1 --n 64 --seed 1 --config configs/mode1_matched_kixd.yaml --text data/text/scenes_v2.0.1.jsonl
uv run python scripts/lab/resample_probes.py runs/gan_a_base_v1/clean runs/gan_val_base_v1/clean

# 1b. ingest -> folds -> noise beds -> presets
uv run python -m atcgen.dataset.local_corpus <clips_dir> runs/calib_v2 --per-station 1500
uv run python -m atcgen.dataset.channel_splits --corpus runs/calib_v2/corpus.jsonl --out runs/channel_data_v2
uv run python -m atcgen.dataset.noise_harvest runs/channel_data_v2/corpus.jsonl runs/channel_data_v2/train/noise --split channel_train
uv run python scripts/lab/jobs.py launch --gpu --id prod-p1-fit -- uv run python -m atcgen.channel.learned.channel_fit runs/channel_data_v2/corpus.jsonl runs/channel_data_v2/train/presets.jsonl --probe-dir runs/gan_a_base_v1/clean --split channel_train --per-station 150 --device cuda
```

Note on 1a: `base` renders ~3% noise-only beds among the probes (the frozen
KIXD calibration was fitted the same way); leave it, it is what the residual
trainer's Domain A expects.

**Check before moving on** — station counts at each stage:

```powershell
uv run python -c "import json; print(json.load(open('runs/calib_v2/corpus_stats.json'))['stations'])"
uv run python -c "import json; s=json.load(open('runs/channel_data_v2/train/presets_stats.json')); print({k: v['n'] for k, v in s['stations'].items()}, 'dropped', s['dropped'])"
```

Every deployed station (KEUG, KOJC, S50, KSLE, KIXD, KSDL) must have presets.
If one has fewer than ~30 kept, its QC drop rate is the first suspect (see
`dropped_clips`). Small non-deployed receivers (SEATTLE_CENTER, KSLE_GROUND)
may have fewer; they are not drawn at render time (§1c).

**1c. Create the production config (REQUIRED — §2–§4 reference it):**

```powershell
Copy-Item configs/mode2_fastcut_kixd.yaml configs/mode2_v1.yaml
```

Then edit `configs/mode2_v1.yaml` under `calibrated.calibration`:

- `corpus_dir: reference-data-for-v1-run/airport_clips_v2` (informational)
- `presets: runs/channel_data_v2/train/presets.jsonl`
- `noise_bank: runs/channel_data_v2/train/noise`
- `station_mix:` — **set it explicitly.** Unset, the render draws each clip's
  channel station in proportion to preset counts; with `--per-station 150` that
  is close to uniform, but small stations (fewer kept presets) would be
  under-drawn. Use uniform weights over the stations in `presets_stats.json`,
  spelled exactly as they appear there, e.g.
  `station_mix: {KEUG_TOWER: 1, KOJC_TOWER: 1, S50_TOWER: 1, KSLE_TOWER: 1, KIXD_TOWER: 1, KSDL_TOWER: 1}`
  (a name with no presets raises at load). Stations you leave out are never
  drawn: the own-SDR dir also carries small non-deployed receivers
  (SEATTLE_CENTER, KSLE_GROUND); keep their presets and noise beds in the pool
  but do not list them.
- `cross_station_prob: 0.1` stays as frozen. Note it was a no-op on the
  single-station KIXD calibration and becomes live here (10% of clips get
  another station's noise bed). The §5 LTAS/KID check is the guard; if it
  fails and the cross-station beds are the cause, zeroing it is an owner call.
- Talker values are already the frozen ones — do not touch (§5).

Validate it loads:

```powershell
uv run python -c "from atcgen.config import load_config; c = load_config('configs/mode2_v1.yaml'); print(c.calibrated.calibration)"
```

Manifest-reading note: the per-preset `passband_hz` field looks degenerate for
many presets (widths < 50 Hz); it does not reflect delivered bandwidth. The
fitted EQ lives in `band_edges_hz`/`band_gains_db` and rendered audio is
full-band.

## 2. FastCUT residual training (frozen decisions from the go/no-go wave)

Frozen: `source+identity` NCE, `--residual-scale-max 0.20`, selection rule
`lexicographic_v1.1_fold_paired_tiebreak` (the trainer applies it; use
`G_selected.pt`, not `G_ema.pt`).

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-p2-resid -- uv run python -m atcgen.channel.learned.residual_train `
  --corpus runs/channel_data_v2/corpus.jsonl --split channel_train --val-split channel_val `
  --tts-dir runs/gan_a_base_v1/clean --val-tts-dir runs/gan_val_base_v1/clean `
  --presets runs/channel_data_v2/train/presets.jsonl --noise-bank runs/channel_data_v2/train/noise `
  --out runs/fastcut_v1 --device cuda `
  --steps 5000 --batch-size 12 --crop-frames 128 --lr 2e-4 --base 48 --n-res 6 --scales 1 2 4 --num-patches 256 `
  --nce-mode source+identity --lambda-nce 10.0 --lambda-gan 1.0 --r1-gamma 1.0 --r1-every 16 --ema-decay 0.9995 `
  --residual-scale-max 0.20 --a-renders 4 --eval-every 500 --eval-clips 64 --save-every 500 --seed 0
```

~89 min at MPS speed; expect meaningfully less on the 3080 (bench first).
`--resume <ckpt>` exists if the window is interrupted; `--save-every 500`
leaves restart points. On OOM drop `--batch-size` to 8 and record it: that
changes the recipe.

**Check the selection before touching the config.** `G_selected.pt` is only
written when at least one eval passed the gates; otherwise the trainer leaves
`G_ema.pt` and the config's strict loader would fail the whole render.

```powershell
uv run python -c "import json; print(json.load(open('runs/fastcut_v1/validation_report.json'))['selection'])"
```

`status` must be `selected`. If it is `no_eligible_candidate`, stop: that is a
result for the owner (the residual did not pass its own fold gates on the new
calibration), not something to route around with `G_ema.pt`. Then in
`configs/mode2_v1.yaml` set `calibrated.residual.enabled: true` and
`calibrated.residual.checkpoint: runs/fastcut_v1/G_selected.pt`.

## 3. Render the corpus (two-view policy, 155,776 + noise-only)

Rationale: sampling with replacement at 100k draws covers only ~56k of the
77,888 texts; the two-view schedule covers every text exactly twice with fresh
voice/speed/channel draws, `base_id`-paired for later consistency training.

`generate_dataset.py` is not resumable and writes `stats.json` only when it
finishes, so a crash at hour four of a five-hour render loses everything.
Shard by default: four round-robin shards (each airport-mixed), one seed and
`--out` each, launched one after another under the GPU lock, exported together
in §4. A lost shard costs a quarter.

```powershell
uv run python scripts/lab/shard_text.py data/text/scenes_v2.0.1_2view.jsonl --n 4     # 38,944 lines each
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s1 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s1 --seed 7  --text sequential:data/text/scenes_v2.0.1_2view.shard1of4.jsonl --set dataset.noise_only_frac=0
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s2 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s2 --seed 17 --text sequential:data/text/scenes_v2.0.1_2view.shard2of4.jsonl --set dataset.noise_only_frac=0
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s3 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s3 --seed 27 --text sequential:data/text/scenes_v2.0.1_2view.shard3of4.jsonl --set dataset.noise_only_frac=0
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-s4 -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 38944 --out runs/train_v1_s4 --seed 37 --text sequential:data/text/scenes_v2.0.1_2view.shard4of4.jsonl --set dataset.noise_only_frac=0

# hallucination-control noise clips (~3%), same channel definition
uv run python scripts/lab/jobs.py launch --gpu --id prod-p3-noise -- uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 4800 --out runs/train_v1_noise --seed 8 --set dataset.noise_only_frac=1.0
```

Each launch is refused (exit 4) while the previous shard holds the lock; the
watcher's `finished` event is the cue for the next one. `--n-samples` must
equal the shard's line count (`sequential:` refuses more). Rough wall-clock:
~0.38 s/clip on MPS → ~16.5 h; a 3080 typically lands at 3–5× that TTS
throughput → ~4–6 h total (bench first). If a shard fails, re-render only that
shard with the same seed and `--out`.

## 4. Gate + export

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s1 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s1 --device cuda
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s2 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s2 --device cuda
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s3 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s3 --device cuda
uv run python scripts/lab/jobs.py launch --gpu --id prod-p4-gate-s4 -- uv run python scripts/gate_dataset.py --dataset runs/train_v1_s4 --device cuda
uv run python scripts/export_corpus_csv.py --dataset runs/train_v1_s1 --dataset runs/train_v1_s2 --dataset runs/train_v1_s3 --dataset runs/train_v1_s4 --dataset runs/train_v1_noise --out data/corpus/V1.0.0 --version V1.0.0 --include-noise-only --reason "V1 production render, mode2_v1, fastcut_v1"
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

**Fidelity check before §3 (required, ~20 min):** render 150 clips with the
finished `configs/mode2_v1.yaml` and compare against the new real set.

```powershell
uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 150 --out runs/prod_fid --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set dataset.noise_only_frac=0
uv run python scripts/analysis/make_matched_sets.py --out runs/prod_fid/kid --real-dir runs/calib_v2/clips --syn v1=runs/prod_fid/wavs
# then the embed_dist command it prints, with --device cuda
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid/wavs --label real --label v1 --limit 1000 --json runs/prod_fid/ltas.json
```

- Quote KID only on level-matched, energy-trimmed comparisons (raw KID is
  ~35–40% padding/level contamination). The overnight KIXD-only matched KID is
  the reference point; a multi-station number is not directly comparable, so
  report it as the V1 baseline rather than against the KIXD one.
- LTAS: the in-band (1–3 kHz) gap must be ≤ ~2 dB (measurement floor). Known
  open gap: both modes leak +13–24 dB at 4 kHz and mode 2 runs +13.5 dB hot
  at 100 Hz vs real; verify whether the trained residual closes this — if not,
  test the cheap fix offline first: `scripts/analysis/filter_variants.py
  runs/prod_fid/wavs --out runs/prod_fid/variants` writes LP (3.8 kHz) and
  LP+HP (150 Hz) copies; re-run matched KID and LTAS on them (mission branch
  B3) and hand the 4-row table to the owner before any band edge lands in the
  config. Per-station
  curves: run `ltas_check.py` on per-station subsets of `runs/calib_v2/clips`
  (clip ids keep the station prefix) if any station's counts look off.

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
