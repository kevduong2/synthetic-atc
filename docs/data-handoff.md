# Real-audio data handoff — completing the airport clip set

How to deliver the remaining airport recordings so calibration, balancing, and
evaluation work without rework. Written 2026-09-01 after the overnight freeze;
companion to `docs/runbook-v1-3080.md` (§1 consumes what this doc delivers)
and `docs/data-licensing.md` (every new source gets a license verdict there
before it enters the pipeline).

## Current inventory vs. target

The deployed-airport text corpus covers six airports, near-evenly
(~12–13.5k utterances each): **KEUG, KOJC, S50, KSLE, KIXD, KSDL**.

Real audio on hand:

| Station | Clips | Hours | Transcripts | Source |
|---|---:|---:|---|---|
| KIXD_TOWER | 7,374 | 15.2 | 6,502 clean (V2.1.2 join) | `reference-data-for-v1-run/updated_kixd_clips/` |
| KSDL_TOWER (+ SEATTLE_CENTER, KSLE_TOWER/GROUND, small) | ~100s | ~1–2 | none | `data/real/calibration/` (own SDR) |
| KEUG, KOJC, S50 | **0** | 0 | — | **missing** |

The original `updated_kixd_clips.zip` was **truncated at 1.2 GB** (missing
end-of-central-directory; recovered by streaming the local headers). If the
full archive contained other airports, they sat past the cutoff. Re-deliver
the complete archive — and note zip entries extract in order, so a re-download
that completes will contain everything.

## Delivery format (hard requirements)

1. **Filename convention — this is load-bearing:**
   `<STATION>_<YYYYMMDD>_<HHMMSS>.wav`, e.g. `KEUG_TOWER_20250815_143022.wav`.
   `local_corpus` parses the station prefix (regex `^(.+)_\d{8}_\d{6}$`) for
   station-stratified splits, and `channel_splits` parses the timestamp to cut
   session-disjoint capture blocks (15-minute-gap rule). A file that doesn't
   match gets no station and no fold. Use `ICAO_FACILITY` for the station
   (`KEUG_TOWER`, not `keug` or `Eugene`), one consistent spelling per
   receiver/frequency.
2. **Audio:** WAV PCM (16 kHz mono PCM16 preferred; the pipeline resamples and
   downmixes anything readable, so higher rates are fine). **Never MP3 or any
   lossy transcode** — codec artifacts would become part of the "real" channel
   we calibrate to.
3. **No processing.** Deliver straight off the capture chain: no loudness
   normalization, no AGC after capture, no denoising, no trimming beyond your
   segmenter. Fixed capture gain per receiver is fine (KIXD sits ~9 dB colder
   than KSDL; the matched-KID protocol handles level — post-hoc "fixing" it
   would destroy information).
4. **Squelch-gated single transmissions** (like the KIXD set: one transmission
   per file, ~0.8 s pad each side, 2–30 s long) are the ideal shape. Long
   continuous recordings also work — `noise_harvest`/VAD handle them — but
   per-transmission clips make transcripts and eval much cleaner.
5. **Verify the archive before handing it over:** `unzip -t archive.zip` must
   report no errors, and per-station counts should match what you expect:

   ```bash
   unzip -l archive.zip | grep -oE '[A-Z0-9]+_[A-Z]+_[0-9]{8}' | sed -E 's/_[0-9]{8}$//' | sort | uniq -c
   ```

## How much per airport, and how balance actually works

**Don't pre-balance by deleting data — deliver everything you have.** Balance
is applied downstream, where it belongs:

- `local_corpus` stratifies its splits by station.
- `channel_fit --limit N` samples presets across stations; after it runs,
  check `presets_stats.json` per-station counts and re-run with a per-station
  cap if one station dominates (runbook §1 covers this).
- Selection/eval metrics are macro-averaged per airport with a worst-airport
  guard, so a big station can't dominate scoring.
- Two levers stay separate on the training side: **resampling** fixes
  convergence balance, **loss weighting** compensates noisier labels — never
  collapse them into one knob.

Volume guidance per station:

| Amount | What it enables |
|---|---|
| ~100 clips (~10 min speech) | minimum for a usable preset fit + noise beds (prior wave calibrated from 99 clips total) |
| **300–500 clips, ≥2 capture days** | comfortable target: presets, noise diversity, session-disjoint folds, per-airport KID reference |
| 1,000+ | diminishing returns for calibration; still valuable as eval/adaptation data if transcribed |

Uneven is survivable (the pipeline caps and macro-averages); **absent is not**
— an airport with zero real audio gets synthetic audio calibrated to *other*
towers' channels, which is exactly the bias the balancing work exists to
avoid. Even ~10 minutes per missing airport beats nothing.

Session diversity matters as much as clip count: several capture blocks spread
over days/hours (the 15-minute-gap rule defines blocks) make the
`channel_val` fold honest. One receiver per station is fine — but note it in
the manifest below, since same-receiver data can never support a
receiver-generalization claim for that station.

## Transcripts (optional for calibration, required for per-airport WER)

Calibration, noise beds, presets, and KID need **no transcripts**. Per-airport
WER evaluation and reward slices do. If transcripts exist:

- Deliver in the asr V2 shape: CSV with `audio,text,suspect`, keyed by
  **basename** (directories are ignored by every consumer), text lowercase,
  numbers spelled out, `suspect` marking doubtful rows.
- If clips get renamed between transcription and delivery, include the rename
  log (`old -> new` per line) — that log is how 6,502 KIXD transcripts were
  recovered; without it they'd have been orphaned.

## One small manifest per delivery

A `README.txt` in the archive root, a few lines per station: receiver/antenna,
frequency, location, capture dates, segmenter used, and the audio's origin
(own SDR? someone else's feed?). The origin line is the licensing gate —
LiveATC-derived audio cannot go down the commercial path
(`docs/data-licensing.md`), so it must be identifiable per station.

## What happens on arrival (for whoever runs it)

```bash
unzip -t new_clips.zip                          # integrity FIRST (the 1.2 GB lesson)
unzip new_clips.zip -d reference-data-for-v1-run/airport_clips_v2/
ls reference-data-for-v1-run/airport_clips_v2/ \
  | sed -E 's/_[0-9]{8}_[0-9]{6}\.wav//' | sort | uniq -c   # station counts
```

Then runbook §1 end-to-end on the combined directory (new clips + existing
`data/real/calibration/` contents can be merged into one input dir — station
prefixes keep them apart). Spot-check afterwards: per-station counts in
`runs/calib_v2/corpus_stats.json` and `presets_stats.json`, and one matched-KID
+ LTAS read per station against its own real clips before starting the big
render.
