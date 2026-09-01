#!/usr/bin/env python
"""Join the V2.1.2 ASR transcripts onto the KIXD clips and cut session splits.

    uv run python scripts/join_kixd_transcripts.py

Transcripts come from `reference-data-for-v1-run/asr/resources/datasets/V2.1.2/`
(`corpus_train.csv` + `corpus_test.csv`, schema `audio,text,suspect`), which
covers **every** clip in `updated_kixd_clips/` — the older
`tests/resources/corpus_snapshot_*.csv` fixtures are a smaller V1-era subset and
are not used.  Rows key on the audio *basename*; the directory component
(`/mnt/data0/kixd/...`) is a training-box path and is discarded, exactly as the
asr repo's own loaders do.  The `rename_kixd_clips_from_forlaron_mtime*.log`
mapping is applied as a fallback for any legacy `KIXD_8-1-2025_clipNNNN.wav`
row, though V2.1.2 turns out to be fully renamed already.

Three filters produce the "clean" pool:

* `suspect=True` — the asr reviewers' own doubt flag, dropped by
  `AviationDataset` by default.
* empty transcripts.
* `[inaudible]`-family bracket tags.  The asr repo scores these through a
  wildcard that its edit distance absorbs; `training/normalize.normalize_atc`
  has no such wildcard, so a surviving tag would be counted as a word the model
  failed to say.

**The V2.1.2 train/test membership is deliberately ignored.**  Its test rows are
spread proportionally across all eight capture days, so honouring it would put
the same session — often the same exchange — on both sides of the boundary.
Splits are cut by whole capture day instead:

    kixd_train        days 20250801-20250806      the training pool
    kixd_heldout      days 20250807 + 20250808    reference view of both
    kixd_dev          200 rows from 20250807      the RL reward's KIXD half
    kixd_locked_day   day 20250808, whole         DELIBERATELY UNSPENT

`kixd_locked_day` is the one-shot read after the config is frozen. Nothing in
the search loop may touch it; that is why the RL dev slice is cut from day
20250807 alone and not from the heldout pair.

Each split is written as an `audio,text` CSV (CRLF, matching the asr repo's
corpus CSVs) and as a `{"audio","text"}` JSONL, which is what
`training/finetune_whisper.py --real-manifest` and
`atcgen.dataset.real_atc.load_local_corpus` read.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference-data-for-v1-run"
DEFAULT_CLIPS = REFERENCE / "updated_kixd_clips"
DEFAULT_CORPUS = REFERENCE / "asr" / "resources" / "datasets" / "V2.1.2"
DEFAULT_LOGS = [
    REFERENCE / "asr" / "rename_kixd_clips_from_forlaron_mtime.log",
    REFERENCE / "asr" / "rename_kixd_clips_from_forlaron_mtime_2.log",
]

RENAME = re.compile(r"^RENAME \[\d+\] (\S+\.wav) -> (\S+\.wav)$")
#: `local_corpus.parse_station` / `channel_splits.parse_timestamp` shape.
CLIP_NAME = re.compile(r"^(?P<station>.+)_(?P<day>\d{8})_(?P<time>\d{6})\.wav$")
BRACKET_TAG = re.compile(r"\[.*?\]")


def read_rename_map(paths: list[Path]) -> dict[str, str]:
    """old basename -> new basename, from the rename logs' RENAME lines."""
    mapping: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            match = RENAME.match(line.strip())
            if match is None:
                continue
            old, new = match.group(1), match.group(2)
            if mapping.setdefault(old, new) != new:
                raise ValueError(f"conflicting rename for {old} in {path}")
    return mapping


def read_corpus(corpus_dir: Path) -> list[dict]:
    """Both halves of a versioned `audio,text,suspect` corpus snapshot."""
    rows = []
    for name in ("corpus_train.csv", "corpus_test.csv"):
        with (corpus_dir / name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"basename": Path(row["audio"]).name,
                             "text": row["text"],
                             "suspect": row["suspect"] == "True",
                             "source": name})
    if not rows:
        raise ValueError(f"no transcript rows under {corpus_dir}")
    return rows


def resolve(rows: list[dict], renames: dict[str, str],
            clips: Path) -> tuple[dict[str, str], Counter]:
    """new basename -> clean transcript, for clips present on disk."""
    on_disk = {path.name for path in clips.glob("*.wav")}
    texts: dict[str, str] = {}
    tally: Counter = Counter()
    for row in rows:
        tally["rows"] += 1
        base = row["basename"]
        if base in on_disk:
            tally["matched_directly"] += 1
        elif renames.get(base) in on_disk:
            base = renames[base]
            tally["matched_via_rename"] += 1
        else:
            tally["no_local_audio"] += 1
            continue
        if row["suspect"]:
            tally["suspect"] += 1
            continue
        text = " ".join(row["text"].split())
        if not text:
            tally["empty_text"] += 1
            continue
        if BRACKET_TAG.search(text):
            tally["bracket_tag"] += 1
            continue
        if base in texts:
            tally["duplicate"] += 1
            continue
        texts[base] = text
        tally["clean"] += 1
    return texts, tally


def capture_day(name: str) -> str:
    match = CLIP_NAME.match(name)
    if match is None:
        raise ValueError(f"clip name does not parse as station_YYYYMMDD_HHMMSS: {name}")
    return match.group("day")


def write_split(out_dir: Path, stem: str, names: list[str], texts: dict[str, str],
                clips: Path) -> int:
    """Write one split as both `audio,text` CSV and `{"audio","text"}` JSONL."""
    rows = [{"audio": str((clips / name).resolve()), "text": texts[name]}
            for name in names]
    with (out_dir / f"{stem}.csv").open("w", newline="") as handle:
        # CRLF and minimal quoting, matching the asr repo's corpus CSVs.
        writer = csv.DictWriter(handle, fieldnames=["audio", "text"],
                                lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{stem}.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS,
                    help="directory of KIXD_TOWER_*.wav clips")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help="versioned corpus dir holding corpus_{train,test}.csv")
    ap.add_argument("--rename-log", type=Path, action="append", default=None,
                    help="legacy-name rename log; repeat (default: both under asr/)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "real" / "kixd")
    ap.add_argument("--dev-day", default="20250807",
                    help="capture day the RL dev slice is sampled from")
    ap.add_argument("--locked-day", default="20250808",
                    help="capture day held back entirely for the post-freeze read")
    ap.add_argument("--dev-rows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="dev-slice sampling seed")
    args = ap.parse_args(argv)

    renames = read_rename_map(args.rename_log or DEFAULT_LOGS)
    texts, tally = resolve(read_corpus(args.corpus), renames, args.clips)

    by_day: dict[str, list[str]] = {}
    for name in sorted(texts):
        by_day.setdefault(capture_day(name), []).append(name)
    heldout_days = {args.dev_day, args.locked_day}
    missing = heldout_days - set(by_day)
    if missing:
        raise ValueError(f"no clean clips on capture day(s) {sorted(missing)}")

    train = [name for day in sorted(set(by_day) - heldout_days) for name in by_day[day]]
    heldout = [name for day in sorted(heldout_days) for name in by_day[day]]
    locked = by_day[args.locked_day]
    dev_pool = by_day[args.dev_day]
    if len(dev_pool) < args.dev_rows:
        raise ValueError(f"day {args.dev_day} has {len(dev_pool)} clean clips, "
                         f"fewer than --dev-rows {args.dev_rows}")
    rl_dev = sorted(random.Random(args.seed).sample(dev_pool, args.dev_rows))

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {
        stem: write_split(args.out, stem, names, texts, args.clips)
        for stem, names in [("kixd_labeled", sorted(texts)), ("kixd_train", train),
                            ("kixd_heldout", heldout), ("kixd_dev", rl_dev),
                            ("kixd_locked_day", locked)]
    }

    summary = {
        "clips_on_disk": len(list(args.clips.glob("*.wav"))),
        "corpus": str(args.corpus.resolve()),
        "rename_pairs": len(renames),
        "join": dict(tally),
        "clean_per_day": {day: len(names) for day, names in sorted(by_day.items())},
        "train_days": sorted(set(by_day) - heldout_days),
        "dev_day": args.dev_day,
        "locked_day": args.locked_day,
        "counts": counts,
        "out_dir": str(args.out.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
