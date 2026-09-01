#!/usr/bin/env python
"""Export a generated dataset's manifest as the asr repo's corpus CSVs.

The asr side eats `audio,text,suspect` CSVs plus a `manifest.json` naming them
and their digests — see
`reference-data-for-v1-run/asr/resources/datasets/V2.1.2/`.  Nothing in atcgen
wrote that shape until now; `manifest.jsonl` stores audio paths *relative to the
manifest*, so the join with the run directory happens here.

    uv run python scripts/export_corpus_csv.py \
        --dataset runs/train_v1 --dataset runs/train_v1_noise \
        --out data/corpus/V1.0.0 --version V1.0.0 --include-noise-only

`--dataset` repeats, and the manifests are merged *before* the split, so a
production set rendered as two runs — the scheduled speech and its noise-only
hallucination control — exports as one corpus with one split rather than two
corpora that each need their own.

Three rules worth naming, because each is a judgement call the file cannot make
for itself:

* `suspect` is `False` on every row.  The flag marks human-doubted transcripts
  in the real corpus; synthetic text is the label the TTS was handed, and rows
  that failed QC or the gate were already filtered upstream.
* Noise-only rows (empty `text`) are dropped by default.  They are Whisper
  hallucination control, and a CSV consumer that reads `text` as ground truth
  will read them as transcription failures.  `--include-noise-only` keeps them,
  and puts them in **`corpus_train.csv` only** — a held-out set exists to
  measure transcription, and an empty reference scores as either a free win or
  a total loss depending on the metric, neither of which says anything about
  the model.
* The test split is cut by **transcript group**, stratified by `kind` (the
  airport, for scene-derived text).  Generation samples the text pool with
  replacement, so a plain row-wise split puts the same utterance on both sides;
  assigning whole text groups to one side keeps the split honest.  Per-scene
  identity does not survive into the manifest, so the airport is the coarsest
  handle that does.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

FIELDS = ["audio", "text", "suspect"]


def read_manifest(dataset: Path) -> list[dict]:
    """Manifest rows with `audio` resolved to an absolute path."""
    manifest = dataset / "manifest.jsonl" if dataset.is_dir() else dataset
    root = manifest.parent
    rows = []
    for line in manifest.open():
        if not line.strip():
            continue
        row = json.loads(line)
        row["audio"] = (root / row["audio"]).resolve()
        rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {manifest}")
    return rows


def read_manifests(datasets: list[Path]) -> tuple[list[dict], dict[str, int]]:
    """Merge several runs' manifests, keeping run order. Rejects duplicates.

    Two runs written to the same `--out` would share wav paths, and the split
    would then put one clip on both sides.
    """
    rows: list[dict] = []
    per_dataset = {}
    for dataset in datasets:
        loaded = read_manifest(dataset)
        per_dataset[str(dataset)] = len(loaded)
        rows.extend(loaded)
    seen = {row["audio"] for row in rows}
    if len(seen) != len(rows):
        raise ValueError("merged manifests share audio paths; the runs must "
                         "have been written to the same directory")
    return rows, per_dataset


def split_rows(rows: list[dict], test_frac: float,
               seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out `test_frac` of rows, by transcript group, stratified by `kind`."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("kind") or "unknown", row["text"])].append(row)
    by_kind: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in groups:
        by_kind[key[0]].append(key)

    rng = random.Random(seed)
    test_keys: set[tuple[str, str]] = set()
    for kind in sorted(by_kind):
        keys = sorted(by_kind[kind])
        rng.shuffle(keys)
        target = test_frac * sum(len(groups[key]) for key in keys)
        taken = 0
        for key in keys:
            # Take a whole group only while it lands the stratum nearer its
            # target than stopping would; a group larger than what is left
            # over-fills the split rather than rounding up into it.
            if taken + len(groups[key]) / 2 > target:
                break
            test_keys.add(key)
            taken += len(groups[key])

    # Both sides keep manifest order: set iteration order is not stable across
    # processes, and the digests in manifest.json have to be.
    train = [row for key, group in groups.items() if key not in test_keys
             for row in group]
    test = [row for key, group in groups.items() if key in test_keys
            for row in group]
    return train, test


def write_csv(path: Path, rows: list[dict]) -> str:
    """Write the `audio,text,suspect` CSV and return its sha256."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS,
                                quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"audio": row["audio"].as_posix(),
                             "text": row["text"], "suspect": "False"})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_version(version: str) -> dict:
    """`V2.1.2` -> the three integer components the asr manifest carries."""
    parts = version.lstrip("Vv").split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"version must look like 'V2.1.2': {version!r}")
    very_major, major, minor = (int(part) for part in parts)
    return {"very_major": very_major, "major": major, "minor": minor}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, action="append", required=True,
                    help="generated run directory (or a manifest.jsonl directly); "
                         "repeat to merge several runs into one corpus")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--test-frac", type=float, default=0.02)
    ap.add_argument("--include-noise-only", action="store_true",
                    help="keep the empty-text hallucination-control rows")
    ap.add_argument("--version", default="V1.0.0", help="corpus version tag")
    ap.add_argument("--reason", default="synthetic export", help="manifest.json note")
    ap.add_argument("--seed", type=int, default=0, help="test-split seed")
    args = ap.parse_args(argv)

    rows, per_dataset = read_manifests(args.dataset)
    total = len(rows)
    speech = [row for row in rows if row["text"].strip()]
    noise_only = [row for row in rows if not row["text"].strip()]

    # The split runs over speech alone; noise-only rows are appended to train
    # afterwards so they can never reach the held-out side.
    train, test = split_rows(speech, args.test_frac, args.seed)
    if args.include_noise_only:
        train = train + noise_only
        rows = speech + noise_only
    else:
        rows = speech

    args.out.mkdir(parents=True, exist_ok=True)
    digests = {"train_csv": write_csv(args.out / "corpus_train.csv", train),
               "test_csv": write_csv(args.out / "corpus_test.csv", test)}

    clips = {row["audio"].parent for row in rows}
    lineage = rows[0].get("lineage", {}) if rows else {}
    manifest = {
        "version": args.version,
        **parse_version(args.version),
        "created_at": datetime.now(UTC).isoformat(),
        "reason": args.reason,
        "train_csv": "corpus_train.csv",
        "test_csv": "corpus_test.csv",
        "clips_dir": (next(iter(clips)).as_posix() if len(clips) == 1 else None),
        "sha256": digests,
        "source": {
            "datasets": {str(Path(name).resolve()): count
                         for name, count in per_dataset.items()},
            "clips_dirs": sorted(path.as_posix() for path in clips),
            "rows_in_manifest": total,
            "noise_only_dropped": 0 if args.include_noise_only else len(noise_only),
            "noise_only_in_train": len(noise_only) if args.include_noise_only else 0,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_frac_target": args.test_frac,
            "split": f"transcript-group, stratified by kind, seed {args.seed}",
            "config_hash": lineage.get("config_hash"),
            "git_revision": lineage.get("git_revision"),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
