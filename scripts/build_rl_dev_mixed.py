#!/usr/bin/env python
"""Materialize the European dev half and mix it with the KIXD half.

    uv run python scripts/build_rl_dev_mixed.py

The RL reward needs a dev slice that a KIXD-only overfit cannot quietly win.
KIXD is the only locally labeled audio, so the second region comes from
`jacktol/atc-dataset` via the registry's `heldout_tail_check` split
(`test[2500:]`, ~427 rows, added in 0ae2b30 and still unspent).  This exports
200 of its rows to real wav files — the HF cache is not a path the local-corpus
loader can read — and concatenates them with the 200 KIXD day-20250807 rows
that `scripts/join_kixd_transcripts.py` wrote.

Equal row counts are the whole point.  Pooling the full sets would let KIXD's
1110 heldout clips swamp the 427 European ones and reproduce exactly the
regional overfit the second half is there to detect; 200 against 200 makes the
pooled WER an unweighted average of the two regions without the reward harness
needing to learn about macro-averaging.  A candidate that wins by memorizing
KIXD tower phraseology shows up as a European regression.

`heldout_tail_check` is descriptive-only by registry policy
(`atcgen/dataset/splits.py:135`) — it is a directional check with roughly a
6-point MDE, never a locked test — and using it as a reward dev slice spends it
further.  `locked_test` (`test[500:2500]`) is untouched.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atcgen.dataset.real_atc import REAL_SR  # noqa: E402
from atcgen.dataset.splits import load_split, split_spec  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EU_OUT = ROOT / "data" / "real" / "eu_heldout"
DEFAULT_KIXD_DEV = ROOT / "data" / "real" / "kixd" / "kixd_dev.csv"
DEFAULT_MIXED = ROOT / "data" / "real" / "kixd" / "rl_dev_mixed.csv"


def write_rows(stem: Path, rows: list[dict]) -> None:
    """Write rows as both a CRLF CSV and a JSONL manifest.

    A `source` column rides along when present. `load_local_corpus` reads only
    the audio and text keys and ignores the rest, so the region label costs
    nothing at load time and is what a per-region WER breakdown groups on --
    without it the mixed slice can only report a pooled number, which is the
    one number that cannot show a KIXD-only overfit.
    """
    fields = ["audio", "text"] + (["source"] if "source" in rows[0] else [])
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    with stem.with_suffix(".jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def export_eu(out_dir: Path, split: str, n_rows: int, seed: int) -> list[dict]:
    """Sample `n_rows` of a registry split and write them out as wavs."""
    import random

    dataset = load_split(split)
    if len(dataset) < n_rows:
        raise ValueError(f"split {split!r} has {len(dataset)} rows, "
                         f"fewer than the {n_rows} requested")
    indices = sorted(random.Random(seed).sample(range(len(dataset)), n_rows))

    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, index in enumerate(indices):
        example = dataset[index]
        audio = np.asarray(example["audio"]["array"], dtype=np.float32)
        path = wavs / f"eu_{position:05d}.wav"
        sf.write(path, audio, REAL_SR)
        rows.append({"audio": str(path.resolve()),
                     "text": " ".join(str(example["text"]).split()),
                     "source": "eu"})
    return rows


def read_rows(path: Path, source: str) -> list[dict]:
    with path.open(newline="") as handle:
        return [{"audio": row["audio"], "text": row["text"], "source": source}
                for row in csv.DictReader(handle)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="heldout_tail_check",
                    help="registry split supplying the European half")
    ap.add_argument("--eu-out", type=Path, default=DEFAULT_EU_OUT)
    ap.add_argument("--kixd-dev", type=Path, default=DEFAULT_KIXD_DEV,
                    help="the KIXD half, from scripts/join_kixd_transcripts.py")
    ap.add_argument("--mixed-out", type=Path, default=DEFAULT_MIXED)
    ap.add_argument("--rows", type=int, default=200, help="rows per region")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    eu_rows = export_eu(args.eu_out, args.split, args.rows, args.seed)
    write_rows(args.eu_out / "eu_heldout", eu_rows)

    kixd_rows = read_rows(args.kixd_dev, "kixd")
    if len(kixd_rows) != len(eu_rows):
        raise ValueError(f"regions are unbalanced: {len(kixd_rows)} KIXD rows vs "
                         f"{len(eu_rows)} European; the mix assumes equal counts")
    args.mixed_out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.mixed_out.with_suffix(""), kixd_rows + eu_rows)

    summary = {
        "eu_split": split_spec(args.split).to_dict(),
        "rows_per_region": args.rows,
        "eu_dir": str(args.eu_out.resolve()),
        "kixd_dev": str(args.kixd_dev.resolve()),
        "mixed": str(args.mixed_out.resolve()),
        "mixed_rows": len(kixd_rows) + len(eu_rows),
        "eu_duration_sec": round(sum(
            sf.info(row["audio"]).duration for row in eu_rows), 1),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
