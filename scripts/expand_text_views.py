#!/usr/bin/env python
"""Expand a text JSONL into N deterministic views per utterance.

    uv run python scripts/expand_text_views.py \
        --text data/text/scenes_v2.0.1.jsonl \
        --out data/text/scenes_v2.0.1_2view.jsonl

    uv run python scripts/generate_dataset.py --config configs/mode1_matched.yaml \
        --n-samples 155776 --out runs/train_v1 \
        --text sequential:data/text/scenes_v2.0.1_2view.jsonl

Sampling with replacement is the wrong tool for a production render. Drawing
100k utterances from a 77,888-line pool covers only about 56k distinct texts:
roughly a quarter of the corpus is never spoken while other lines are rendered
four or five times, and which quarter is lost depends on the seed. Writing the
schedule out instead makes coverage exact — every text appears exactly
`--views` times, under independent voice, speed and channel draws, because the
builder redraws those per sample regardless of the text.

Each output line carries two passthrough fields that survive onto the manifest
row as their own columns:

* `base_id` — stable per *source line*, shared by that line's views, so the
  views of one utterance can be paired after the fact (the same handle
  `scripts/paired_support.py` intersects on).
* `view_index` — 0..N-1 within a base_id.

`base_id` is derived from the source line's position, not a hash of its text,
because the corpus genuinely repeats itself — readbacks especially — and
hashing would collapse distinct scheduled lines into one id and break the
pairing.

The file is shuffled as a whole (seeded), so a run truncated part-way still
sees a representative mix of airports and categories rather than the corpus in
scene order, and the two views of one text land far apart.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEXT = ROOT / "data" / "text" / "scenes_v2.0.1.jsonl"
DEFAULT_OUT = ROOT / "data" / "text" / "scenes_v2.0.1_2view.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def expand(records: list[dict], views: int, seed: int) -> list[dict]:
    """`views` copies of every record, tagged and shuffled as one pool."""
    width = max(6, len(str(len(records) - 1)))
    out = []
    for index, record in enumerate(records):
        for view in range(views):
            out.append({**record, "base_id": f"t{index:0{width}d}",
                        "view_index": view})
    random.Random(seed).shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", type=Path, default=DEFAULT_TEXT,
                    help="source JSONL, one utterance per line")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--views", type=int, default=2,
                    help="renders scheduled per source utterance")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed")
    args = ap.parse_args(argv)
    if args.views < 1:
        ap.error("--views must be at least 1")

    records = read_jsonl(args.text)
    expanded = expand(records, args.views, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in expanded:
            handle.write(json.dumps(record) + "\n")

    kinds = Counter(record.get("kind", "external") for record in expanded)
    summary = {
        "source": str(args.text.resolve()),
        "source_utterances": len(records),
        "views": args.views,
        "lines": len(expanded),
        "distinct_base_ids": len({record["base_id"] for record in expanded}),
        "seed": args.seed,
        "kinds": dict(kinds.most_common()),
        "out": str(args.out.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
