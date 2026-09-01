#!/usr/bin/env python
"""Split a JSONL text corpus into N round-robin shards for a resumable render.

`scripts/generate_dataset.py` is not resumable and writes `stats.json` only at
the end, so the 155,776-clip production render is sharded: one shard per
`--out`, each with its own seed, exported together afterwards.  Round-robin
(line i goes to shard i mod N) keeps every shard airport-mixed even though the
source file is grouped by airport, so a lost shard costs an even slice rather
than one airport.

    uv run python scripts/lab/shard_text.py data/text/scenes_v2.0.1_2view.jsonl --n 4

writes `data/text/scenes_v2.0.1_2view.shard1of4.jsonl` … `shard4of4.jsonl`
and prints each shard's line count (that count is the `--n-samples` to render
with `sequential:`, which refuses anything larger).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def shard_paths(src: Path, n: int) -> list[Path]:
    return [src.with_name(f"{src.stem}.shard{i + 1}of{n}{src.suffix}") for i in range(n)]


def shard_file(src: Path, n: int) -> list[tuple[Path, int]]:
    if n < 1:
        raise ValueError("--n must be >= 1")
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    outs = shard_paths(src, n)
    buckets: list[list[str]] = [[] for _ in range(n)]
    for i, line in enumerate(lines):
        buckets[i % n].append(line)
    result = []
    for path, bucket in zip(outs, buckets):
        path.write_text("".join(ln + "\n" for ln in bucket), encoding="utf-8")
        result.append((path, len(bucket)))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="JSONL file, one utterance per line")
    ap.add_argument("--n", type=int, required=True, help="number of shards")
    args = ap.parse_args(argv)
    total = 0
    for path, count in shard_file(Path(args.src), args.n):
        print(f"{count:>8}  {path}")
        total += count
    print(f"{total:>8}  total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
