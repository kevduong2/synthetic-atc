"""Build capture-blocked channel folds before deriving channel artifacts."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .local_corpus import parse_station

_TIMESTAMP_RE = re.compile(r"^(.+)_(\d{8})_(\d{6})$")


def parse_timestamp(clip_id: str) -> datetime | None:
    """Read capture time from an ID so nearby clips cannot cross folds."""
    match = _TIMESTAMP_RE.match(Path(clip_id).stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()[1:]), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _initial_blocks(rows: list[dict], gap_min: float) -> list[list[dict]]:
    parsed = sorted(
        (row for row in rows if row["_timestamp"] is not None),
        key=lambda row: (row["_timestamp"], row["clip_id"]),
    )
    blocks: list[list[dict]] = []
    gap_seconds = gap_min * 60.0
    for row in parsed:
        if (not blocks or
                (row["_timestamp"] - blocks[-1][-1]["_timestamp"]).total_seconds()
                > gap_seconds):
            blocks.append([])
        blocks[-1].append(row)
    for row in sorted(
            (item for item in rows if item["_timestamp"] is None),
            key=lambda item: item["clip_id"]):
        blocks.append([row])
    return blocks


def _pseudo_split(block: list[dict]) -> list[list[dict]]:
    """Create a usable held-out block when capture metadata has only one."""
    gaps = [
        (block[index]["_timestamp"] - block[index - 1]["_timestamp"]).total_seconds()
        for index in range(1, len(block))
    ]
    cut = max(range(1, len(block)), key=lambda index: gaps[index - 1])
    return [block[:cut], block[cut:]]


def _station_blocks(rows: list[dict], gap_min: float) -> list[dict]:
    blocks = _initial_blocks(rows, gap_min)
    pseudo = False
    if len(blocks) < 2 and len(rows) >= 2:
        blocks = _pseudo_split(blocks[0])
        pseudo = True
    return [{"rows": block, "pseudo_split": pseudo} for block in blocks]


def _closest_subset(blocks: list[dict], target: int,
                    rng: random.Random) -> set[str]:
    """Block ids whose clip total lands closest to `target`, never all blocks.

    Greedily adding shuffled blocks overshoots badly when sessions are lumpy —
    one 14-clip block against a 4-clip target would put over half a station in
    validation and starve Domain B.  Blocks per station number in the tens, so
    the subset space is enumerated outright (capped; greedy fallback beyond
    it).  Ties prefer fewer validation blocks, then a seeded draw.
    """
    if target <= 0 or len(blocks) < 2:
        return set()
    if len(blocks) > 16:                # 2^16 subsets; never hit at 99 clips
        order = list(blocks)
        rng.shuffle(order)
        chosen: set[str] = set()
        total = 0
        for block in order[:-1]:
            if total >= target:
                break
            chosen.add(block["block_id"])
            total += len(block["rows"])
        return chosen
    best: tuple[int, int, float] | None = None
    best_ids: set[str] = set()
    for mask in range(1, (1 << len(blocks)) - 1):
        ids = {block["block_id"] for index, block in enumerate(blocks)
               if mask >> index & 1}
        total = sum(len(block["rows"]) for index, block in enumerate(blocks)
                    if mask >> index & 1)
        key = (abs(total - target), len(ids), rng.random())
        if best is None or key < best:
            best, best_ids = key, ids
    return best_ids


def build_channel_splits(
    corpus: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.15,
    gap_min: float = 15.0,
    seed: int = 0,
) -> dict:
    """Write station/session-disjoint development folds and their audit trail."""
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    if gap_min <= 0.0:
        raise ValueError("gap_min must be positive")
    source = Path(corpus)
    if not source.is_file():
        raise ValueError(f"corpus does not exist: {source}")

    rows = [json.loads(line) for line in source.read_text().splitlines()
            if line.strip()]
    by_station: dict[str, list[dict]] = defaultdict(list)
    unparsed: list[str] = []
    for index, row in enumerate(rows):
        clip_id = row["clip_id"]
        timestamp = parse_timestamp(clip_id)
        station = row.get("station") or parse_station(clip_id)
        row["station"] = station
        row["_index"] = index
        row["_timestamp"] = timestamp
        row["_pseudo_split"] = False
        if timestamp is None:
            unparsed.append(clip_id)
        by_station[station].append(row)

    rng = random.Random(seed)
    station_stats: dict[str, dict] = {}
    single_clip: list[str] = []
    for station in sorted(by_station):
        station_rows = by_station[station]
        blocks = _station_blocks(station_rows, gap_min)
        for index, block in enumerate(blocks):
            block["block_id"] = f"{station}/b{index:02d}"
            for row in block["rows"]:
                row["block_id"] = block["block_id"]
                row["_pseudo_split"] = block["pseudo_split"]

        target = round(val_frac * len(station_rows))
        if val_frac > 0.0 and len(station_rows) >= 2:
            target = max(1, target)
        selected = _closest_subset(blocks, target, rng)
        val_clips = sum(len(block["rows"]) for block in blocks
                        if block["block_id"] in selected)
        if len(station_rows) == 1:
            single_clip.append(station)
        for block in blocks:
            split = ("channel_val" if block["block_id"] in selected
                     else "channel_train")
            for row in block["rows"]:
                row["split"] = split
        station_stats[station] = {
            "blocks": len(blocks),
            "clips": len(station_rows),
            "val_clips": val_clips,
            "val_blocks": len(selected),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w") as handle:
        for row in sorted(rows, key=lambda item: item["_index"]):
            output = {key: value for key, value in row.items()
                      if not key.startswith("_")}
            path = Path(output["path"])
            if not path.is_absolute():
                path = source.parent / path
            output["path"] = str(path.resolve())
            output["timestamp"] = (
                row["_timestamp"].isoformat() if row["_timestamp"] else None)
            handle.write(json.dumps(output) + "\n")

    with (out / "split_manifest.jsonl").open("w") as handle:
        for row in sorted(rows, key=lambda item: item["_index"]):
            handle.write(json.dumps({
                "clip_id": row["clip_id"],
                "station": row["station"],
                "timestamp": (row["_timestamp"].isoformat()
                              if row["_timestamp"] else None),
                "block_id": row["block_id"],
                "split": row["split"],
                "pseudo_split": row["_pseudo_split"],
            }) + "\n")

    totals = {
        "blocks": sum(item["blocks"] for item in station_stats.values()),
        "clips": len(rows),
        "val_clips": sum(item["val_clips"] for item in station_stats.values()),
        "val_blocks": sum(item["val_blocks"] for item in station_stats.values()),
    }
    stats = {
        "stations": station_stats,
        "totals": totals,
        "args": {"val_frac": val_frac, "gap_min": gap_min, "seed": seed},
        "warnings": {
            "single_clip_stations": single_clip,
            "unparsed_timestamps": unparsed,
        },
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--gap-min", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    stats = build_channel_splits(
        args.corpus, args.out, args.val_frac, args.gap_min, args.seed)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
