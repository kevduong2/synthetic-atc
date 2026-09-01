"""Break a power-check / RL run's dev_rows.jsonl down by source and capture hour.

    uv run python analyze_dev_rows.py runs/power_check_kixd [--baseline <dir>]

Reports, per cell: aggregate WER recomputed from the rows (a check that the
rows sum to the scored number), WER by `source`, and WER by capture-hour block
parsed out of the KIXD filename.  Then, for every non-base arm, the paired
per-cluster delta against `base` at the same seed, and how many clusters move
in the arm's favour -- direction consistency, which is what a handful of
clusters cannot fake the way a pooled mean can.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STAMP = re.compile(r"_(\d{8})_(\d{2})(\d{2})(\d{2})")

# Hour blocks: the dev slice is one capture day, so the hour is the only real
# acoustic cluster in it (traffic level, controller, receiver drift).
BLOCKS = [("00-03", range(0, 4)), ("04-11", range(4, 12)), ("12-15", range(12, 16)),
          ("16-19", range(16, 20)), ("20-23", range(20, 24))]


def block_of(path: str) -> str:
    m = STAMP.search(Path(path).name)
    if not m:
        return "unknown"
    hh = int(m.group(2))
    for name, rng in BLOCKS:
        if hh in rng:
            return name
    return "unknown"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def wer(rows) -> tuple[float, int, int]:
    e = sum(r["errors"] for r in rows)
    w = sum(r["ref_words"] for r in rows)
    return (e / w if w else float("nan")), e, w


def group(rows, keyfn) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    run = Path(a.run_dir)

    cells = {}
    for f in sorted(run.glob("trials/*/dev_rows.jsonl")):
        cells[f.parent.name] = load(f)
    for f in sorted(run.glob("trials/*/*/dev_rows.jsonl")):
        cells[f.parent.name] = load(f)
    if not cells:
        raise SystemExit(f"no dev_rows.jsonl under {run}/trials")

    base_rows = None
    for extra in (run / "harness" / "baseline_dev_rows.jsonl",
                  run / "harness" / "zero_shot_dev_rows.jsonl"):
        if extra.exists():
            base_rows = load(extra)
            cells = {"zero_shot": base_rows, **cells}
            break

    print(f"{'cell':<16}{'rows':>6}{'wer':>9}{'errs':>8}{'refw':>8}   halluc")
    for name, rows in cells.items():
        w, e, rw = wer(rows)
        h = sum(1 for r in rows if r.get("hallucinated"))
        print(f"{name:<16}{len(rows):>6}{w:>9.4f}{e:>8}{rw:>8}   {h}")

    # by source
    srcs = {s for rows in cells.values() for r in rows for s in [r.get("source")]}
    if len(srcs - {None}) > 1:
        print("\n-- WER by source --")
        for name, rows in cells.items():
            parts = [f"{k}={wer(v)[0]:.4f}(n{len(v)})"
                     for k, v in sorted(group(rows, lambda r: r.get("source")).items())]
            print(f"{name:<16}" + "  ".join(parts))
    else:
        only = (srcs - {None}) or {"unlabelled"}
        print(f"\n-- WER by source: single source {only}, breakdown degenerate --")

    print("\n-- WER by capture-hour block --")
    names = [b for b, _ in BLOCKS]
    print(f"{'cell':<16}" + "".join(f"{b:>12}" for b in names))
    per_cell_block = {}
    for name, rows in cells.items():
        g = group(rows, lambda r: block_of(r["audio"]))
        per_cell_block[name] = {b: wer(g[b])[0] if g.get(b) else float("nan") for b in names}
        counts = {b: len(g.get(b, [])) for b in names}
        print(f"{name:<16}" + "".join(
            f"{per_cell_block[name][b]:>12.4f}" if g.get(b) else f"{'-':>12}" for b in names))
    print(f"{'(n clips)':<16}" + "".join(f"{counts[b]:>12}" for b in names))

    # paired deltas vs the base arm at the same seed
    print("\n-- paired per-block delta vs base (negative = arm has LOWER WER) --")
    arms = {}
    for name in cells:
        if name in ("zero_shot",) or "_s" not in name:
            continue
        arm, seed = name.rsplit("_s", 1)
        arms.setdefault(seed, {})[arm] = name
    for seed, by_arm in sorted(arms.items()):
        if "base" not in by_arm:
            continue
        b = per_cell_block[by_arm["base"]]
        for arm, cell in sorted(by_arm.items()):
            if arm == "base":
                continue
            d = {k: per_cell_block[cell][k] - b[k] for k in names
                 if per_cell_block[cell][k] == per_cell_block[cell][k] and b[k] == b[k]}
            neg = sum(1 for v in d.values() if v < 0)
            print(f"seed {seed} {arm:<10}" + "".join(f"{d.get(k, float('nan')):>+12.4f}" for k in names)
                  + f"   {neg}/{len(d)} blocks lower")


if __name__ == "__main__":
    main()
