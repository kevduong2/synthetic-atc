#!/usr/bin/env python
"""Paired common-support gating across channel views (FastCUT plan §8.2).

Equal-count top-up after per-view gating silently compares different content
distributions: pipeline-dependent gate failures change which transcripts
survive on each side.  The primary comparisons therefore retain a `base_id`
only when EVERY compared view's row is gate-selected, so all arms train on the
identical content set and differ only in channel rendering.

    uv run python scripts/paired_support.py \
        --view runs/channel_matrix_fastcut/views/procedural_matched \
        --view runs/channel_matrix_fastcut/views/calibrated_dsp \
        --tiers gold silver adversarial --adversarial-cap 0.05

Writes `manifest_paired.jsonl` into each view directory (rows in base order)
and prints per-view yield plus the intersection size.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atcgen.gate.gate import select_tiers  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", action="append", required=True,
                    help="view directory holding manifest_gated.jsonl; repeat")
    ap.add_argument("--tiers", nargs="+", default=("gold", "silver", "adversarial"))
    ap.add_argument("--adversarial-cap", type=float, default=0.05)
    args = ap.parse_args(argv)
    if len(args.view) < 2:
        ap.error("need at least two --view directories to pair")

    selected: dict[str, dict[str, dict]] = {}
    yields: dict[str, dict] = {}
    for view in args.view:
        gated = Path(view) / "manifest_gated.jsonl"
        rows = select_tiers(gated, tuple(args.tiers), args.adversarial_cap)
        by_base = {row["base_id"]: row for row in rows}
        if len(by_base) != len(rows):
            raise ValueError(f"duplicate base_id in {gated}")
        total = sum(1 for line in gated.open() if line.strip())
        selected[view] = by_base
        yields[view] = {"total": total, "selected": len(rows),
                        "yield": round(len(rows) / max(total, 1), 4)}

    common = set.intersection(*(set(v) for v in selected.values()))
    report = {"views": yields, "common_support": len(common),
              "tiers": list(args.tiers), "adversarial_cap": args.adversarial_cap}
    for view, by_base in selected.items():
        out = Path(view) / "manifest_paired.jsonl"
        kept = sorted(common)
        with out.open("w") as handle:
            for base_id in kept:
                handle.write(json.dumps(by_base[base_id]) + "\n")
        report["views"][view]["paired_manifest"] = str(out)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
