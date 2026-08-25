#!/usr/bin/env python
"""CLI for the verification gate (research-findings §4.4, D8).

Runs the frozen teacher pool over a built dataset and writes
`manifest_gated.jsonl` (every row, plus `tier` and a `gate` blob) and
`gate_stats.json` next to it.  Nothing is deleted — downstream selects by
tier, so a gate pass is re-runnable with different thresholds.

Examples:
  uv run python scripts/gate_dataset.py --dataset runs/smoke_gate
  uv run python scripts/gate_dataset.py --dataset data/train_v1 --batch 16
  uv run python scripts/gate_dataset.py --dataset data/train_v1 \\
      --out runs/gate_strict --gold-wer 0.15 --device cpu
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from atcgen.gate import GateConfig, default_teachers, gate_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="built dataset directory (holds manifest.jsonl and wavs/)")
    ap.add_argument("--out", default=None,
                    help="where to write manifest_gated.jsonl (default: --dataset)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8, help="clips per teacher call")
    ap.add_argument("--device", default=None, help="mps | cpu | cuda (default: auto)")
    ap.add_argument("--quiet", action="store_true", help="no progress bar")
    defaults = GateConfig()
    for item in fields(GateConfig):
        flag = "--" + item.name.replace("_", "-")
        current = getattr(defaults, item.name)
        if isinstance(current, bool):
            # `type=bool` would read "--flag False" as True
            ap.add_argument(flag, action=argparse.BooleanOptionalAction,
                            default=None, help=f"GateConfig.{item.name}")
        else:
            ap.add_argument(flag, type=type(current), default=None,
                            help=f"GateConfig.{item.name}")
    args = ap.parse_args(argv)

    overrides = {item.name: getattr(args, item.name)
                 for item in fields(GateConfig)
                 if getattr(args, item.name) is not None}
    config = GateConfig(**overrides)

    stats = gate_dataset(args.dataset, args.out, teachers=default_teachers(args.device),
                         config=config, max_samples=args.max_samples,
                         batch_size=args.batch, progress=not args.quiet)

    print(f"wrote {stats['manifest']}")
    print(f"      {Path(stats['manifest']).parent / 'gate_stats.json'}")
    print(json.dumps({"tiers": stats["tiers"],
                      "tier_fractions": stats["tier_fractions"],
                      "rejection_reasons": stats["rejection_reasons"],
                      "best_teacher_wer": stats["best_teacher_wer"],
                      "throughput": stats["throughput"]}, indent=2))
    return stats


if __name__ == "__main__":
    main()
