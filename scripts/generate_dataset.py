#!/usr/bin/env python
"""CLI for generating a synthetic ATC dataset.

Everything is declared in a YAML profile under configs/; only the four knobs
that change per run are flags.

Examples:
  uv run python scripts/generate_dataset.py --n-samples 25 --out runs/smoke
  uv run python scripts/generate_dataset.py --config configs/mode1_wide.yaml \\
      --n-samples 5000 --out data/train_v1 --seed 7
  uv run python scripts/generate_dataset.py --text my_team_transcripts.jsonl \\
      --n-samples 500 --out data/team_v1
"""

import argparse

from atcgen.config import load_config
from atcgen.dataset.build import DEFAULT_CONFIG, build_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="generator profile YAML (default: configs/mode1_default.yaml)")
    ap.add_argument("--n-samples", type=int, required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=None, help="overrides the profile's seed")
    ap.add_argument("--text", default=None,
                    help="'grammar' for the built-in phraseology generator, or a JSONL "
                         "path of {'spoken','transcript','role','kind','weight','category'} "
                         "records from any external script")
    args = ap.parse_args(argv)

    overrides = {} if args.seed is None else {"seed": args.seed}
    config = load_config(args.config, overrides)
    manifest = build_dataset(config, args.out, args.n_samples, text_source=args.text)
    print(f"wrote {manifest}")
    print(f"      {manifest.parent / 'stats.json'}")
    return manifest


if __name__ == "__main__":
    main()
