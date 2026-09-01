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

A production set is two renders off one profile (see docs/cli-reference.md):
the scheduled speech, and the noise-only hallucination control rendered by the
same channel.

  uv run python scripts/generate_dataset.py --config configs/mode1_matched.yaml \\
      --n-samples 155776 --out runs/train_v1 --seed 7 \\
      --text sequential:data/text/scenes_v2.0.1_2view.jsonl \\
      --set dataset.noise_only_frac=0
  uv run python scripts/generate_dataset.py --config configs/mode1_matched.yaml \\
      --n-samples 4800 --out runs/train_v1_noise --seed 8 \\
      --set dataset.noise_only_frac=1.0
"""

import argparse

import yaml

from atcgen.config import load_config
from atcgen.dataset.build import DEFAULT_CONFIG, build_dataset


def parse_set(items: list[str]) -> dict:
    """`--set a.b=1.0` pairs as a dot-path override map for `load_config`.

    Values go through the YAML scalar parser, so `1.0` is a float, `0` an int
    and `true` a bool -- the config validators are strict about which.
    """
    overrides = {}
    for item in items:
        path, sep, value = item.partition("=")
        if not sep or not path.strip():
            raise ValueError(f"--set needs 'dotted.path=value': {item!r}")
        overrides[path.strip()] = yaml.safe_load(value)
    return overrides


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="generator profile YAML (default: configs/mode1_default.yaml)")
    ap.add_argument("--n-samples", type=int, required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=None, help="overrides the profile's seed")
    ap.add_argument("--text", default=None,
                    help="'grammar' for the built-in phraseology generator, a JSONL "
                         "path of {'spoken','transcript','role','kind','weight','category'} "
                         "records from any external script (sampled with replacement), "
                         "or 'sequential:<path>' to render that file in order exactly "
                         "once -- see scripts/expand_text_views.py")
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                    dest="overrides",
                    help="override any config field by dot-path, e.g. "
                         "--set dataset.noise_only_frac=1.0; repeat for more. "
                         "Prefer this over forking a profile so both renders of "
                         "a set share one channel definition")
    args = ap.parse_args(argv)

    overrides = parse_set(args.overrides)
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = load_config(args.config, overrides)
    manifest = build_dataset(config, args.out, args.n_samples, text_source=args.text)
    print(f"wrote {manifest}")
    print(f"      {manifest.parent / 'stats.json'}")
    return manifest


if __name__ == "__main__":
    main()
