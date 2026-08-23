#!/usr/bin/env python
"""CLI for generating a synthetic ATC dataset.

Examples:
  uv run python scripts/generate_dataset.py --n-samples 10 --out data/smoke
  uv run python scripts/generate_dataset.py --n-samples 5000 --channel mix \\
      --gan-checkpoint runs/cyclegan/G_ab_best.pt --out data/train_v1
  uv run python scripts/generate_dataset.py --text my_team_transcripts.jsonl ...
"""

import argparse

from atcgen.dataset.build import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-samples", type=int, required=True)
    ap.add_argument("--channel", choices=["dsp", "gan", "mix", "clean"], default="dsp")
    ap.add_argument("--gan-checkpoint", default=None, help="CycleGAN G_ab checkpoint (required for gan/mix)")
    ap.add_argument("--text", default="grammar",
                    help="'grammar' for built-in phraseology generator, or path to a JSONL file "
                         "of {'spoken':..., 'transcript':...} records from any external script")
    ap.add_argument("--noise-dir", default=None,
                    help="folder of real noise-bed wavs (atcgen.dataset.real_atc.export_noise_beds); "
                         "mixed in instead of synthetic static ~60%% of the time")
    ap.add_argument("--noise-only-frac", type=float, default=0.03,
                    help="fraction of noise-only samples with empty transcript")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.channel in ("gan", "mix") and not args.gan_checkpoint:
        ap.error("--gan-checkpoint is required with --channel gan|mix")

    manifest = build_dataset(
        out_dir=args.out,
        n_samples=args.n_samples,
        text_source=args.text,
        channel=args.channel,
        gan_checkpoint=args.gan_checkpoint,
        seed=args.seed,
        noise_dir=args.noise_dir,
        noise_only_frac=args.noise_only_frac,
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
