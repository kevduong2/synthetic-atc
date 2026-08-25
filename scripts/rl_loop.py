#!/usr/bin/env python
"""Search generator-config knobs against downstream ASR reward.

Each trial generates a fresh synthetic batch from a candidate config, runs a
short whisper-tiny fine-tune on it, and scores it by how much the dev WER on
real ATC audio drops.  That costs roughly ten minutes, so the useful budget is
`--iterations x --pop-size` in the tens; `cem` is the default optimizer because
it only uses the ranking of the rewards, which is the safer assumption at that
sample size.  Run `--optimizer random` once as a control.

Trial 0 is the base profile itself (disable with --no-seed-default), so every
later reward can be read against the hand-tuned config measured by the same
harness on the same day.  The run is crash-idempotent: rerun the same command
after a kill and it resumes from `trials.jsonl` plus `optimizer_state.json`.

Examples:
  uv run python scripts/rl_loop.py --out runs/rl_v1 --iterations 4 --pop-size 4
  uv run python scripts/rl_loop.py --out runs/rl_ctl --optimizer random \\
      --iterations 4 --pop-size 4
  uv run python scripts/rl_loop.py --out runs/rl_v2 --optimizer reinforce \\
      --n-synth 400 --ft-steps 500 --dev-indices 0:400 --device cuda
"""

import argparse
from pathlib import Path

import yaml

from atcgen.rl.loop import format_trial, run_loop
from atcgen.rl.policy import CrossEntropyMethod, RandomSearch, ReinforceGaussian
from atcgen.rl.reward import TrueRewardHarness
from atcgen.rl.space import default_atc_space

DEFAULT_CONFIG = "configs/mode1_matched.yaml"


def _indices(text: str) -> tuple[int, int]:
    """Parse a `lo:hi` dev-set slice."""
    lo, _, hi = text.partition(":")
    if not hi:
        raise argparse.ArgumentTypeError("dev indices must look like 0:200")
    return int(lo), int(hi)


def _make_optimizer(name, dim, seed, init_mean):
    """Seed both learners at the hand-tuned config rather than the cube's centre."""
    if name == "random":
        return RandomSearch(dim, seed=seed)
    if name == "reinforce":
        return ReinforceGaussian(dim, seed=seed, init_mean=init_mean)
    return CrossEntropyMethod(dim, seed=seed, init_mean=init_mean)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", default=DEFAULT_CONFIG,
                    help=f"profile the search mutates (default: {DEFAULT_CONFIG})")
    ap.add_argument("--out", required=True, help="run directory (resumed if it exists)")
    ap.add_argument("--optimizer", choices=["cem", "reinforce", "random"], default="cem")
    ap.add_argument("--iterations", type=int, default=4, help="batches to run")
    ap.add_argument("--pop-size", type=int, default=4, help="candidates per batch")
    ap.add_argument("--seed", type=int, default=0, help="optimizer sampling seed")
    ap.add_argument("--no-seed-default", action="store_true",
                    help="skip the trial-0 evaluation of the base profile")
    ap.add_argument("--no-resume", action="store_true",
                    help="restart numbering and discard the existing trial log")
    ap.add_argument("--n-synth", type=int, default=200,
                    help="synthetic clips generated per trial")
    ap.add_argument("--ft-steps", type=int, default=300, help="fine-tune steps per trial")
    ap.add_argument("--dev-indices", type=_indices, default=(0, 200),
                    help="real dev-set slice as lo:hi (default: 0:200)")
    ap.add_argument("--text-pool", type=int, default=400,
                    help="utterances in the shared text pool; fixed across trials so "
                         "candidates differ only by channel/voice knobs")
    ap.add_argument("--device", default=None, help="torch device (default: autodetect)")
    args = ap.parse_args(argv)

    with open(args.base_config, encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}

    space = default_atc_space()
    default_vector = space.default_vector(base_config)
    default_overrides = space.describe(default_vector)
    optimizer = _make_optimizer(args.optimizer, space.dim, args.seed, default_vector)

    out_dir = Path(args.out)
    reward_fn = TrueRewardHarness(
        out_dir / "harness",
        dev_indices=args.dev_indices,
        text_pool_size=args.text_pool,
        n_synth=args.n_synth,
        ft_steps=args.ft_steps,
        device=args.device,
    )

    print(f"{args.optimizer} over {space.dim} knobs, "
          f"{args.iterations} x {args.pop_size} trials -> {out_dir}")
    trials = run_loop(
        space, optimizer, reward_fn, base_config, out_dir,
        iterations=args.iterations,
        pop_size=args.pop_size,
        seed_default_first=not args.no_seed_default,
        resume=not args.no_resume,
        on_trial=lambda trial: print(format_trial(trial, space, default_overrides)),
    )

    best = max((trial for trial in trials if not trial.result.proxy),
               key=lambda trial: trial.result.reward, default=None)
    if best is not None:
        print(f"\nbest: trial {best.index}, reward {best.result.reward:+.4f}")
        print(f"      {out_dir / 'best_config.yaml'}")
    return trials


if __name__ == "__main__":
    main()
