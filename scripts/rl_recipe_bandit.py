#!/usr/bin/env python
"""L3: choose which synthetic recipe to generate next, by Thompson sampling.

Each pull generates a batch from one recipe bucket, transcribes it with a
frozen teacher and with the current student, and keeps the samples inside the
hardness window (`WER_teacher < tau1`, `tau2 < WER_student < tau3`) in a
persistent `selected/` buffer.  The in-window fraction is the Bernoulli reward
the bucket's Beta posterior is updated with.  Student hardness only ever
*selects data* here; it never reaches a generator objective (D5).

Every `--counterfactual-every` pulls (and once at the end) the proxy is
recalibrated for real: the same frozen init is fine-tuned on a sample of the
selected buffer and on a freshly generated uniform mixture, both scored on the
real reward-validation slice.  That delta is the number P4c's exit bar reads.

The run is crash-idempotent: rerun the same command after a kill and it
resumes from `pulls.jsonl` plus `state.json`.

Examples:
  # smoke: four pulls, no fine-tuning
  uv run python scripts/rl_recipe_bandit.py --out runs/smoke_bandit \\
      --pulls 4 --n-batch 12 --counterfactual-every 0

  # a real run: ~30 pulls with two recalibration rounds
  uv run python scripts/rl_recipe_bandit.py --out runs/bandit_v1 \\
      --pulls 30 --n-batch 60 --counterfactual-every 15 --cf-m 150 --cf-steps 250

  # point the window at a fine-tuned student between rounds
  uv run python scripts/rl_recipe_bandit.py --out runs/bandit_v1 --pulls 60 \\
      --student runs/sft_a3/checkpoint
"""

import argparse
from pathlib import Path

import yaml

from atcgen.rl.bandit import (
    GEN_SEED,
    REWARD_VAL,
    AsrCounterfactual,
    AsrPullEngine,
    HardnessWindow,
    RecipeBandit,
    format_posteriors,
    format_pull,
)
from atcgen.rl.recipes import RECIPES, check_recipe

DEFAULT_CONFIG = "configs/mode1_matched.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory (resumed if it exists)")
    ap.add_argument("--base-config", default=DEFAULT_CONFIG,
                    help=f"profile the recipes mutate (default: {DEFAULT_CONFIG})")
    ap.add_argument("--pulls", type=int, default=30, help="total pulls to reach")
    ap.add_argument("--n-batch", type=int, default=60, help="clips generated per pull")
    ap.add_argument("--student", default="openai/whisper-tiny.en",
                    help="checkpoint dir or model id the window is defined against")
    ap.add_argument("--teacher", default="openai/whisper-base.en",
                    help="frozen label-trust model; never the student (D4)")
    ap.add_argument("--tau1", type=float, default=0.8,
                    help="teacher WER above this: label untrustworthy, sample dropped. "
                         "Set above the teacher's WER on undegraded audio (0.24 median "
                         "for whisper-base.en) so it fires on degradation, not on the "
                         "teacher's out-of-domain gap")
    ap.add_argument("--tau2", type=float, default=0.4,
                    help="student WER below this: too easy, sample goes to spillover. "
                         "Re-calibrate whenever --student changes: put it at the "
                         "student's median WER on undegraded audio")
    ap.add_argument("--tau3", type=float, default=1.2,
                    help="student WER above this: hopeless, sample goes to spillover")
    ap.add_argument("--counterfactual-every", type=int, default=8,
                    help="recalibration cadence in pulls; 0 disables it entirely")
    ap.add_argument("--cf-steps", type=int, default=300,
                    help="fine-tune steps per counterfactual arm")
    ap.add_argument("--cf-m", type=int, default=150, help="clips per counterfactual arm")
    ap.add_argument("--cf-eval-n", type=int, default=400,
                    help=f"real clips scored per counterfactual arm, taken from "
                         f"the {REWARD_VAL!r} split of atcgen.dataset.splits")
    ap.add_argument("--cf-init", default=None,
                    help="frozen init for both counterfactual arms (default: --student)")
    ap.add_argument("--seed", type=int, default=GEN_SEED,
                    help="base seed for Thompson draws and per-pull generation")
    ap.add_argument("--noise-only-frac", type=float, default=0.0,
                    help="noise-only fraction forced onto every pull; unscoreable "
                         "clips cannot enter the window, so the default is 0")
    ap.add_argument("--asr-batch", type=int, default=16, help="decode batch size")
    ap.add_argument("--table-every", type=int, default=5,
                    help="print the posterior table every N pulls")
    ap.add_argument("--device", default=None, help="torch device (default: autodetect)")
    args = ap.parse_args(argv)

    with open(args.base_config, encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}

    window = HardnessWindow(tau1=args.tau1, tau2=args.tau2, tau3=args.tau3)

    # Capped domain randomization is a report, not a gate (§4.3): a bucket that
    # runs past the measured envelope is worth knowing about, not worth failing.
    for name, recipe in sorted(RECIPES.items()):
        for finding in check_recipe(recipe, base_config):
            print(f"envelope warning [{name}]: {finding}")

    engine = AsrPullEngine(base_config, teacher=args.teacher, student=args.student,
                           device=args.device, asr_batch=args.asr_batch,
                           noise_only_frac=args.noise_only_frac)

    counterfactual = None
    if args.counterfactual_every > 0:
        counterfactual = AsrCounterfactual(
            engine, init=args.cf_init or args.student, window=window,
            ft_steps=args.cf_steps, eval_n=args.cf_eval_n, seed=args.seed)

    out_dir = Path(args.out)
    bandit = RecipeBandit(
        out_dir, engine, window=window, n_batch=args.n_batch, seed=args.seed,
        counterfactual=counterfactual, counterfactual_every=args.counterfactual_every,
        cf_m=args.cf_m, on_pull=lambda row: _report(row, bandit, args.table_every),
        on_counterfactual=lambda row: print("\n" + _format_counterfactual(row) + "\n"))

    print(f"{len(RECIPES)} recipes, {args.pulls} pulls x {args.n_batch} clips -> {out_dir}")
    print(f"window: teacher < {args.tau1}, {args.tau2} < student < {args.tau3}"
          f"  |  teacher {args.teacher}  student {args.student}  device {engine.device}")
    if bandit.pulls_done:
        print(f"resuming at pull {bandit.pulls_done}")

    bandit.run(args.pulls)

    print("\n" + format_posteriors(bandit))
    best = max(bandit.posteriors.arms, key=bandit.posteriors.mean)
    print(f"\nbest bucket: {best} (posterior mean {bandit.posteriors.mean(best):.3f})")
    print(f"selected buffer: {out_dir / 'selected' / 'manifest.jsonl'}")
    return bandit


def _report(row, bandit, table_every: int) -> None:
    print(format_pull(row))
    if table_every > 0 and row["pull"] % table_every == table_every - 1:
        print("\n" + format_posteriors(bandit) + "\n")


def _format_counterfactual(row) -> str:
    if row.get("status") != "ok":
        return (f"counterfactual {row['round']} after pull {row['after_pull']}: "
                f"{row.get('status')} ({row.get('reason')})")
    return (f"counterfactual {row['round']} after pull {row['after_pull']}:"
            f" n={row['n']}/arm  init {row['wer_init']:.4f}"
            f"  selected {row['wer_selected']:.4f}  uniform {row['wer_uniform']:.4f}"
            f"  delta {row['delta_wer_selected_vs_uniform']:+.4f}"
            f"  ({row['wall_time_sec'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
