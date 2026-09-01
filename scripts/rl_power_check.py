#!/usr/bin/env python
"""Measure whether the reward can tell known-good from known-bad configs.

Before spending two hours on a CEM search, spend forty minutes finding out
whether the reward resolves anything at all.  This runs a fixed, named list of
configs through the same `TrueRewardHarness` the search uses, at several seeds
each, and reports the between-arm gap against the within-arm (seed-to-seed)
spread.  If a deliberately awful config does not score clearly worse than the
hand-tuned one, the search that follows would be fitting noise, and the right
response is to fix the reward -- more dev rows, more synthetic clips, more
fine-tune steps -- not to run it anyway.

The arms:

*   **base** -- the profile exactly as it ships. The reference point.
*   **aug_off** -- voice augmentation disabled and the speaking rate pinned to
    1.0. A plausible, defensible config, not a broken one: it should land near
    `base`, and a large gap here is itself informative.
*   **degraded** -- deliberately awful: an SNR range of 0-6 dB, dropouts on
    40% of clips, maximum AM distortion. If the reward cannot separate this
    from `base`, it cannot separate anything.

`aug_off` moves four things at once -- pitch, tempo, eq tilt and the speaking
rate -- so a gap there says "the talker matters" without saying which part of
the talker.  Three further arms split it, and together they partition it:

*   **speed_fixed** -- rate pinned to 1.0, every voice augmentation left on.
*   **voiceaug_off** -- pitch, tempo and eq tilt off, the rate range left alone.
    `speed_fixed` and `voiceaug_off` are complementary halves of `aug_off`, so
    whichever reproduces its gap is the half that carries the effect.
*   **pitch_off** -- pitch alone, the one augmentation with a measured fidelity
    cost (configs/mode2_fastcut.yaml's header records a 2.6x WavLM KID
    regression from pitch shifting).  If it buys no WER, dropping it is free.

`degraded` addresses `channel.chain` steps, so the arms below require a Mode 1
(procedural) base profile.  Seeds vary both the generator draw and the
fine-tune batch order, which is what makes the spread an honest estimate of
trial-to-trial noise; the dev slice, its zero-shot baseline and the shared text
pool are computed once and reused across every cell.

The run is crash-idempotent: rerun the same command and finished cells are
read back from `results.jsonl` rather than recomputed.

Examples:
  uv run python scripts/rl_power_check.py --out runs/power_check \\
      --base-config configs/mode1_matched_kixd.yaml \\
      --dev-corpus data/real/kixd/rl_dev_mixed.csv --dev-indices 0:400
  uv run python scripts/rl_power_check.py --out runs/power_check_fast \\
      --arms base,degraded --seeds 0 --n-synth 100 --ft-steps 150
"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import yaml

from atcgen.rl.reward import (
    DEFAULT_DEV_CORPUS,
    DEFAULT_DEV_SPLIT,
    GEN_SEED,
    TrueRewardHarness,
)
from atcgen.rl.space import (
    chain_param_knob,
    chain_prob_knob,
    dist_bound_knob,
    dist_prob_knob,
)

DEFAULT_CONFIG = "configs/mode1_matched_kixd.yaml"

#: Each arm is a list of (knob, concrete value) mutations against the base
#: profile. Knobs are reused from the search space rather than dotted-path
#: overrides because the interesting leaves live inside `channel.chain`, which
#: is a list addressed by primitive name -- `_chain_step` already does that,
#: and a knob's declared lo/hi are irrelevant here since `apply` takes the
#: concrete value.
ARMS = {
    "base": [],
    "aug_off": [
        (dist_prob_knob("pitch", "voice_augment.pitch_semitones"), 0.0),
        (dist_prob_knob("tempo", "voice_augment.tempo"), 0.0),
        (dist_prob_knob("eq_tilt", "voice_augment.eq_tilt_db"), 0.0),
        (dist_bound_knob("speed_lo", "tts.speed", 0, 0.5, 2.0), 1.0),
        (dist_bound_knob("speed_hi", "tts.speed", 1, 0.5, 2.0), 1.0),
    ],
    # The two complementary halves of `aug_off`, plus pitch on its own.
    "speed_fixed": [
        (dist_bound_knob("speed_lo", "tts.speed", 0, 0.5, 2.0), 1.0),
        (dist_bound_knob("speed_hi", "tts.speed", 1, 0.5, 2.0), 1.0),
    ],
    "voiceaug_off": [
        (dist_prob_knob("pitch", "voice_augment.pitch_semitones"), 0.0),
        (dist_prob_knob("tempo", "voice_augment.tempo"), 0.0),
        (dist_prob_knob("eq_tilt", "voice_augment.eq_tilt_db"), 0.0),
    ],
    "pitch_off": [
        (dist_prob_knob("pitch", "voice_augment.pitch_semitones"), 0.0),
    ],
    "degraded": [
        (chain_param_knob("snr_lo", "additive_noise", "snr_db", 2, 0.0, 40.0), 0.0),
        (chain_param_knob("snr_hi", "additive_noise", "snr_db", 3, 0.0, 40.0), 6.0),
        (chain_prob_knob("dropouts", "dropouts"), 0.4),
        (chain_param_knob("am_depth_hi", "am_distortion", "depth", 1, 0.0, 1.0), 0.35),
    ],
}


def _indices(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition(":")
    if not hi:
        raise argparse.ArgumentTypeError("dev indices must look like 0:400")
    return int(lo), int(hi)


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_arm(base_config: dict, arm: str) -> dict:
    """The base profile with `arm`'s mutations applied. `base_config` is not touched."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    config = copy.deepcopy(base_config)
    for knob, value in ARMS[arm]:
        knob.apply(config, value)
    return config


def _done(path: Path) -> dict[tuple[str, int], dict]:
    """Cells already recorded in `results.jsonl`, keyed by (arm, seed)."""
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(row["arm"], row["seed"]): row for row in rows}


def summarize(rows: list[dict]) -> dict:
    """Per-arm mean/spread, and each arm's gap from `base` in spread units.

    `separation` is the gap divided by the pooled seed-to-seed standard
    deviation: how many units of the reward's own noise separate this arm from
    the reference. It is a crude effect size on a handful of seeds, not a test,
    and it is reported as such -- with two seeds the standard deviation itself
    has one degree of freedom.
    """
    by_arm: dict[str, list[float]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row["reward"])

    stats = {
        arm: {
            "seeds": len(rewards),
            "mean_reward": statistics.fmean(rewards),
            "stdev_reward": statistics.stdev(rewards) if len(rewards) > 1 else None,
            "rewards": rewards,
        }
        for arm, rewards in sorted(by_arm.items())
    }

    spreads = [part["stdev_reward"] for part in stats.values()
               if part["stdev_reward"] is not None]
    pooled = statistics.fmean(spreads) if spreads else None
    reference = stats.get("base", {}).get("mean_reward")
    for part in stats.values():
        gap = None if reference is None else part["mean_reward"] - reference
        part["gap_vs_base"] = gap
        part["separation"] = (abs(gap) / pooled
                              if gap is not None and pooled else None)
    return {"arms": stats, "pooled_stdev": pooled}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", default=DEFAULT_CONFIG,
                    help=f"procedural profile the arms mutate (default: {DEFAULT_CONFIG})")
    ap.add_argument("--out", required=True, help="run directory (resumed if it exists)")
    ap.add_argument("--arms", type=_csv, default=list(ARMS),
                    help=f"comma-separated arm names (default: {','.join(ARMS)})")
    ap.add_argument("--seeds", type=lambda text: [int(v) for v in _csv(text)],
                    default=[0, 1], help="comma-separated seeds (default: 0,1)")
    ap.add_argument("--n-synth", type=int, default=200,
                    help="synthetic clips generated per cell")
    ap.add_argument("--ft-steps", type=int, default=300, help="fine-tune steps per cell")
    ap.add_argument("--dev-indices", type=_indices, default=(0, 200),
                    help="real dev-set slice as lo:hi (default: 0:200)")
    ap.add_argument("--dev-corpus", default=DEFAULT_DEV_CORPUS,
                    help="HF dataset id, or a local audio,text CSV/JSONL manifest")
    ap.add_argument("--dev-split", default=DEFAULT_DEV_SPLIT,
                    help="HF split, ignored for a local manifest")
    ap.add_argument("--text-pool", type=int, default=400,
                    help="utterances in the shared text pool, fixed across all cells")
    ap.add_argument("--device", default=None, help="torch device (default: autodetect)")
    args = ap.parse_args(argv)

    unknown = [arm for arm in args.arms if arm not in ARMS]
    if unknown:
        ap.error(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")

    with open(args.base_config, encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    mode = base_config.get("mode", "procedural")
    if mode != "procedural" and "degraded" in args.arms:
        ap.error(f"the degraded arm mutates channel.chain steps, but "
                 f"{args.base_config} is mode {mode!r}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = _done(results_path)

    # one harness for every cell: the dev slice, its zero-shot baseline and the
    # text pool are the controls, and recomputing them per cell would both cost
    # time and let the arms differ by something other than the config
    harness = TrueRewardHarness(
        out_dir / "harness",
        dev_corpus=args.dev_corpus,
        dev_split=args.dev_split,
        dev_indices=args.dev_indices,
        text_pool_size=args.text_pool,
        n_synth=args.n_synth,
        ft_steps=args.ft_steps,
        device=args.device,
    )
    # the bounded aggregate, because that is what the reward subtracts; the
    # unbounded number is in the cached baseline report next to it
    baseline = harness.baseline_report["wer_bounded"]["atc_normalized"]
    cells = [(arm, seed) for arm in args.arms for seed in args.seeds]
    print(f"{len(cells)} cells ({len(args.arms)} arms x {len(args.seeds)} seeds) "
          f"-> {out_dir}")
    print(f"reward: {args.dev_corpus} [{args.dev_indices[0]}:{args.dev_indices[1]}], "
          f"zero-shot bounded WER {baseline:.4f}")

    for arm, seed in cells:
        if (arm, seed) in done:
            print(f"[{arm} seed {seed}] cached, reward "
                  f"{done[(arm, seed)]['reward']:+.4f}")
            continue

        # both the generator draw and the fine-tune batch order move with the
        # seed, so the spread across seeds covers the whole per-trial pipeline
        harness.gen_seed = GEN_SEED + seed
        harness.ft_seed = seed
        config = build_arm(base_config, arm)
        trial_dir = out_dir / "trials" / f"{arm}_s{seed}"

        start = time.monotonic()
        result = harness(config, str(trial_dir))
        row = {
            "arm": arm,
            "seed": seed,
            "reward": result.reward,
            "wer_after": result.wer_after,
            "wer_baseline": result.wer_baseline,
            "hallucination_rate": result.hallucination_rate,
            "by_source": result.metrics.get("by_source"),
            "wall_time_sec": round(time.monotonic() - start, 1),
            "trial_dir": str(trial_dir),
        }
        with open(results_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        done[(arm, seed)] = row
        print(f"[{arm} seed {seed}] reward {result.reward:+.4f} "
              f"(WER {result.wer_after:.4f} vs {result.wer_baseline:.4f}) "
              f"{row['wall_time_sec']:.0f}s")

    summary = summarize([done[cell] for cell in cells if cell in done])
    summary["dev"] = {"corpus": args.dev_corpus, "split": args.dev_split,
                      "indices": list(args.dev_indices),
                      "baseline_wer": baseline}
    summary["settings"] = {"base_config": args.base_config, "n_synth": args.n_synth,
                           "ft_steps": args.ft_steps, "text_pool": args.text_pool}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'arm':<12} {'seeds':>5} {'mean':>9} {'stdev':>9} "
          f"{'gap':>9} {'separation':>11}")
    for arm, part in summary["arms"].items():
        stdev = "-" if part["stdev_reward"] is None else f"{part['stdev_reward']:.4f}"
        gap = "-" if part["gap_vs_base"] is None else f"{part['gap_vs_base']:+.4f}"
        sep = ("-" if part["separation"] is None
               else f"{part['separation']:.1f}x")
        print(f"{arm:<12} {part['seeds']:>5} {part['mean_reward']:>+9.4f} "
              f"{stdev:>9} {gap:>9} {sep:>11}")
    print(f"\n{out_dir / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
