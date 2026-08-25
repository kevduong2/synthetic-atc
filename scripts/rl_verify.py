#!/usr/bin/env python
"""Blind A/B(/C) verification: does the RL-searched config beat the hand-tuned one?

`scripts/rl_loop.py` picks `best_config.yaml` by reward measured on the same
dev slice every trial trains against -- exactly the setup that would let a
search overfit to that slice's quirks. This script is the held-out check: a
test slice neither the search nor its harness ever touched, three arms
(zero-shot, hand-tuned `base`, searched `best`) trained/evaluated identically
against it, and a paired bootstrap over the identical utterances so "better"
comes with a confidence interval and a p-value instead of one lucky number.

`base` and `best` render their synthetic batch from the *same* fresh text
pool and the *same* forced generator seed (the common-random-numbers pattern
`atcgen.rl.reward.TrueRewardHarness` uses for its own trials) -- the pool
seed here is deliberately different from the search harness's (`--text-seed`
default 4321 vs. the harness's 1234), so a config that only worked because it
happened to fit the search's text pool will not get a free pass here.

Examples:
  uv run python scripts/rl_verify.py --run runs/rl_v1
  uv run python scripts/rl_verify.py --run runs/rl_v1 --test-indices 0:500 \\
      --n-synth 600 --ft-steps 600 --save-models
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from atcgen.config import config_hash, load_config
from atcgen.dataset.real_atc import load_real_atc
from atcgen.rl.finetune_lite import prepare_features, transcribe
from atcgen.rl.reward import GEN_SEED, render_and_finetune, write_text_pool
from atcgen.rl.stats import paired_bootstrap
from training.evaluate import build_report, pick_device

DEFAULT_BASE_CONFIG = "configs/mode1_matched.yaml"
BASE_MODEL = "openai/whisper-tiny.en"
FT_SEED = 0             # shared fine-tune seed: base and best differ only by config
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
ARM_ORDER = ["zero_shot", "base", "best"]
PAIR_ORDER = ["zero_shot_vs_base", "zero_shot_vs_best", "base_vs_best"]


def _indices(text: str) -> tuple[int, int]:
    """Parse a `lo:hi` test-set slice."""
    lo, _, hi = text.partition(":")
    if not hi:
        raise argparse.ArgumentTypeError("test indices must look like 0:500")
    return int(lo), int(hi)


def _load_yaml_mapping(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _release_device_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_test_slice(corpus: str, split: str, indices: tuple[int, int], processor):
    """The blind test slice: references, categories, and pre-extracted features."""
    dataset = load_real_atc(split, corpus)
    lo, hi = indices
    dataset = dataset.select(range(lo, min(hi, len(dataset))))
    refs = list(dataset["text"])
    categories = (list(dataset["category"]) if "category" in dataset.column_names
                 else [None] * len(dataset))
    features = prepare_features(dataset, processor)
    return refs, categories, features


def run_zero_shot_arm(refs, categories, features, *, device, processor, dataset_name) -> tuple:
    """Zero-shot `BASE_MODEL`: no fine-tuning, the floor every arm should beat."""
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to(device).eval()
    hypotheses = transcribe(model, processor, features, device)
    report = build_report(refs, hypotheses, categories, model="zero_shot", dataset=dataset_name)
    del model
    _release_device_memory(device)
    return report, hypotheses


def run_config_arm(name: str, config: dict, arm_dir: Path, *, pool_path, n_synth: int,
                   ft_steps: int, ft_batch: int, ft_lr: float, device, processor,
                   features: list, refs: list, categories: list, dataset_name: str,
                   save_dir: Path | None) -> tuple:
    """Render `n_synth` clips from `config`, fine-tune, score the blind test slice.

    Returns `(report, hypotheses, config_hash, ft_seconds)`.
    """
    start = time.monotonic()
    model = render_and_finetune(
        config, arm_dir, base_model=BASE_MODEL, pool_path=pool_path, n_synth=n_synth,
        ft_steps=ft_steps, ft_batch=ft_batch, ft_lr=ft_lr, ft_seed=FT_SEED,
        gen_seed=GEN_SEED, device=device, processor=processor)
    ft_seconds = time.monotonic() - start

    hashed = config_hash(load_config(arm_dir / "config.yaml"))
    hypotheses = transcribe(model, processor, features, device)
    report = build_report(refs, hypotheses, categories, model=name, dataset=dataset_name)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

    del model
    _release_device_memory(device)
    return report, hypotheses, hashed, ft_seconds


def _speech_subset(refs: list[str], hyps_by_arm: dict[str, list[str]]) -> tuple:
    """Non-empty-reference rows only -- matches `build_report`'s WER filter,
    so bootstrap deltas are comparable to the reported `wer.atc_normalized`."""
    mask = [bool(ref.strip()) for ref in refs]
    speech_refs = [ref for ref, keep in zip(refs, mask) if keep]
    speech_hyps = {
        arm: [hyp for hyp, keep in zip(hyps, mask) if keep]
        for arm, hyps in hyps_by_arm.items()
    }
    return speech_refs, speech_hyps


def pairwise_deltas(refs: list[str], hyps_by_arm: dict[str, list[str]], *,
                    n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    speech_refs, speech_hyps = _speech_subset(refs, hyps_by_arm)
    pairs = [tuple(name.split("_vs_")) for name in PAIR_ORDER]
    return {
        f"{a}_vs_{b}": paired_bootstrap(speech_refs, speech_hyps[a], speech_hyps[b],
                                        n_boot=n_boot, seed=seed)
        for a, b in pairs
    }


def _fmt(value) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def format_table(verify_report: dict) -> str:
    """Compact human-readable summary: per-arm metrics, then pairwise deltas."""
    lines = [f"{'arm':<10} {'atc_wer':>8} {'raw_wer':>8} {'callsign_acc':>13} {'halluc_rate':>12}"]
    for name in ARM_ORDER:
        report = verify_report["arms"][name]["report"]
        lines.append(
            f"{name:<10} {_fmt(report['wer']['atc_normalized']):>8} "
            f"{_fmt(report['wer']['raw']):>8} "
            f"{_fmt(report['callsign']['token_accuracy']):>13} "
            f"{_fmt(report['hallucination']['rate']):>12}"
        )
    lines.append("")
    lines.append(f"{'pair':<20} {'delta':>9} {'ci_low':>9} {'ci_high':>9} {'p_value':>9}")
    for pair_name in PAIR_ORDER:
        boot = verify_report["pairwise_bootstrap"][pair_name]
        lines.append(
            f"{pair_name:<20} {boot['delta']:>+9.4f} {boot['ci_low']:>+9.4f} "
            f"{boot['ci_high']:>+9.4f} {boot['p_value']:>9.4f}"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="search run directory (reads best_config.yaml)")
    ap.add_argument("--base-config", default=DEFAULT_BASE_CONFIG,
                    help=f"hand-tuned profile the search mutated (default: {DEFAULT_BASE_CONFIG})")
    ap.add_argument("--out", default=None, help="output dir (default: <run>/verify)")
    ap.add_argument("--test-corpus", default="jacktol/atc-dataset")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--test-indices", type=_indices, default=(0, 500),
                    help="blind test slice as lo:hi, never used during search (default: 0:500)")
    ap.add_argument("--n-synth", type=int, default=600)
    ap.add_argument("--ft-steps", type=int, default=600)
    ap.add_argument("--ft-batch", type=int, default=8)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--text-pool", type=int, default=1200)
    ap.add_argument("--text-seed", type=int, default=4321,
                    help="fresh pool seed, distinct from the search harness's 1234 "
                         "(default: 4321)")
    ap.add_argument("--device", default=None, help="torch device (default: autodetect)")
    ap.add_argument("--save-models", action="store_true",
                    help="save each arm's fine-tuned model under <out>/models/<arm>")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)

    base_config = _load_yaml_mapping(args.base_config)
    best_config_path = run_dir / "best_config.yaml"
    best_config = _load_yaml_mapping(best_config_path)

    pool_path = write_text_pool(out_dir / "text_pool.jsonl", args.text_pool, args.text_seed)

    lo, hi = args.test_indices
    dataset_name = f"{args.test_corpus}:{args.test_split}[{lo}:{hi}]"
    print(f"loading blind test slice {dataset_name} ...")
    refs, categories, features = load_test_slice(
        args.test_corpus, args.test_split, args.test_indices, processor)
    print(f"{len(refs)} utterances")

    print("zero-shot arm ...")
    zs_report, zs_hyps = run_zero_shot_arm(
        refs, categories, features, device=device, processor=processor, dataset_name=dataset_name)
    arms = {"zero_shot": {"report": zs_report, "hypotheses": zs_hyps}}
    ft_seconds = {}

    for name, config in (("base", base_config), ("best", best_config)):
        print(f"{name} arm: rendering {args.n_synth} clips + {args.ft_steps} fine-tune steps ...")
        arm_dir = out_dir / "arms" / name
        save_dir = (out_dir / "models" / name) if args.save_models else None
        report, hyps, hashed, seconds = run_config_arm(
            name, config, arm_dir, pool_path=pool_path, n_synth=args.n_synth,
            ft_steps=args.ft_steps, ft_batch=args.ft_batch, ft_lr=args.ft_lr,
            device=device, processor=processor, features=features, refs=refs,
            categories=categories, dataset_name=dataset_name, save_dir=save_dir)
        arms[name] = {
            "report": report, "hypotheses": hyps,
            "config_path": str((arm_dir / "config.yaml").resolve()),
            "config_hash": hashed,
        }
        ft_seconds[name] = round(seconds, 3)

    print("pairwise bootstrap ...")
    hyps_by_arm = {name: arm["hypotheses"] for name, arm in arms.items()}
    bootstrap = pairwise_deltas(refs, hyps_by_arm)

    verify_report = {
        "schema_version": 1,
        "run": str(run_dir.resolve()),
        "base_config_path": str(Path(args.base_config).resolve()),
        "best_config_path": str(best_config_path.resolve()),
        "params": {
            "base_model": BASE_MODEL,
            "n_synth": args.n_synth, "ft_steps": args.ft_steps, "ft_batch": args.ft_batch,
            "ft_lr": args.ft_lr, "ft_seed": FT_SEED, "gen_seed": GEN_SEED,
            "text_pool": args.text_pool, "text_seed": args.text_seed,
            "test_corpus": args.test_corpus, "test_split": args.test_split,
            "test_indices": list(args.test_indices), "device": str(device),
            "bootstrap_n": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "test_set": {"corpus": args.test_corpus, "split": args.test_split,
                     "indices": list(args.test_indices), "n": len(refs)},
        "arms": arms,
        "pairwise_bootstrap": bootstrap,
        "ft_seconds": ft_seconds,
    }

    report_path = out_dir / "verify_report.json"
    report_path.write_text(json.dumps(verify_report, indent=2, sort_keys=True))
    print(f"\nwrote {report_path}\n")
    table = format_table(verify_report)
    print(table)
    return verify_report


if __name__ == "__main__":
    main()
