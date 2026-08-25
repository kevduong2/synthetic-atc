#!/usr/bin/env python
"""Staged L2 student recipe: SFT -> (synthetic mix) -> GRPO.

Glue, not a framework. Produces the experiment-matrix arms of
`docs/plans/research-integration.md` with budget-matched optimizer steps --
every arm runs the same `--sft-steps` fixed optimizer steps, so arms differ in
*what* the student sees, never in how much gradient it gets:

| arm         | stage 1 SFT pool                 | stage 2 |
|-------------|----------------------------------|---------|
| real_only   | real slice (A1)                  | -       |
| synth_only  | synthetic manifests (A2 / A2u)   | -       |
| mix         | --mix-ratio real / synth (A3)    | -       |
| mix_grpo    | --mix-ratio real / synth (A3)    | GRPO on the same mixture (A4) |

SFT reuses `atcgen.rl.finetune_lite.finetune` (fixed-steps, seeded, no HF
Trainer) rather than a second trainer implementation; GRPO calls
`training.grpo.run_grpo` on the stage-1 checkpoint. Per §4.6 the fine-tune is
full -- no PEFT LoRA, which is structurally incompatible with Whisper's
log-mel encoder.

Every stage writes a `save_pretrained` checkpoint directory (loadable by
`training/evaluate.py --model <dir>`) and the run writes `run.json` with the
config, seeds, pool sizes, step counts and wall times.

  uv run python training/recipe.py --arm mix_grpo \
      --real-split train --real-indices 0:8000 \
      --synth-manifest data/train_v1 --mix-ratio 0.75 \
      --sft-steps 1000 --grpo-steps 300 \
      --dev-split train --dev-indices 9000:9400 --out runs/a4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atcgen.rl.finetune_lite import finetune                       # noqa: E402
from atcgen.tracking import start_run                              # noqa: E402
from training.evaluate import pick_device                          # noqa: E402
from training.grpo import (                                        # noqa: E402
    SR, DataSpec, GRPOConfig, RewardWeights, UtterancePool, concat_pools,
    evaluate_dev, load_pool, parse_indices, run_grpo,
)

ARMS = ("real_only", "synth_only", "mix", "mix_grpo")


class LazyFeatures(Sequence):
    """Whisper features for an `UtterancePool`, extracted on access.

    `finetune` only ever calls `len()` and integer indexing, so the pool can
    stay lazy: one log-mel is ~1 MB, and materializing an 8k-utterance SFT pool
    up front would cost ~8 GB against ~10 ms of extraction per clip amortized
    inside the training step.
    """

    def __init__(self, pool: UtterancePool, processor: WhisperProcessor) -> None:
        self._pool = pool
        self._processor = processor

    def __len__(self) -> int:
        return len(self._pool)

    def __getitem__(self, index):
        utterance = self._pool[int(index)]
        return {
            "input_features": self._processor(
                utterance.audio, sampling_rate=SR).input_features[0],
            "labels": self._processor.tokenizer(utterance.text).input_ids,
        }


def mixed_pool(real: UtterancePool, synthetic: UtterancePool, ratio: float,
               seed: int) -> UtterancePool:
    """Seeded interleave of two pools at an exact real:synthetic ratio.

    Uses as much data as the ratio allows: whichever side binds fixes the
    total, and the other is subsampled from a seeded permutation. The two
    sides are then shuffled together so the mixture is interleaved rather than
    concatenated (`finetune` permutes again, but a concatenated pool would
    still be wrong for any consumer that does not).
    """
    if not len(real):
        return synthetic
    if not len(synthetic):
        return real
    ratio = min(max(ratio, 0.0), 1.0)
    if ratio <= 0.0:
        total_real, total_synth = 0, len(synthetic)
    elif ratio >= 1.0:
        total_real, total_synth = len(real), 0
    else:
        total = min(len(real) / ratio, len(synthetic) / (1.0 - ratio))
        total_real = min(len(real), int(round(total * ratio)))
        total_synth = min(len(synthetic), int(round(total * (1.0 - ratio))))

    rng = np.random.default_rng(seed)
    real_rows = rng.permutation(len(real))[:total_real]
    synth_rows = rng.permutation(len(synthetic))[:total_synth]
    combined = concat_pools([real, synthetic])
    indices = np.concatenate([real_rows, synth_rows + len(real)])
    rng.shuffle(indices)
    return combined.select(indices)


@dataclass
class RecipeConfig:
    arm: str = "mix"
    out: str = "runs/recipe"
    model: str = "openai/whisper-tiny.en"

    real_corpus: str = "jacktol/atc-dataset"
    real_split: str | None = "train"
    real_indices: tuple[int, int] | None = None
    synth_manifests: list[str] = field(default_factory=list)
    mix_ratio: float = 0.75

    dev_manifests: list[str] = field(default_factory=list)
    dev_split: str | None = None
    dev_indices: tuple[int, int] | None = None

    sft_steps: int = 500
    sft_batch: int = 8
    sft_lr: float = 1e-5

    grpo_steps: int = 300
    grpo_batch: int = 4
    grpo_group: int = 6
    grpo_lr: float = 1e-6
    grpo_beta: float = 0.04
    grpo_temperature: float = 0.9
    grpo_eval_every: int = 50
    weights: RewardWeights = field(default_factory=RewardWeights)

    seed: int = 0
    device: str | None = None
    dev_batch: int = 8

    def json(self) -> dict:
        from dataclasses import asdict

        payload = asdict(self)
        for key in ("real_indices", "dev_indices"):
            payload[key] = list(payload[key]) if payload[key] else None
        return payload


def _real_spec(cfg: RecipeConfig) -> DataSpec:
    return DataSpec(real_corpus=cfg.real_corpus, real_split=cfg.real_split,
                    real_indices=cfg.real_indices)


def _synth_spec(cfg: RecipeConfig) -> DataSpec:
    return DataSpec(manifests=list(cfg.synth_manifests))


def _dev_spec(cfg: RecipeConfig) -> DataSpec:
    return DataSpec(manifests=list(cfg.dev_manifests), real_corpus=cfg.real_corpus,
                    real_split=cfg.dev_split, real_indices=cfg.dev_indices)


def build_sft_pool(cfg: RecipeConfig) -> UtterancePool:
    """The stage-1 training pool implied by `cfg.arm`."""
    if cfg.arm == "real_only":
        return load_pool(_real_spec(cfg))
    if cfg.arm == "synth_only":
        return load_pool(_synth_spec(cfg))
    if cfg.arm in ("mix", "mix_grpo"):
        return mixed_pool(load_pool(_real_spec(cfg)), load_pool(_synth_spec(cfg)),
                          cfg.mix_ratio, cfg.seed)
    raise ValueError(f"unknown arm {cfg.arm!r}; expected one of {ARMS}")


def _tracking_log(run, values: dict, step: int | None = None) -> None:
    """Best-effort log: tracking is observability, never training control flow."""
    try:
        run.log(values, step=step)
    except Exception:
        pass


def run_recipe(cfg: RecipeConfig) -> dict:
    """Run the arm end to end. Returns the run summary (also at out/run.json)."""
    tracking_run = start_run(project="atcgan-fastcut", name=Path(cfg.out).name,
                             config=cfg.json(), tags=("asr", cfg.arm))
    try:
        return _run_recipe(cfg, tracking_run)
    finally:
        try:
            tracking_run.finish()
        except Exception:
            pass


def _run_recipe(cfg: RecipeConfig, tracking_run) -> dict:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)

    processor = WhisperProcessor.from_pretrained(cfg.model)
    pool = build_sft_pool(cfg)
    dev_spec = _dev_spec(cfg)
    dev_pool = (load_pool(dev_spec)
                if (dev_spec.manifests or dev_spec.real_split) else None)

    model = WhisperForConditionalGeneration.from_pretrained(cfg.model)
    model.config.use_cache = False
    model.to(device)

    started = time.monotonic()
    def on_sft_step(step: int, loss: float) -> None:
        if step % 10 == 0:
            _tracking_log(tracking_run, {"sft/loss": loss}, step=step)

    finetune(model, LazyFeatures(pool, processor), steps=cfg.sft_steps,
             batch_size=cfg.sft_batch, lr=cfg.sft_lr, seed=cfg.seed, device=device,
             on_step=on_sft_step)
    sft_seconds = time.monotonic() - started
    _tracking_log(tracking_run, {"sft/wall_seconds": sft_seconds,
                                 "sft/pool_size": len(pool)}, step=cfg.sft_steps)

    sft_dir = out / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(sft_dir)
    processor.save_pretrained(sft_dir)

    losses = list(getattr(model, "_ft_losses", []))
    stages = [{
        "name": "sft",
        "checkpoint": str(sft_dir),
        "pool_size": len(pool),
        "steps": cfg.sft_steps,
        "batch_size": cfg.sft_batch,
        "lr": cfg.sft_lr,
        "samples_seen": cfg.sft_steps * cfg.sft_batch,
        "loss_first": losses[0] if losses else None,
        "loss_tail": losses[-10:],
        "wall_seconds": round(sft_seconds, 2),
    }]

    if dev_pool is not None:
        dev = evaluate_dev(model, processor, dev_pool, device, cfg.dev_batch, 100)
        stages[-1]["dev"] = dev
        _tracking_log(tracking_run, {f"dev/{key}": value
                                     for key, value in dev.items()})

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    if cfg.arm == "mix_grpo":
        grpo_out = out / "grpo"
        grpo_cfg = GRPOConfig(
            init=str(sft_dir), out=str(grpo_out),
            train=DataSpec(manifests=list(cfg.synth_manifests),
                           real_corpus=cfg.real_corpus, real_split=cfg.real_split,
                           real_indices=cfg.real_indices),
            dev=dev_spec,
            steps=cfg.grpo_steps, batch=cfg.grpo_batch, group=cfg.grpo_group,
            temperature=cfg.grpo_temperature, lr=cfg.grpo_lr, beta=cfg.grpo_beta,
            weights=cfg.weights, eval_every=cfg.grpo_eval_every,
            dev_batch=cfg.dev_batch, seed=cfg.seed, device=cfg.device,
        )
        started = time.monotonic()
        summary = run_grpo(grpo_cfg, train_pool=pool, dev_pool=dev_pool)
        _tracking_log(tracking_run, {
            "grpo/best_dev_wer": summary["best"].get("dev_wer"),
            "grpo/best_step": summary["best"].get("step"),
            "grpo/wall_seconds": summary.get("wall_seconds"),
            "grpo/pool_size": len(pool),
        })
        stages.append({
            "name": "grpo",
            "checkpoint": summary["best"]["checkpoint"] or summary["last_checkpoint"],
            "last_checkpoint": summary["last_checkpoint"],
            "pool_size": len(pool),
            "steps": cfg.grpo_steps,
            "wall_seconds": round(time.monotonic() - started, 2),
            "grpo": summary,
        })

    run = {
        "arm": cfg.arm,
        "config": cfg.json(),
        "device": str(device),
        "seed": cfg.seed,
        "final_checkpoint": stages[-1]["checkpoint"],
        "stages": stages,
        "wall_seconds": round(sum(stage["wall_seconds"] for stage in stages), 2),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return run


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, default="mix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/whisper-tiny.en")

    ap.add_argument("--real-corpus", default="jacktol/atc-dataset")
    ap.add_argument("--real-split", default="train")
    ap.add_argument("--real-indices", default=None, help="LO:HI slice of --real-split")
    ap.add_argument("--synth-manifest", action="append", default=[],
                    help="synthetic dataset dir or manifest.jsonl; repeatable")
    ap.add_argument("--mix-ratio", type=float, default=0.75, help="real fraction of the mix")

    ap.add_argument("--dev-manifest", action="append", default=[])
    ap.add_argument("--dev-split", default=None)
    ap.add_argument("--dev-indices", default=None)
    ap.add_argument("--dev-batch", type=int, default=8)

    ap.add_argument("--sft-steps", type=int, default=500)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--sft-lr", type=float, default=1e-5)

    ap.add_argument("--grpo-steps", type=int, default=300)
    ap.add_argument("--grpo-batch", type=int, default=4)
    ap.add_argument("--grpo-group", type=int, default=6)
    ap.add_argument("--grpo-lr", type=float, default=1e-6)
    ap.add_argument("--grpo-beta", type=float, default=0.04)
    ap.add_argument("--grpo-temperature", type=float, default=0.9)
    ap.add_argument("--grpo-eval-every", type=int, default=50)
    ap.add_argument("--w-rep", type=float, default=0.5)
    ap.add_argument("--w-len", type=float, default=0.3)
    ap.add_argument("--w-hal", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    return ap


def config_from_args(args) -> RecipeConfig:
    return RecipeConfig(
        arm=args.arm, out=args.out, model=args.model,
        real_corpus=args.real_corpus, real_split=args.real_split,
        real_indices=parse_indices(args.real_indices),
        synth_manifests=args.synth_manifest, mix_ratio=args.mix_ratio,
        dev_manifests=args.dev_manifest, dev_split=args.dev_split,
        dev_indices=parse_indices(args.dev_indices), dev_batch=args.dev_batch,
        sft_steps=args.sft_steps, sft_batch=args.sft_batch, sft_lr=args.sft_lr,
        grpo_steps=args.grpo_steps, grpo_batch=args.grpo_batch,
        grpo_group=args.grpo_group, grpo_lr=args.grpo_lr, grpo_beta=args.grpo_beta,
        grpo_temperature=args.grpo_temperature, grpo_eval_every=args.grpo_eval_every,
        weights=RewardWeights(w_rep=args.w_rep, w_len=args.w_len, w_hal=args.w_hal),
        seed=args.seed, device=args.device,
    )


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if args.arm in ("synth_only", "mix", "mix_grpo") and not args.synth_manifest:
        ap.error(f"--arm {args.arm} requires at least one --synth-manifest")
    if args.arm in ("real_only", "mix", "mix_grpo") and not args.real_indices:
        ap.error(f"--arm {args.arm} requires --real-indices LO:HI")
    run = run_recipe(config_from_args(args))
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
