#!/usr/bin/env python
"""Fine-tune Whisper under the fixed Tier 3 ATC training regimes.

Small validation run on the Mac:
  uv run python training/finetune_whisper.py --manifest data/smoke/manifest.jsonl \
      --model openai/whisper-tiny.en --out runs/whisper_smoke --max-steps 5 --eval-set holdout

5080 protocol (run with cached model/data; never on the development machine):
  # Baseline 1: zero-shot Whisper-small.en (evaluation only)
  uv run python training/evaluate.py --model openai/whisper-small.en \
      --split-name locked_test --report-out reports/zero_shot_small_en.json

  # Baseline 2: real-only fine-tuning
  uv run python training/finetune_whisper.py --real-only \
      --model openai/whisper-small.en --out runs/whisper_real_only \
      --epochs 3 --batch-size 16 --fp16

  # Regime 1: synthetic-only fine-tuning
  uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl \
      --model openai/whisper-small.en --out runs/whisper_synthetic_only \
      --epochs 3 --batch-size 16 --fp16

  # Regime 2: synthetic first, then real last
  uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl \
      --curriculum --model openai/whisper-small.en --out runs/whisper_curriculum \
      --epochs 3 --batch-size 16 --fp16

Pass ``--real-manifest path/to/manifest.jsonl`` to use local labeled real data
instead of the public real training split. Both manifest flags may be repeated.
``--mix-real`` remains the joint shuffled ~1:1 alternative to curriculum.
"""

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration, WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atcgen.dataset.build import load_manifest           # noqa: E402
from atcgen.dataset.real_atc import load_real_atc         # noqa: E402


@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    dataset: Any


def curriculum_phases(synthetic: Any, real: Any) -> list[TrainingPhase]:
    """Return the fixed sequential order: all synthetic, then all real."""
    if synthetic is None or real is None:
        raise ValueError("curriculum requires both synthetic and real datasets")
    return [TrainingPhase("synthetic", synthetic), TrainingPhase("real", real)]


def _load_manifest_set(paths: Sequence[str]):
    from datasets import concatenate_datasets

    datasets = [load_manifest(path).select_columns(["audio", "text"]) for path in paths]
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def _load_real_train(paths: Sequence[str] | None):
    if paths:
        return _load_manifest_set(paths)
    return load_real_atc(split="train").select_columns(["audio", "text"])


def _mix_one_to_one(synthetic, real, seed: int = 0):
    from datasets import concatenate_datasets

    # Upsample real to approximately 1:1 with synthetic. The final partial
    # excess is retained, matching the original joint-mixing behavior.
    reps = max(1, round(len(synthetic) / len(real)))
    return concatenate_datasets([synthetic] + [real] * reps).shuffle(seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", default=[],
                    help="synthetic manifest.jsonl; repeat for multiple synthetic sets")
    ap.add_argument("--real-manifest", action="append", default=[],
                    help="local labeled-real manifest; repeat to combine (default: public train split)")
    regime = ap.add_mutually_exclusive_group()
    regime.add_argument("--mix-real", action="store_true",
                        help="jointly mix real ATC training data at approximately 1:1")
    regime.add_argument("--curriculum", action="store_true",
                        help="train sequentially on synthetic manifest(s), then real data")
    regime.add_argument("--real-only", action="store_true",
                        help="train the real-only Tier 3 baseline")
    ap.add_argument("--eval-set", choices=["real", "holdout"], default="real",
                    help="'real' = public real test split; 'holdout' = synthetic (or real-only) slice")
    ap.add_argument("--eval-samples", type=int, default=200,
                    help="max real eval samples when --eval-set real")
    ap.add_argument("--model", default="openai/whisper-small.en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3,
                    help="epochs per phase (applies independently to both curriculum phases)")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="maximum steps per phase; -1 uses --epochs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--eval-holdout", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42,
                    help="trainer seed (data order, dropout); 42 is the HF default")
    args = ap.parse_args()

    if args.real_only:
        if args.manifest:
            ap.error("--real-only cannot be combined with --manifest")
    elif not args.manifest:
        ap.error("--manifest is required unless --real-only is selected")
    if args.real_manifest and not (args.real_only or args.mix_real or args.curriculum):
        ap.error("--real-manifest requires --real-only, --mix-real, or --curriculum")

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    synthetic = _load_manifest_set(args.manifest) if args.manifest else None
    needs_real = args.real_only or args.mix_real or args.curriculum
    real = _load_real_train(args.real_manifest) if needs_real else None

    if args.eval_set == "real":
        eval_raw = load_real_atc(split="test")
        eval_raw = eval_raw.select(range(min(args.eval_samples, len(eval_raw))))
    else:
        holdout_source = real if args.real_only else synthetic
        split = holdout_source.train_test_split(
            test_size=max(args.eval_holdout, 1 / max(len(holdout_source), 2)),
            seed=args.seed,
        )
        if args.real_only:
            real = split["train"]
        else:
            synthetic = split["train"]
        eval_raw = split["test"]

    if args.real_only:
        phases = [TrainingPhase("real", real)]
    elif args.curriculum:
        phases = curriculum_phases(synthetic, real)
    elif args.mix_real:
        phases = [TrainingPhase("joint", _mix_one_to_one(synthetic, real, args.seed))]
    else:
        phases = [TrainingPhase("synthetic", synthetic)]

    def prepare(ex):
        audio = ex["audio"]
        ex["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        ex["labels"] = processor.tokenizer(ex["text"]).input_ids
        return ex

    eval_ds = eval_raw.map(prepare, remove_columns=eval_raw.column_names,
                           desc="extracting eval features")
    prepared_phases = [
        TrainingPhase(
            phase.name,
            phase.dataset.map(
                prepare,
                remove_columns=phase.dataset.column_names,
                desc=f"extracting {phase.name} features",
            ),
        )
        for phase in phases
    ]

    base_training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=100,
        fp16=args.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="epoch" if args.max_steps < 0 else "no",
        logging_steps=25,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = None
    for index, phase in enumerate(prepared_phases, start=1):
        phase_out = (Path(args.out) / f"phase_{index}_{phase.name}"
                     if len(prepared_phases) > 1 else Path(args.out))
        training_args = replace(base_training_args, output_dir=str(phase_out))
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=phase.dataset,
            eval_dataset=eval_ds,
            data_collator=Collator(processor),
        )
        print(f"starting {phase.name} phase ({index}/{len(prepared_phases)})")
        trainer.train()

    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
