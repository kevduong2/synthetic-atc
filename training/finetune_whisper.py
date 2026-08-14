#!/usr/bin/env python
"""Fine-tune Whisper on a synthetic ATC dataset (optionally mixed with real data).

Small validation run on the Mac:
  uv run python training/finetune_whisper.py --manifest data/smoke/manifest.jsonl \\
      --model openai/whisper-tiny.en --out runs/whisper_smoke --max-steps 5

Real run on the 5080:
  uv run python training/finetune_whisper.py --manifest data/train_v1/manifest.jsonl \\
      --model openai/whisper-small.en --out runs/whisper_atc --epochs 3 --batch-size 16 --fp16
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="synthetic manifest.jsonl")
    ap.add_argument("--mix-real", action="store_true",
                    help="also mix in the real ATC train split (jacktol/atc-dataset)")
    ap.add_argument("--model", default="openai/whisper-small.en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--eval-holdout", type=float, default=0.02)
    args = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    ds = load_manifest(args.manifest)
    if args.mix_real:
        from datasets import concatenate_datasets
        real = load_real_atc(split="train")
        ds = concatenate_datasets([ds.select_columns(["audio", "text"]),
                                   real.select_columns(["audio", "text"])])

    def prepare(ex):
        audio = ex["audio"]
        ex["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        ex["labels"] = processor.tokenizer(ex["text"]).input_ids
        return ex

    ds = ds.map(prepare, remove_columns=ds.column_names, desc="extracting features")
    split = ds.train_test_split(test_size=max(args.eval_holdout, 1 / max(len(ds), 2)), seed=0)

    training_args = Seq2SeqTrainingArguments(
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
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=Collator(processor),
    )
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
