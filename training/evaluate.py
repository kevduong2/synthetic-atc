#!/usr/bin/env python
"""Evaluate a Whisper model's WER on synthetic and/or real ATC test sets.

Examples:
  # baseline on real ATC test set
  uv run python training/evaluate.py --model openai/whisper-small.en --dataset real

  # fine-tuned checkpoint on a synthetic holdout
  uv run python training/evaluate.py --model runs/whisper_atc --dataset data/holdout/manifest.jsonl
"""

import argparse
import sys
from pathlib import Path

import jiwer
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atcgen.dataset.build import load_manifest          # noqa: E402
from atcgen.dataset.real_atc import load_real_atc        # noqa: E402
from training.normalize import normalize_atc             # noqa: E402


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/whisper-small.en")
    ap.add_argument("--dataset", default="real",
                    help="'real' (jacktol/atc-dataset test split) or path to a manifest.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    if args.dataset == "real":
        ds = load_real_atc(split="test")
    else:
        ds = load_manifest(args.dataset)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    refs, hyps = [], []
    for ex in tqdm(ds, desc="transcribing"):
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inputs, max_new_tokens=128)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        refs.append(normalize_atc(ex["text"]))
        hyps.append(normalize_atc(hyp))

    wer = jiwer.wer(refs, hyps)
    print(f"\nmodel: {args.model}\ndataset: {args.dataset}\nsamples: {len(refs)}\nWER: {wer:.4f}")
    for r, h in list(zip(refs, hyps))[:5]:
        print(f"\nREF: {r}\nHYP: {h}")


if __name__ == "__main__":
    main()
