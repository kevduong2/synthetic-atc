#!/usr/bin/env python
"""Evaluate a Whisper model with the fixed Tier 3 ATC metrics.

Examples:
  # baseline on the public real ATC test set
  uv run python training/evaluate.py --model openai/whisper-small.en --dataset real \
      --out reports/zero_shot.json

  # fine-tuned checkpoint on a manifest-backed evaluation set
  uv run python training/evaluate.py --model runs/whisper_atc \
      --dataset data/holdout/manifest.jsonl --out reports/whisper_atc.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import jiwer
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atcgen.dataset.build import load_manifest          # noqa: E402
from atcgen.dataset.real_atc import load_real_atc        # noqa: E402
from atcgen.text.lexicon import AIRLINE_TELEPHONY         # noqa: E402
from training.normalize import normalize_atc             # noqa: E402


_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
})
_PHONETIC_WORDS = frozenset({
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "x", "ray", "yankee", "zulu",
})
_AIRLINES = tuple(
    sorted(
        (tuple(normalize_atc(name).split()) for name in AIRLINE_TELEPHONY),
        key=len,
        reverse=True,
    )
)
_COMPACT_N_NUMBER = re.compile(r"\bN[0-9]{1,5}[A-Z]{0,2}\b", re.IGNORECASE)


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _number_end(tokens: Sequence[str], start: int) -> int:
    end = start
    while end < len(tokens) and tokens[end] in _NUMBER_WORDS:
        end += 1
    return end


def extract_callsigns(text: str) -> list[tuple[str, ...]]:
    """Return callsign token sequences in normalized ``text``.

    The deliberately small heuristic recognizes an exported airline telephony
    name followed by number words/digits, plus compact or spoken N-numbers.
    Returned sequences use the existing ATC normalizer unchanged.
    """
    normalized = normalize_atc(text)
    tokens = normalized.split()
    found: list[tuple[str, ...]] = []
    occupied: set[int] = set()

    for start in range(len(tokens)):
        for airline in _AIRLINES:
            number_start = start + len(airline)
            if tuple(tokens[start:number_start]) != airline:
                continue
            end = _number_end(tokens, number_start)
            if end > number_start:
                found.append(tuple(tokens[start:end]))
                occupied.update(range(start, end))
            break

    # A literal N123AB normalizes to "n one two three a b"; a spoken N-number
    # starts with "november" and normally spells suffix letters phonetically.
    compact_starts = {
        len(normalize_atc(text[:match.start()]).split())
        for match in _COMPACT_N_NUMBER.finditer(text)
    }
    for start, token in enumerate(tokens):
        if start in occupied or (token != "november" and start not in compact_starts):
            continue
        end = _number_end(tokens, start + 1)
        if end == start + 1:
            continue
        suffix_end = end
        while suffix_end < len(tokens) and suffix_end - end < 2:
            suffix = tokens[suffix_end]
            if suffix not in _PHONETIC_WORDS and not (len(suffix) == 1 and suffix.isalpha()):
                break
            suffix_end += 1
        found.append(tuple(tokens[start:suffix_end]))

    return found


def has_callsign(text: str) -> bool:
    """Whether ``text`` contains an airline-number or N-number callsign."""
    return bool(extract_callsigns(text))


def _wer(references: Sequence[str], hypotheses: Sequence[str]) -> float | None:
    return float(jiwer.wer(list(references), list(hypotheses))) if references else None


def _wer_pair(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    normalized_refs = [normalize_atc(text) for text in references]
    normalized_hyps = [normalize_atc(text) for text in hypotheses]
    return {
        "raw": _wer(references, hypotheses),
        "atc_normalized": _wer(normalized_refs, normalized_hyps),
    }


def _contains_sequence(tokens: Sequence[str], sequence: Sequence[str]) -> bool:
    width = len(sequence)
    return any(tuple(tokens[start:start + width]) == tuple(sequence)
               for start in range(len(tokens) - width + 1))


def build_report(references: Sequence[str], hypotheses: Sequence[str],
                 categories: Sequence[str | None] | None = None,
                 *, model: str | None = None,
                 dataset: str | None = None) -> dict:
    """Build the JSON-serializable Tier 3 report from aligned transcripts."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if categories is None:
        categories = [None] * len(references)
    if len(categories) != len(references):
        raise ValueError("categories must have the same length as references")

    rows = [
        {
            "reference": reference,
            "hypothesis": hypothesis,
            "category": category or "uncategorized",
        }
        for reference, hypothesis, category in zip(references, hypotheses, categories)
    ]
    speech = [row for row in rows if row["reference"].strip()]
    noise = [row for row in rows if not row["reference"].strip()]

    per_category_rows: dict[str, list[dict]] = defaultdict(list)
    for row in speech:
        per_category_rows[row["category"]].append(row)

    callsign_rows = [row for row in speech if has_callsign(row["reference"])]
    reference_sequences = [
        sequence
        for row in callsign_rows
        for sequence in extract_callsigns(row["reference"])
    ]
    reproduced = 0
    for row in callsign_rows:
        hyp_tokens = normalize_atc(row["hypothesis"]).split()
        reproduced += sum(
            _contains_sequence(hyp_tokens, sequence)
            for sequence in extract_callsigns(row["reference"])
        )

    hallucinations = sum(
        bool(normalize_atc(row["hypothesis"]).strip()) for row in noise
    )
    report = {
        "schema_version": 1,
        "model": model,
        "dataset": dataset,
        "samples": {
            "total": len(rows),
            "speech": len(speech),
            "noise_only": len(noise),
        },
        "wer": _wer_pair(
            [row["reference"] for row in speech],
            [row["hypothesis"] for row in speech],
        ),
        "per_category": {
            category: {
                "samples": len(category_rows),
                "wer": _wer_pair(
                    [row["reference"] for row in category_rows],
                    [row["hypothesis"] for row in category_rows],
                ),
            }
            for category, category_rows in sorted(per_category_rows.items())
        },
        "callsign": {
            "samples": len(callsign_rows),
            "wer": _wer_pair(
                [row["reference"] for row in callsign_rows],
                [row["hypothesis"] for row in callsign_rows],
            ),
            "reference_sequences": len(reference_sequences),
            "exact_sequences": reproduced,
            "token_accuracy": (
                reproduced / len(reference_sequences) if reference_sequences else None
            ),
        },
        "hallucination": {
            "samples": len(noise),
            "non_empty_hypotheses": hallucinations,
            "rate": hallucinations / len(noise) if noise else None,
        },
    }
    return report


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/whisper-small.en")
    ap.add_argument("--dataset", default="real",
                    help="'real' (jacktol/atc-dataset test split) or path to a manifest.jsonl")
    ap.add_argument("--out", default=None,
                    help="write the complete Tier 3 JSON report to this path")
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

    refs, hyps, categories = [], [], []
    for ex in tqdm(ds, desc="transcribing"):
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inputs, max_new_tokens=128)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        refs.append(ex["text"])
        hyps.append(hyp)
        categories.append(ex.get("category", "uncategorized"))

    report = build_report(refs, hyps, categories, model=args.model, dataset=args.dataset)
    if args.out:
        write_report(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
