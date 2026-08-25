#!/usr/bin/env python
"""Evaluate a Whisper model with the entity/safety panel plus WER.

Aggregate WER alone never decides anything here: ATC command accuracy and WER
decouple (Helmke et al.), so the report leads with the entity panel -- slot F1
per type, exact callsign accuracy, critical-number substitution rate -- and
carries WER with its substitution/deletion/insertion split alongside.

Evaluation sets come from `atcgen.dataset.splits` (D11: disjoint from anything
the generator, the gate or the search loop has seen). `--split-name
locked_test` is the final-report slice and should be read once per arm.

Examples:
  # A0 development baseline: zero-shot tiny.en on the model-selection split
  uv run python training/evaluate.py --model openai/whisper-tiny.en \
      --split-name model_select --report-out runs/eval/a0_model_select.json

  # a fine-tuned checkpoint on a manifest-backed synthetic set
  uv run python training/evaluate.py --model runs/whisper_atc \
      --dataset data/holdout/manifest.jsonl --report-out reports/whisper_atc.json
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import jiwer
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atcgen.dataset.build import load_manifest                # noqa: E402
from atcgen.dataset.real_atc import load_real_atc              # noqa: E402
from atcgen.dataset.splits import SPLIT_NAMES, load_split, split_spec  # noqa: E402
from atcgen.eval.entity_metrics import entity_panel, resolve_ref_entities  # noqa: E402
from training.normalize import normalize_atc                   # noqa: E402

SCHEMA_VERSION = 2

#: Short clips lose context and long ones invite Whisper's looping failure
#: mode, so WER is reported per band as well as overall.
DURATION_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<3s", 0.0, 3.0),
    ("3-6s", 3.0, 6.0),
    (">6s", 6.0, float("inf")),
)

Transcriber = Callable[[list[np.ndarray]], list[str]]


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _measures(references: Sequence[str], hypotheses: Sequence[str]):
    """jiwer word measures over the non-empty-reference pairs, or None."""
    pairs = [(reference, hypothesis)
             for reference, hypothesis in zip(references, hypotheses)
             if reference.strip()]
    if not pairs:
        return None
    return jiwer.process_words([reference for reference, _ in pairs],
                               [hypothesis for _, hypothesis in pairs])


def _wer_pair(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    """WER before and after ATC normalization, with the S/D/I split.

    The error counts come from the normalized pair: the raw number is inflated
    by spelling conventions ("35" vs "three five") that the normalizer exists
    to fold away, so counting substitutions there measures the transcript
    style rather than the model.
    """
    raw = _measures(list(references), list(hypotheses))
    normalized = _measures([normalize_atc(text) for text in references],
                           [normalize_atc(text) for text in hypotheses])
    pair = {
        "raw": float(raw.wer) if raw is not None else None,
        "atc_normalized": float(normalized.wer) if normalized is not None else None,
    }
    if normalized is None:
        pair.update(substitutions=0, deletions=0, insertions=0,
                    hits=0, reference_words=0)
    else:
        pair.update(
            substitutions=int(normalized.substitutions),
            deletions=int(normalized.deletions),
            insertions=int(normalized.insertions),
            hits=int(normalized.hits),
            reference_words=int(normalized.hits + normalized.substitutions
                                + normalized.deletions),
        )
    return pair


def _slice_wer(rows: Sequence[dict]) -> dict:
    return {
        "samples": len(rows),
        "wer": _wer_pair([row["reference"] for row in rows],
                         [row["hypothesis"] for row in rows]),
    }


def _duration_slices(rows: Sequence[dict]) -> dict:
    banded: dict[str, list[dict]] = {name: [] for name, _, _ in DURATION_BANDS}
    for row in rows:
        duration = row.get("duration")
        if duration is None:
            continue
        for name, low, high in DURATION_BANDS:
            if low <= duration < high:
                banded[name].append(row)
                break
    return {name: _slice_wer(band) for name, band in banded.items() if band}


def build_report(references: Sequence[str], hypotheses: Sequence[str],
                 categories: Sequence[str | None] | None = None,
                 *, model: str | None = None,
                 dataset: str | None = None,
                 ref_entities: Sequence | None = None,
                 durations: Sequence[float | None] | None = None,
                 split: dict | None = None,
                 airlines: dict[str, str] | None = None,
                 max_examples: int = 5) -> dict:
    """Build the JSON-serializable evaluation report from aligned transcripts.

    `ref_entities[i]` is the ground truth for utterance `i` when the row has
    one (synthetic manifests carry grammar-emitted `entities`); otherwise the
    reference is parsed. `durations[i]` in seconds enables the duration-band
    slices. Noise-only rows (empty reference) are excluded from every WER and
    from the entity panel; they are what `hallucination` scores.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if categories is None:
        categories = [None] * len(references)
    if len(categories) != len(references):
        raise ValueError("categories must have the same length as references")
    if durations is None:
        durations = [None] * len(references)
    if len(durations) != len(references):
        raise ValueError("durations must have the same length as references")

    rows = [
        {
            "index": index,
            "reference": reference,
            "hypothesis": hypothesis,
            "category": category or "uncategorized",
            "duration": duration,
        }
        for index, (reference, hypothesis, category, duration)
        in enumerate(zip(references, hypotheses, categories, durations))
    ]
    speech = [row for row in rows if row["reference"].strip()]
    noise = [row for row in rows if not row["reference"].strip()]

    per_category_rows: dict[str, list[dict]] = defaultdict(list)
    for row in speech:
        per_category_rows[row["category"]].append(row)

    speech_refs = [row["reference"] for row in speech]
    speech_hyps = [row["hypothesis"] for row in speech]
    speech_labels = resolve_ref_entities(
        speech_refs,
        [ref_entities[row["index"]] for row in speech] if ref_entities else None,
        airlines=airlines,
    )
    entities = entity_panel(speech_refs, speech_hyps, speech_labels,
                            airlines=airlines, max_examples=max_examples)
    callsign_rows = [row for row, labels in zip(speech, speech_labels)
                     if any(entity.type == "callsign" for entity in labels)]

    hallucinations = sum(
        bool(normalize_atc(row["hypothesis"]).strip()) for row in noise
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "dataset": dataset,
        "split": split,
        "samples": {
            "total": len(rows),
            "speech": len(speech),
            "noise_only": len(noise),
        },
        "wer": _wer_pair(speech_refs, speech_hyps),
        "entities": entities,
        "per_category": {
            category: _slice_wer(category_rows)
            for category, category_rows in sorted(per_category_rows.items())
        },
        "slices": {
            "duration": _duration_slices(speech),
        },
        "callsign": {
            "samples": len(callsign_rows),
            "wer": _wer_pair([row["reference"] for row in callsign_rows],
                             [row["hypothesis"] for row in callsign_rows]),
            "reference_callsigns": entities["callsign"]["total"],
            "exact_matches": entities["callsign"]["correct"],
            "accuracy": entities["callsign"]["accuracy"],
            "substitutions": entities["callsign"]["substitutions"],
        },
        "hallucination": {
            "samples": len(noise),
            "non_empty_hypotheses": hallucinations,
            "rate": hallucinations / len(noise) if noise else None,
        },
    }


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def whisper_batch_transcriber(model, processor, device, *,
                        max_new_tokens: int = 128) -> Transcriber:
    """A batched `Transcriber` over a loaded Whisper model."""

    def transcribe(batch: list[np.ndarray]) -> list[str]:
        features = processor(batch, sampling_rate=16000,
                             return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(features, max_new_tokens=max_new_tokens)
        return [text.strip()
                for text in processor.batch_decode(ids, skip_special_tokens=True)]

    return transcribe


def _rows_of(dataset) -> list[dict]:
    """Reference-side columns of a HF Dataset or a plain list of dicts."""
    collected = []
    for row in dataset:
        audio = row["audio"]
        array = np.asarray(audio["array"], dtype=np.float32)
        rate = audio.get("sampling_rate", 16000)
        collected.append({
            "audio": array,
            "reference": row.get("text", ""),
            "category": row.get("category", "uncategorized"),
            "entities": row.get("entities"),
            "duration": row.get("duration", len(array) / rate if rate else None),
        })
    return collected


def evaluate_dataset(dataset, transcriber: Transcriber, *, batch_size: int = 8,
                     model: str | None = None, dataset_name: str | None = None,
                     split: dict | None = None, max_examples: int = 5,
                     progress: bool = True, hyps_out: str | Path | None = None) -> dict:
    """Transcribe every row of `dataset` with `transcriber` and score it.

    `transcriber` maps a list of 16 kHz mono waveforms to a list of
    hypotheses, which is the whole model dependency -- tests inject a fake one
    and never load Whisper. `hyps_out` writes one {index, reference,
    hypothesis} JSON object per line for paired significance tests across
    models (atcgen.rl.stats.paired_bootstrap needs per-utterance pairs).
    """
    rows = _rows_of(dataset)
    hypotheses: list[str] = []
    batches = range(0, len(rows), batch_size)
    for start in tqdm(batches, desc="transcribing", disable=not progress):
        chunk = rows[start:start + batch_size]
        hypotheses.extend(transcriber([row["audio"] for row in chunk]))
    if len(hypotheses) != len(rows):
        raise ValueError(
            f"transcriber returned {len(hypotheses)} hypotheses for {len(rows)} rows")

    if hyps_out:
        path = Path(hyps_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for index, (row, hypothesis) in enumerate(zip(rows, hypotheses)):
                f.write(json.dumps({"index": index, "reference": row["reference"],
                                    "hypothesis": hypothesis}) + "\n")

    return build_report(
        [row["reference"] for row in rows], hypotheses,
        [row["category"] for row in rows],
        model=model, dataset=dataset_name, split=split,
        ref_entities=[row["entities"] for row in rows],
        durations=[row["duration"] for row in rows],
        max_examples=max_examples,
    )


def load_eval_dataset(*, split_name: str | None, dataset: str | None):
    """Resolve the CLI's dataset selection to (dataset, name, split spec)."""
    if split_name:
        spec = split_spec(split_name)
        if split_name == "locked_test":
            print("WARNING: locked_test is the final-report split -- one read "
                  "per arm, never for tuning (D11).", file=sys.stderr)
        return load_split(split_name), spec.dataset_name(), spec.to_dict()
    dataset = dataset or "real"
    if dataset == "real":
        return load_real_atc(split="test"), "jacktol/atc-dataset:test", None
    return load_manifest(dataset), dataset, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/whisper-small.en")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--split-name", choices=SPLIT_NAMES, default=None,
                        help="evaluation slice from atcgen.dataset.splits (D11)")
    source.add_argument("--dataset", default=None,
                        help="'real' (jacktol test split) or path to a manifest.jsonl")
    ap.add_argument("--report-out", default=None,
                    help="write the complete JSON report to this path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-examples", type=int, default=5,
                    help="worst-entity examples kept in the report")
    ap.add_argument("--hyps-out", default=None,
                    help="write per-utterance {index, reference, hypothesis} JSONL")
    args = ap.parse_args()

    device = pick_device(args.device)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    ds, dataset_name, split = load_eval_dataset(split_name=args.split_name,
                                                dataset=args.dataset)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    report = evaluate_dataset(
        ds, whisper_batch_transcriber(model, processor, device),
        batch_size=args.batch_size, model=args.model,
        dataset_name=dataset_name, split=split, max_examples=args.max_examples,
        hyps_out=args.hyps_out)
    if args.report_out:
        write_report(report, args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
