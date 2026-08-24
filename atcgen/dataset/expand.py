"""Expand a labeled real set with channel-matched synthetic speech (04 §2.5)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from ..config import GeneratorConfig, load_config
from ..text.grammar import Utterance
from ..text.sources import DEFAULT_CATEGORY, JsonlTextSource
from .build import build_dataset
from .local_corpus import _assign_splits, parse_station


class _ExpansionTextSource:
    """A finite pool exposed for the builder's category-aware sampler."""

    def __init__(self, records: list[Utterance]):
        self.records = records

    def sample(self, rng):
        return rng.choice(self.records)


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            if not isinstance(value.get("audio"), str) or not value["audio"]:
                raise ValueError(f"{path}:{line_number} is missing a non-empty 'audio'")
            records.append(value)
    if not records:
        raise ValueError(f"real manifest is empty: {path}")
    return records


def _source_audio(record: dict, manifest: Path) -> str:
    path = Path(record["audio"])
    if not path.is_absolute():
        path = manifest.parent / path
    return str(path.resolve())


def _split_real(records: list[dict], manifest: Path, holdout_frac: float,
                seed: int) -> tuple[list[dict], list[dict]]:
    """Apply the station-stratified convention used by ``local_corpus``."""
    prepared = []
    for record in records:
        item = dict(record)
        item["audio"] = _source_audio(item, manifest)
        station = item.get("station") or parse_station(item["audio"])
        if station == "unknown" and item.get("speaker"):
            station = f"speaker:{item['speaker']}"
        item["station"] = station
        prepared.append(item)

    _assign_splits(prepared, holdout_frac, seed)
    train = [record for record in prepared if record["split"] == "train"]
    holdout = [record for record in prepared if record["split"] == "holdout"]
    return train, holdout


def _utterance(record: dict) -> Utterance | None:
    text = record.get("text") or record.get("transcript")
    spoken = record.get("spoken") or text
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(spoken, str) or not spoken.strip():
        spoken = text
    weight = record.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
        raise ValueError("real transcript weight must be a positive number")
    return Utterance(
        spoken=spoken,
        transcript=text,
        role=record.get("role", "unknown"),
        kind=record.get("kind", "real_transcript"),
        weight=float(weight),
        category=record.get("category") or DEFAULT_CATEGORY,
    )


def _synthetic_quotas(quotas: dict[str, float], real_train: list[dict],
                      target_total: int, synthetic_count: int,
                      noise_only_frac: float) -> dict[str, float]:
    """Convert combined-set quota deficits into builder speech-pool quotas."""
    if not synthetic_count:
        return {}
    speech_draws = synthetic_count * (1.0 - noise_only_frac)
    if speech_draws <= 0 and quotas:
        raise ValueError("category quotas cannot be met when all synthetic samples are noise")
    real_categories = Counter(record.get("category") or DEFAULT_CATEGORY
                              for record in real_train)
    return {
        name: min(1.0, max(0.0, target_total * fraction
                          - real_categories.get(name, 0)) / speech_draws)
        for name, fraction in quotas.items()
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def expand(config: GeneratorConfig, out_dir: str | Path,
           target_total: int | None = None) -> Path:
    """Combine real-train clips with enough synthetic clips to reach the target."""
    if config.calibrated is None:
        raise ValueError("expansion requires a calibrated.expansion config section")
    expansion = config.calibrated.expansion
    total = expansion.target_total if target_total is None else target_total
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("target_total must be a positive integer")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_manifest = Path(expansion.real_manifest)
    real_records = _read_jsonl(real_manifest)
    real_train, holdout = _split_real(
        real_records, real_manifest, expansion.holdout_frac, config.seed)
    if total < len(real_train):
        raise ValueError(
            f"target_total ({total}) is smaller than real-train count ({len(real_train)})")

    holdout_rows = [{**record, "origin": "real"} for record in holdout]
    _write_jsonl(out / "holdout_manifest.jsonl", holdout_rows)

    synthetic_count = total - len(real_train)
    synthetic_rows: list[dict] = []
    builder_stats = {"qc": {"total": 0, "kept": 0, "discarded": 0,
                            "discard_rate": 0.0, "reasons": {},
                            "reason_rates": {}, "enabled": config.qc.enabled,
                            "asr_roundtrip": config.qc.asr_roundtrip,
                            "max_retries": config.qc.max_retries,
                            "kept_with_flag": 0}}
    adjusted_quotas = _synthetic_quotas(
        expansion.category_quotas, real_train, total, synthetic_count,
        config.dataset.noise_only_frac)

    if synthetic_count:
        text_records = [item for record in real_train if (item := _utterance(record))]
        if expansion.external_texts is not None:
            text_records.extend(JsonlTextSource(expansion.external_texts).records)
        if not text_records:
            raise ValueError("no usable real-train or external transcripts for synthesis")

        build_config = replace(
            config,
            dataset=replace(config.dataset, category_quotas=adjusted_quotas),
        )
        synthetic_manifest = build_dataset(
            build_config, out / "synthetic", synthetic_count,
            _ExpansionTextSource(text_records))
        synthetic_rows = [json.loads(line) for line in synthetic_manifest.read_text().splitlines()
                          if line.strip()]
        for record in synthetic_rows:
            record["audio"] = (Path("synthetic") / record["audio"]).as_posix()
            record["origin"] = "synthetic"
        builder_stats = json.loads((synthetic_manifest.parent / "stats.json").read_text())

    real_rows = [{**record, "origin": "real"} for record in real_train]
    combined = real_rows + synthetic_rows
    manifest = out / "manifest.jsonl"
    _write_jsonl(manifest, combined)

    categories = Counter(record.get("category") or DEFAULT_CATEGORY for record in combined)
    category_fractions = {
        name: round(count / (len(combined) or 1), 4)
        for name, count in sorted(categories.items())
    }
    stats = {
        "target_total": total,
        "real_input_count": len(real_records),
        "real_train_count": len(real_rows),
        "synthetic_count": len(synthetic_rows),
        "combined_count": len(combined),
        "holdout_size": len(holdout_rows),
        "category_counts": dict(sorted(categories.items())),
        "category_fractions": category_fractions,
        "category_quotas": {
            "targets": dict(expansion.category_quotas),
            "synthetic_targets": {name: round(value, 6)
                                  for name, value in adjusted_quotas.items()},
            "achieved": {name: category_fractions.get(name, 0.0)
                         for name in sorted(expansion.category_quotas)},
        },
        "tier0": builder_stats["qc"],
    }
    (out / "expand_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-total", type=int)
    args = parser.parse_args()
    print(expand(load_config(args.config), args.out, args.target_total))


if __name__ == "__main__":
    main()
