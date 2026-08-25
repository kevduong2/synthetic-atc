#!/usr/bin/env python
"""Build immutable clean ATC content and paired channel views.

The ``base`` stage renders TTS exactly once.  The ``views`` stage then reads
those WAV files, so comparisons between channel pipelines cannot be confounded
by a second (and potentially nondeterministic) TTS render.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from atcgen.channel.chain import ChannelRecord, UtteranceMeta
from atcgen.channel.learned.residual import load_translator
from atcgen.channel.primitives import TARGET_SR, resample
from atcgen.config import GeneratorConfig, dump_resolved, load_config
from atcgen.dataset import build as dataset_build
from atcgen.entities import entities_to_dicts
from atcgen.eval.qc import QCTally, qc_sample
from atcgen.text.sources import WeightedSampler, make_text_source
from atcgen.tts.augment import VoiceAugment

PIPELINES = (
    "clean",
    "procedural_matched",
    "procedural_wide",
    "calibrated_dsp",
    "calibrated_fastcut",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _base_stats(rows: list[dict[str, Any]], seed: int, config_hash: str,
                noise_only_frac: float) -> dict[str, Any]:
    categories = Counter(row["category"] for row in rows)
    return {
        "stage": "base",
        "n_samples": len(rows),
        "seed": seed,
        "config_hash": config_hash,
        "noise_only": {
            "target": noise_only_frac,
            "count": categories.get("noise", 0),
            "achieved": round(categories.get("noise", 0) / (len(rows) or 1), 4),
        },
        "categories": dict(categories),
        "duration_seconds": round(sum(row["duration"] for row in rows), 3),
    }


def build_base(config: GeneratorConfig, out_dir: str | Path, n_samples: int,
               *, seed: int, text_source: str | Any = "grammar:region=eu",
               noise_only_frac: float = 0.03, tts=None) -> Path:
    """Render the immutable clean content pool and return its manifest path."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if not 0.0 <= noise_only_frac <= 1.0:
        raise ValueError("noise_only_frac must be between 0 and 1")
    if config.output.format != "wav":
        raise ValueError(f"unsupported output format: {config.output.format}")

    out = Path(out_dir)
    clean_dir = out / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    _, resolved_hash = dump_resolved(config, out)

    source_spec = (text_source if isinstance(text_source, str)
                   else type(text_source).__name__)
    source = make_text_source(text_source) if isinstance(text_source, str) else text_source
    sampler = WeightedSampler.for_source(source, config.dataset.category_quotas)
    if tts is None:
        from atcgen.tts import KokoroTTS

        tts = KokoroTTS(voices=config.tts.voices)
    augment = VoiceAugment.from_config(config.voice_augment)

    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    manifest = out / "manifest.jsonl"
    with manifest.open("w") as handle:
        for index in range(n_samples):
            rng = random.Random(f"{seed}:base:{index}")
            aug_rng = random.Random(f"{seed}:aug:{index}")
            noise_only = rng.random() < noise_only_frac

            if noise_only:
                wav = dataset_build._noise_bed(TARGET_SR, rng)
                sr = TARGET_SR
                utterance = None
                voice = None
                speed = None
                frontend = {"pitch": None, "tempo": None, "eq_tilt_db": None}
            else:
                utterance = dataset_build._next_utterance(sampler, source, rng)
                voice = rng.choice(config.tts.voices)
                speed = float(config.tts.speed.sample(rng))
                wav = dataset_build._synthesize(
                    tts, utterance.spoken, rng, voice, speed)
                sr = int(tts.sample_rate)
                wav, frontend = augment(wav, sr, aug_rng)

            # This single draw travels with the content into every later view.
            loudness_db = config.output.loudness_db.sample(rng)
            relative = f"clean/b{index:06d}.wav"
            path = out / relative
            sf.write(path, np.asarray(wav, dtype=np.float32), sr)
            digest = _sha256(path)
            hashes[relative] = digest
            text = utterance.transcript if utterance else ""
            row = {
                "base_id": f"b{index:06d}",
                "audio_clean": relative,
                "sr": sr,
                "text": text,
                "text_display": ((utterance.display or text) if utterance else ""),
                "role": utterance.role if utterance else "none",
                "kind": utterance.kind if utterance else "noise",
                "category": utterance.category if utterance else "noise",
                "entities": entities_to_dicts(utterance.entities) if utterance else [],
                "voice": voice,
                "speed": round(speed, 3) if speed is not None else None,
                "frontend": {
                    "pitch": frontend.get("pitch"),
                    "tempo": frontend.get("tempo"),
                    "eq_tilt_db": frontend.get("eq_tilt_db"),
                },
                "loudness_db": loudness_db,
                "duration": round(len(wav) / sr, 3),
                "clean_sha256": digest,
            }
            handle.write(json.dumps(row) + "\n")
            rows.append(row)

    stats = _base_stats(rows, seed, resolved_hash, noise_only_frac)
    _write_json(out / "stats.json", stats)
    _write_json(out / "hashes.json", {
        "config_hash": resolved_hash,
        "clean_sha256": hashes,
        "text_source": source_spec,
    })
    return manifest


def _alpha_spec(value: str | float) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        amount = float(value)
        if amount < 0.0:
            raise ValueError("alpha must be non-negative")
        return amount, amount
    pieces = str(value).split(":")
    if len(pieces) == 1:
        amount = float(pieces[0])
        if amount < 0.0:
            raise ValueError("alpha must be non-negative")
        return amount, amount
    if len(pieces) != 2:
        raise ValueError("alpha must be X or LO:HI")
    lo, hi = map(float, pieces)
    if lo > hi:
        raise ValueError("alpha range must satisfy LO <= HI")
    if lo < 0.0:
        raise ValueError("alpha must be non-negative")
    return lo, hi


def _rms_post(wav: np.ndarray, loudness_db: float | None,
              config: GeneratorConfig) -> np.ndarray:
    output = replace(config.output, sample_rate=TARGET_SR, loudness_mode="rms")
    return dataset_build._post(wav, loudness_db, TARGET_SR, output)


def _view_stats(rows: list[dict[str, Any]], pipeline: str, seed: int,
                config_hash: str, tally: QCTally) -> dict[str, Any]:
    return {
        "stage": "views",
        "pipeline": pipeline,
        "n_samples": len(rows),
        "seed": seed,
        "config_hash": config_hash,
        "duration_seconds": round(sum(row["duration"] for row in rows), 3),
        "categories": dict(Counter(row["category"] for row in rows)),
        "qc": tally.summary(),
    }


def build_view(base_dir: str | Path, pipeline: str, config: GeneratorConfig,
               out_dir: str | Path, *, seed: int, keep_preloudness: bool = False,
               derive_from: str | Path | None = None,
               checkpoint: str | Path | None = None, alpha: str | float = "1.0",
               apply_prob: float = 1.0, transcriber=None) -> Path:
    """Build one paired channel view and return its manifest path."""
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}")
    if config.output.sample_rate != TARGET_SR:
        raise ValueError(f"paired views require output.sample_rate={TARGET_SR}")
    if not 0.0 <= apply_prob <= 1.0:
        raise ValueError("apply_prob must be between 0 and 1")
    derived = pipeline == "calibrated_fastcut"
    if derived and (derive_from is None or checkpoint is None):
        raise ValueError("calibrated_fastcut requires derive_from and checkpoint")
    if not derived and derive_from is not None:
        raise ValueError("derive_from is only valid for calibrated_fastcut")

    base = Path(base_dir)
    base_rows = _read_rows(base / "manifest.jsonl")
    out = Path(out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    pre_dir = out / "pre_loudness"
    if keep_preloudness:
        pre_dir.mkdir(parents=True, exist_ok=True)
    _, resolved_hash = dump_resolved(config, out)
    lineage = dataset_build._lineage(config, resolved_hash, f"paired:{base}")
    gates = dataset_build._gates(config.qc)
    tally = QCTally()

    source_by_id: dict[str, dict[str, Any]] = {}
    translator = None
    alpha_lo, alpha_hi = _alpha_spec(alpha)
    if derived:
        source_root = Path(derive_from)
        source_rows = _read_rows(source_root / "manifest.jsonl")
        source_by_id = {row["base_id"]: row for row in source_rows}
        if len(source_by_id) != len(source_rows):
            raise ValueError("derive-from manifest has duplicate base_id values")
        expected = [row["base_id"] for row in base_rows]
        if [row["base_id"] for row in source_rows] != expected:
            raise ValueError("derive-from base_ids/order do not match the base pool")
        missing = [source_root / "pre_loudness" / Path(row["audio"]).name
                   for row in source_rows
                   if not (source_root / "pre_loudness" / Path(row["audio"]).name).exists()]
        if missing:
            raise FileNotFoundError(
                f"derive-from view is missing pre-loudness audio: {missing[0]}")
        translator = load_translator(checkpoint, strict=True)
    else:
        source_root = None
        backend_name = "calibrated" if pipeline == "calibrated_dsp" else "procedural"
        backend = None if pipeline == "clean" else dataset_build.make_backend(
            config, backend_name)

    rows: list[dict[str, Any]] = []
    previous = None
    manifest = out / "manifest.jsonl"
    with manifest.open("w") as handle:
        for index, base_row in enumerate(base_rows):
            base_id = base_row["base_id"]
            if derived:
                source_row = source_by_id[base_id]
                source_path = source_root / "pre_loudness" / Path(source_row["audio"]).name
                pre_source, source_sr = _read_wav(source_path)
                if source_sr != TARGET_SR:
                    raise ValueError(f"pre-loudness source is not {TARGET_SR} Hz: {source_path}")
                rng = random.Random(f"{seed}:fastcut:{base_id}")
                apply_draw = rng.random() < apply_prob
                amount = alpha_lo if alpha_lo == alpha_hi else rng.uniform(alpha_lo, alpha_hi)
                selected = base_row["kind"] != "noise" and apply_draw
                translated = selected and amount > 0.0
                pre = (translator(pre_source, TARGET_SR, alpha=amount)
                       if translated else pre_source.copy())
                channel = copy.deepcopy(source_row["gen"]["channel"])
                channel.setdefault("steps", []).append({
                    "primitive": "residual_translate",
                    "applied": selected,
                    "alpha": amount,
                    "checkpoint_sha256": translator.checkpoint_sha256,
                    "checkpoint_step": translator.training_step,
                })
                channel_draw_id = source_row["channel_draw_id"]
                voice = source_row["gen"].get("voice", base_row.get("voice"))
                speed = source_row["gen"].get("speed", base_row.get("speed"))
            else:
                clean, clean_sr = _read_wav(base / base_row["audio_clean"])
                if _sha256(base / base_row["audio_clean"]) != base_row["clean_sha256"]:
                    raise ValueError(f"clean_sha256 mismatch for {base_id}")
                rng = random.Random(f"{seed}:{pipeline}:{base_id}")
                hops = (2 if base_row["role"] == "pilot"
                        and rng.random() < config.dataset.pilot_double_hop_prob else 1)
                if backend is None:
                    pre = resample(clean, clean_sr, TARGET_SR)
                    record = ChannelRecord(hops=1, clean_arm=True)
                else:
                    meta = UtteranceMeta(role=base_row["role"], kind=base_row["kind"],
                                         category=base_row["category"])
                    pre, record = backend(clean, clean_sr, rng, meta,
                                          interference=previous, hops=hops)
                channel = record.as_dict()
                channel_draw_id = hashlib.sha256(json.dumps(
                    channel, sort_keys=True).encode()).hexdigest()[:16]
                voice, speed = base_row.get("voice"), base_row.get("speed")

            if derived and not translated:
                # Preserve the exact DSP endpoint (including its PCM
                # quantization), rather than normalizing a decoded copy again.
                final, final_sr = _read_wav(source_root / source_row["audio"])
                if final_sr != TARGET_SR:
                    raise ValueError("derive-from rendered audio is not 16 kHz")
            else:
                final = _rms_post(pre, base_row.get("loudness_db"), config)
            result = qc_sample(final, TARGET_SR, base_row.get("text"), gates, transcriber)
            tally.add(result)
            relative = f"wavs/{index:06d}.wav"
            sf.write(out / relative, final, TARGET_SR)
            if keep_preloudness:
                sf.write(pre_dir / Path(relative).name, pre, TARGET_SR)
            if base_row["kind"] != "noise":
                previous = final

            gen = {
                "mode": pipeline,
                "voice": voice,
                "speed": speed,
                "channel": channel,
                "qc": {"ok": result.ok, "reason": result.reason, "attempts": 1},
            }
            row = {
                "audio": relative,
                "text": base_row["text"],
                "text_display": base_row["text_display"],
                "role": base_row["role"],
                "kind": base_row["kind"],
                "category": base_row["category"],
                "duration": round(len(final) / TARGET_SR, 3),
                "entities": base_row["entities"],
                "base_id": base_id,
                "pipeline": pipeline,
                "channel_draw_id": channel_draw_id,
                "gen": gen,
                "lineage": lineage,
            }
            handle.write(json.dumps(row) + "\n")
            rows.append(row)

    stats = _view_stats(rows, pipeline, seed, resolved_hash, tally)
    _write_json(out / "stats.json", stats)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser("base", help="render immutable clean content")
    base.add_argument("--out", required=True)
    base.add_argument("--n", type=int, required=True)
    base.add_argument("--seed", type=int, required=True)
    base.add_argument("--config", required=True)
    base.add_argument("--text", default="grammar:region=eu")
    base.add_argument("--noise-only-frac", type=float, default=0.03)

    views = sub.add_parser("views", help="render one channel view")
    views.add_argument("--base", required=True)
    views.add_argument("--pipeline", choices=PIPELINES, required=True)
    views.add_argument("--config", required=True)
    views.add_argument("--out", required=True)
    views.add_argument("--seed", type=int, required=True)
    views.add_argument("--keep-preloudness", action="store_true")
    views.add_argument("--derive-from")
    views.add_argument("--checkpoint")
    views.add_argument("--alpha", default="1.0")
    views.add_argument("--apply-prob", type=float, default=1.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "base":
        manifest = build_base(config, args.out, args.n, seed=args.seed,
                              text_source=args.text,
                              noise_only_frac=args.noise_only_frac)
    else:
        manifest = build_view(
            args.base, args.pipeline, config, args.out, seed=args.seed,
            keep_preloudness=args.keep_preloudness, derive_from=args.derive_from,
            checkpoint=args.checkpoint, alpha=args.alpha, apply_prob=args.apply_prob)
    stats = json.loads((manifest.parent / "stats.json").read_text())
    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


if __name__ == "__main__":
    main()
