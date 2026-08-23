"""Harvest non-speech receiver noise with an energy VAD."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .local_corpus import HOP_MS, TARGET_SR, _mono_16k, _speech_frames, parse_station

SPEECH_PAD_MS = 20
SQUELCH_GATE_DB = -60.0


def _manifest_sources(manifest: Path):
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            path = Path(record["path"])
            if not path.is_absolute():
                path = manifest.parent / path
            yield record.get("clip_id", path.stem), record.get("station", "unknown"), path


def _sources(source: Path):
    if source.is_file():
        yield from _manifest_sources(source)
        return
    manifest = source / "corpus.jsonl"
    if manifest.exists():
        yield from _manifest_sources(manifest)
        return
    for path in sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"):
        yield path.stem, parse_station(path), path


def _non_speech_segments(wav: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    active = _speech_frames(wav)
    hop = int(TARGET_SR * HOP_MS / 1000)
    pad = int(TARGET_SR * SPEECH_PAD_MS / 1000)
    speech_spans: list[tuple[int, int]] = []
    start = 0
    while start < len(active):
        if not active[start]:
            start += 1
            continue
        end = start + 1
        while end < len(active) and active[end]:
            end += 1
        lo = max(0, start * hop - pad)
        hi = min(len(wav), end * hop + int(TARGET_SR * 25 / 1000) + pad)
        if speech_spans and lo <= speech_spans[-1][1]:
            speech_spans[-1] = (speech_spans[-1][0], max(hi, speech_spans[-1][1]))
        else:
            speech_spans.append((lo, hi))
        start = end

    noise_spans: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in speech_spans:
        if lo - cursor >= min_samples:
            noise_spans.append((cursor, lo))
        cursor = max(cursor, hi)
    if len(wav) - cursor >= min_samples:
        noise_spans.append((cursor, len(wav)))
    return noise_spans


def _rms_db(wav: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    return float(20.0 * np.log10(rms + 1e-10))


def _ltas_centroid(wav: np.ndarray) -> float:
    nperseg = min(512, len(wav))
    if nperseg < 2:
        return 0.0
    freqs, power = signal.welch(
        wav.astype(np.float64), TARGET_SR, nperseg=nperseg, noverlap=nperseg // 2
    )
    total = float(power.sum())
    return float(np.dot(freqs, power) / total) if total > 0.0 else 0.0


def harvest(
    corpus_manifest_or_dir: str | Path,
    out_dir: str | Path,
    min_ms: int = 200,
) -> Path:
    """Write VAD-selected noise WAVs and return ``noise_stats.jsonl``."""
    if min_ms <= 0:
        raise ValueError("min_ms must be positive")
    source = Path(corpus_manifest_or_dir)
    if not source.exists():
        raise ValueError(f"source does not exist: {source}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old_segment in out.glob("[0-9][0-9][0-9][0-9].wav"):
        old_segment.unlink()
    min_samples = int(TARGET_SR * min_ms / 1000)

    records: list[dict] = []
    for clip_id, station, path in _sources(source):
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError, ValueError, sf.LibsndfileError):
            continue
        wav = _mono_16k(wav, sr)
        for lo, hi in _non_speech_segments(wav, min_samples):
            segment = wav[lo:hi]
            rms_db = _rms_db(segment)
            wav_path = out / f"{len(records):04d}.wav"
            sf.write(wav_path, segment, TARGET_SR, subtype="PCM_16")
            records.append({
                "source_clip": clip_id,
                "station": station,
                "duration": round(len(segment) / TARGET_SR, 6),
                "rms_db": round(rms_db, 3),
                "ltas_centroid_hz": round(_ltas_centroid(segment), 3),
                "squelch_gated": rms_db < SQUELCH_GATE_DB,
            })

    stats_path = out / "noise_stats.jsonl"
    with stats_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_manifest_or_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min-ms", type=int, default=200)
    args = parser.parse_args()
    stats_path = harvest(args.corpus_manifest_or_dir, args.out_dir, args.min_ms)
    records = [json.loads(line) for line in stats_path.read_text().splitlines() if line]
    gated = sum(record["squelch_gated"] for record in records)
    print(json.dumps({
        "noise_stats": str(stats_path),
        "segments": len(records),
        "gated_fraction": gated / len(records) if records else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
