"""Ingest local receiver recordings into a normalized ATC corpus."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 16000
FRAME_MS = 25
HOP_MS = 10
SILENCE_PEAK = 0.015
MIN_ACTIVE_FRAC = 0.03

_STATION_RE = re.compile(r"^(.+)_\d{8}_\d{6}$")


def parse_station(path: str | Path) -> str:
    """Return the station prefix encoded in a receiver filename."""
    match = _STATION_RE.match(Path(path).stem)
    return match.group(1) if match else "unknown"


def _mono_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.reshape(-1)
    if sr != TARGET_SR and len(wav):
        common = np.gcd(sr, TARGET_SR)
        wav = signal.resample_poly(
            wav, TARGET_SR // common, sr // common
        ).astype(np.float32)
    return wav.astype(np.float32, copy=False)


def _frame_db(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    if len(wav) < frame:
        power = np.mean(wav.astype(np.float64) ** 2) if len(wav) else 0.0
        return np.array([10.0 * np.log10(power + 1e-20)])
    squared = wav.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, len(wav) - frame + 1, hop)
    power = (cumulative[starts + frame] - cumulative[starts]) / frame
    return 10.0 * np.log10(power + 1e-20)


def _speech_frames(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Energy VAD with adaptive high/low hysteresis thresholds."""
    db = _frame_db(wav, sr)
    floor = float(np.percentile(db, 20))
    speech_level = float(np.percentile(db, 90))
    if speech_level - floor < 6.0:
        return np.ones(len(db), dtype=bool)

    high = max(floor + 6.0, speech_level - 15.0)
    low = max(floor + 3.0, speech_level - 20.0)
    strong = db >= high
    weak = db >= low

    active = np.zeros(len(db), dtype=bool)
    start = 0
    while start < len(db):
        if not weak[start]:
            start += 1
            continue
        end = start + 1
        while end < len(db) and weak[end]:
            end += 1
        if strong[start:end].any():
            active[start:end] = True
        start = end
    return active


def _active_fraction(wav: np.ndarray) -> float:
    return float(np.mean(_speech_frames(wav))) if len(wav) else 0.0


def _audio_hash(wav: np.ndarray) -> str:
    canonical = np.asarray(wav, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _clip_id(stem: str, used: set[str]) -> str:
    candidate = stem
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _assign_splits(records: list[dict], holdout_frac: float, seed: int) -> None:
    by_station: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_station[record["station"]].append(i)

    rng = random.Random(seed)
    for station in sorted(by_station):
        indices = by_station[station]
        rng.shuffle(indices)
        n_holdout = round(len(indices) * holdout_frac)
        if len(indices) >= 2:
            n_holdout = min(len(indices) - 1, max(1, n_holdout))
        else:
            n_holdout = int(holdout_frac >= 0.5)
        holdout = set(indices[:n_holdout])
        for i in indices:
            records[i]["split"] = "holdout" if i in holdout else "train"


def build_corpus(
    src_dir: str | Path,
    out_dir: str | Path,
    holdout_frac: float = 0.15,
    seed: int = 0,
) -> Path:
    """Normalize local WAVs, apply QC, and write ``corpus.jsonl``."""
    if not 0.0 <= holdout_frac <= 1.0:
        raise ValueError("holdout_frac must be between 0 and 1")

    src = Path(src_dir)
    if not src.is_dir():
        raise ValueError(f"not a directory: {src}")
    out = Path(out_dir)
    clip_dir = out / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for old_clip in clip_dir.glob("*.wav"):
        old_clip.unlink()

    paths = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    drops = Counter({
        "duplicate": 0,
        "silence_only": 0,
        "unreadable": 0,
        "zero_length": 0,
    })
    seen_hashes: set[str] = set()
    used_ids: set[str] = set()
    records: list[dict] = []
    resampled = 0
    converted_to_mono = 0

    for path in paths:
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError, ValueError, sf.LibsndfileError):
            drops["unreadable"] += 1
            continue
        if wav.shape[0] == 0 or sr <= 0:
            drops["zero_length"] += 1
            continue
        if not np.isfinite(wav).all():
            drops["unreadable"] += 1
            continue

        if wav.shape[1] > 1:
            converted_to_mono += 1
        if sr != TARGET_SR:
            resampled += 1
        normalized = _mono_16k(wav, sr)
        if len(normalized) == 0:
            drops["zero_length"] += 1
            continue

        digest = _audio_hash(normalized)
        if digest in seen_hashes:
            drops["duplicate"] += 1
            continue
        seen_hashes.add(digest)

        peak = float(np.max(np.abs(normalized)))
        if peak < SILENCE_PEAK or _active_fraction(normalized) < MIN_ACTIVE_FRAC:
            drops["silence_only"] += 1
            continue

        clip_id = _clip_id(path.stem, used_ids)
        rel_path = Path("clips") / f"{clip_id}.wav"
        sf.write(out / rel_path, normalized, TARGET_SR, subtype="PCM_16")
        rms = float(np.sqrt(np.mean(normalized.astype(np.float64) ** 2)))
        records.append({
            "clip_id": clip_id,
            "path": rel_path.as_posix(),
            "station": parse_station(path),
            "duration": round(len(normalized) / TARGET_SR, 6),
            "rms_db": round(20.0 * np.log10(rms + 1e-10), 3),
            "peak": round(peak, 6),
            "split": "train",
        })

    _assign_splits(records, holdout_frac, seed)
    manifest = out / "corpus.jsonl"
    with manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    stats = {
        "source_dir": str(src),
        "total_files": len(paths),
        "kept": len(records),
        "dropped": dict(drops),
        "resampled": resampled,
        "converted_to_mono": converted_to_mono,
        "stations": dict(sorted(Counter(r["station"] for r in records).items())),
        "splits": dict(sorted(Counter(r["split"] for r in records).items())),
    }
    with (out / "corpus_stats.json").open("w") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = build_corpus(args.src_dir, args.out_dir, args.holdout_frac, args.seed)
    stats = json.loads((manifest.parent / "corpus_stats.json").read_text())
    print(json.dumps({"manifest": str(manifest), **stats}, indent=2))


if __name__ == "__main__":
    main()
