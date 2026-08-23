import json
import random
from collections import Counter, defaultdict

import numpy as np
import pytest
import soundfile as sf

from atcgen.channel.primitives import NoiseBank
from atcgen.dataset.local_corpus import build_corpus, parse_station
from atcgen.dataset.noise_harvest import harvest

SR = 16000


def _burst_clip(freq: float, sr: int = SR) -> np.ndarray:
    wav = np.zeros(sr, dtype=np.float32)
    start, end = int(0.2 * sr), int(0.8 * sr)
    t = np.arange(end - start) / sr
    wav[start:end] = 0.2 * np.sin(2 * np.pi * freq * t)
    return wav


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_build_corpus_qc_station_parsing_and_splits(tmp_path):
    src = tmp_path / "receiver"
    src.mkdir()
    specs = {"ALPHA_TOWER": (10, 500), "BRAVO_CENTER": (4, 680)}
    for station, (count, base_freq) in specs.items():
        for i in range(count):
            name = f"{station}_20260101_{120000 + i:06d}.wav"
            sf.write(src / name, _burst_clip(base_freq + i * 17), SR)

    sf.write(src / "misc-a.wav", _burst_clip(810), SR)
    stereo_8k = np.column_stack([_burst_clip(830, 8000)] * 2)
    sf.write(src / "misc-b.wav", stereo_8k, 8000)
    sf.write(src / "duplicate_name_20260101_130000.wav", _burst_clip(500), SR)
    sf.write(src / "silent_20260101_130001.wav", np.zeros(SR), SR)

    manifest = build_corpus(src, tmp_path / "corpus", holdout_frac=0.25, seed=7)
    rows = _records(manifest)
    stats = json.loads((manifest.parent / "corpus_stats.json").read_text())

    assert len(rows) == 16
    assert stats["dropped"]["duplicate"] == 1
    assert stats["dropped"]["silence_only"] == 1
    assert stats["resampled"] == 1
    assert stats["converted_to_mono"] == 1
    assert parse_station("KSDL_TOWER_20260819_125740.wav") == "KSDL_TOWER"
    assert parse_station("receiver.wav") == "unknown"
    assert Counter(row["station"] for row in rows) == {
        "ALPHA_TOWER": 10,
        "BRAVO_CENTER": 4,
        "unknown": 2,
    }

    by_station = defaultdict(list)
    for row in rows:
        by_station[row["station"]].append(row["split"])
        wav, sr = sf.read(manifest.parent / row["path"])
        assert sr == SR and wav.ndim == 1
    for splits in by_station.values():
        assert set(splits) == {"train", "holdout"}
        assert splits.count("holdout") / len(splits) == pytest.approx(0.25, abs=0.25)


def test_harvest_vad_gated_stats_and_noise_bank(tmp_path):
    src = tmp_path / "receiver"
    src.mkdir()
    rng = np.random.default_rng(3)
    for station, floor in (("GATED_TOWER", None), ("OPEN_CENTER", 0.003)):
        wav = np.empty(int(1.4 * SR), dtype=np.float32)
        t = np.arange(int(0.5 * SR)) / SR
        speech = 0.2 * np.sin(2 * np.pi * 700 * t)
        wav[: len(speech)] = speech
        gap = slice(len(speech), len(speech) + int(0.4 * SR))
        wav[gap] = 0.0 if floor is None else rng.normal(0.0, floor, gap.stop - gap.start)
        wav[gap.stop:] = speech
        sf.write(src / f"{station}_20260101_120000.wav", wav, SR)

    manifest = build_corpus(src, tmp_path / "corpus", holdout_frac=0.5)
    noise_dir = tmp_path / "noise"
    stats_path = harvest(manifest, noise_dir, min_ms=200)
    rows = _records(stats_path)

    assert len(rows) == 2
    assert all(row["duration"] >= 0.2 for row in rows)
    by_station = {row["station"]: row for row in rows}
    assert by_station["GATED_TOWER"]["squelch_gated"] is True
    assert by_station["OPEN_CENTER"]["squelch_gated"] is False
    assert by_station["OPEN_CENTER"]["ltas_centroid_hz"] > 0
    for path in noise_dir.glob("[0-9][0-9][0-9][0-9].wav"):
        wav, sr = sf.read(path)
        assert sr == SR
        assert np.max(np.abs(wav)) < 0.03

    bank = NoiseBank(noise_dir)
    sample = bank.sample(2 * SR, random.Random(0))
    assert sample.shape == (2 * SR,)
    assert sample.dtype == np.float32
