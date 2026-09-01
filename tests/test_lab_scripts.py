"""scripts/lab/resample_probes.py and shard_text.py; channel_fit.select_rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from atcgen.channel.learned.channel_fit import select_rows
from atcgen.channel.primitives import TARGET_SR
from scripts.lab import resample_probes, shard_text


def _rows(**counts):
    rows = []
    for station, n in counts.items():
        rows.extend({"station": station, "clip_id": f"{station}_{i}"} for i in range(n))
    return rows


def test_select_rows_without_caps_keeps_file_order():
    rows = _rows(KEUG=2, KIXD=3)
    assert select_rows(rows) == rows
    assert select_rows(rows, limit=3) == rows[:3]


def test_select_rows_per_station_balances_and_is_seeded():
    rows = _rows(KEUG_TOWER=5, KIXD_TOWER=500, S50=2)
    picked = select_rows(rows, per_station=4, seed=0)
    by = {}
    for r in picked:
        by.setdefault(r["station"], []).append(r["clip_id"])
    assert {k: len(v) for k, v in by.items()} == {"KEUG_TOWER": 4, "KIXD_TOWER": 4, "S50": 2}
    assert picked == select_rows(rows, per_station=4, seed=0)
    other = [r["clip_id"] for r in select_rows(rows, per_station=4, seed=1)
             if r["station"] == "KIXD_TOWER"]
    assert other != by["KIXD_TOWER"]
    # round-robin interleave: a total cap still spans every station
    capped = select_rows(rows, per_station=4, limit=3, seed=0)
    assert [r["station"] for r in capped] == ["KEUG_TOWER", "KIXD_TOWER", "S50"]


def test_resample_probes_rewrites_wavs_and_manifest(tmp_path: Path, capsys):
    clean = tmp_path / "probe" / "clean"
    clean.mkdir(parents=True)
    t = np.arange(24000, dtype=np.float32) / 24000
    sf.write(clean / "b000000.wav", 0.5 * np.sin(2 * np.pi * 440 * t), 24000)
    sf.write(clean / "b000001.wav", np.zeros(TARGET_SR, np.float32), TARGET_SR)
    (tmp_path / "probe" / "manifest.jsonl").write_text(
        json.dumps({"path": "clean/b000000.wav", "sr": 24000}) + "\n"
        + json.dumps({"path": "clean/b000001.wav", "sr": TARGET_SR}) + "\n")
    assert resample_probes.main([str(clean)]) == 0
    assert "1/2 wavs resampled" in capsys.readouterr().out
    for p in clean.glob("*.wav"):
        wav, sr = sf.read(p, dtype="float32")
        assert sr == TARGET_SR and len(wav) == TARGET_SR
    rows = [json.loads(l) for l in (tmp_path / "probe" / "manifest.jsonl").read_text().splitlines()]
    assert [r["sr"] for r in rows] == [TARGET_SR, TARGET_SR]
    # idempotent
    stats = resample_probes.resample_dir(clean)
    assert stats["resampled"] == 0 and stats["manifest_rows"] == 0


def test_shard_text_round_robin_covers_every_line_once(tmp_path: Path, capsys):
    src = tmp_path / "texts.jsonl"
    src.write_text("".join(json.dumps({"i": i, "airport": "A" if i < 5 else "B"}) + "\n"
                           for i in range(10)))
    assert shard_text.main([str(src), "--n", "4"]) == 0
    out = capsys.readouterr().out
    assert "      10  total" in out
    paths = shard_text.shard_paths(src, 4)
    assert [p.name for p in paths] == [f"texts.shard{i}of4.jsonl" for i in (1, 2, 3, 4)]
    seen = []
    for p in paths:
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        assert {r["airport"] for r in rows} == {"A", "B"}
        seen.extend(r["i"] for r in rows)
    assert sorted(seen) == list(range(10))
    assert [len(p.read_text().splitlines()) for p in paths] == [3, 3, 2, 2]


def test_filter_variants_attenuate_out_of_band_energy(tmp_path: Path, capsys):
    from scripts.analysis import filter_variants

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    # in-band 1 kHz tone + out-of-band 5 kHz tone + 60 Hz hum
    x = 0.3 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t) + 0.3 * np.sin(2 * np.pi * 60 * t)
    src = tmp_path / "wavs"
    src.mkdir()
    sf.write(src / "000000.wav", x.astype(np.float32), sr)
    assert filter_variants.main([str(src), "--out", str(tmp_path / "v")]) == 0
    assert "lp_hp" in capsys.readouterr().out

    def band_power(y, f, bw=50):
        spec = np.abs(np.fft.rfft(y)) ** 2
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        return spec[(freqs > f - bw) & (freqs < f + bw)].sum()

    lp, _ = sf.read(tmp_path / "v" / "lp" / "000000.wav", dtype="float32")
    lphp, _ = sf.read(tmp_path / "v" / "lp_hp" / "000000.wav", dtype="float32")
    assert len(lp) == len(x) and len(lphp) == len(x)
    assert band_power(lp, 1000) > 0.5 * band_power(x, 1000)       # in-band kept
    assert band_power(lp, 5000) < 1e-3 * band_power(x, 5000)      # LP removed 5 kHz
    assert band_power(lp, 60) > 0.5 * band_power(x, 60)           # LP alone keeps hum
    assert band_power(lphp, 60) < 1e-2 * band_power(x, 60)        # HP removed hum
    assert band_power(lphp, 1000) > 0.5 * band_power(x, 1000)


def test_local_corpus_per_station_cap_bounds_the_ingest(tmp_path: Path):
    from atcgen.dataset.local_corpus import build_corpus, cap_per_station

    src = tmp_path / "clips"
    src.mkdir()
    rng = np.random.default_rng(0)
    names = [f"KIXD_TOWER_20250801_{i:06d}.wav" for i in range(12)] + \
            [f"KEUG_TOWER_20250801_{i:06d}.wav" for i in range(3)]
    for name in names:
        sf.write(src / name, (0.3 * rng.standard_normal(16000)).astype(np.float32), 16000)
    picked = cap_per_station(sorted(src.glob("*.wav")), 5, seed=0)
    assert [p.name.startswith("KEUG") for p in picked].count(True) == 3
    assert [p.name.startswith("KIXD") for p in picked].count(True) == 5
    assert picked == cap_per_station(sorted(src.glob("*.wav")), 5, seed=0)
    manifest = build_corpus(src, tmp_path / "out", per_station=5, seed=0)
    stats = json.loads((manifest.parent / "corpus_stats.json").read_text())
    assert stats["total_files"] == 15 and stats["selected_files"] == 8
    assert stats["stations"] == {"KEUG_TOWER": 3, "KIXD_TOWER": 5}
    assert len(list((tmp_path / "out" / "clips").glob("*.wav"))) == 8
