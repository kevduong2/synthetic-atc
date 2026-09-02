"""scripts/lab/resample_probes.py and shard_text.py; channel_fit.select_rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
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


def test_filter_variants_build_b3_cohorts_from_manifest(tmp_path: Path):
    from scripts.analysis import filter_variants

    src = tmp_path / "render" / "wavs"
    src.mkdir(parents=True)
    manifest = src.parent / "manifest.jsonl"
    rows = []
    for index, residual_on in enumerate((False, True, True)):
        name = f"{index:06d}.wav"
        sf.write(src / name, np.zeros(16000, dtype=np.float32), 16000)
        steps = [{"primitive": "residual_translate"}] if residual_on else []
        rows.append({"audio": f"wavs/{name}", "gen": {"channel": {"steps": steps}}})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    out = tmp_path / "variants"
    assert filter_variants.main([
        str(src), "--out", str(out), "--manifest", str(manifest),
    ]) == 0
    assert sorted(path.name for path in (out / "off").glob("*.wav")) == ["000000.wav"]
    expected_on = ["000001.wav", "000002.wav"]
    for name in ("on", "on_lp", "on_lp_hp"):
        assert sorted(path.name for path in (out / name).glob("*.wav")) == expected_on


@pytest.mark.parametrize(("reference", "compared", "sign"), [
    ("real", "v1", 1),
    ("v1", "real", -1),
])
def test_ltas_check_writes_direct_cohort_gaps(
        tmp_path: Path, monkeypatch, reference: str, compared: str, sign: int):
    from scripts.analysis import ltas_check

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    for label, high_band_scale in (("real", 0.1), ("v1", 0.4)):
        directory = tmp_path / label
        directory.mkdir()
        audio = np.sin(2 * np.pi * 400 * t) + high_band_scale * np.sin(2 * np.pi * 2000 * t)
        sf.write(directory / "000000.wav", audio.astype(np.float32), sr)
    output = tmp_path / f"{reference}.json"
    monkeypatch.setattr(sys, "argv", [
        "ltas_check.py", str(tmp_path / "real"), str(tmp_path / "v1"),
        "--label", "real", "--label", "v1", "--cohort-reference", reference,
        "--json", str(output),
    ])

    ltas_check.main()

    result = json.loads(output.read_text())
    direct = result["direct_gaps"][compared]
    assert direct["definition"] == f"{compared} - {reference}"
    assert np.sign(direct["gap_db"][5]) == sign
    assert direct["max_abs_gap_1k_3k"] > 0


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


def test_local_corpus_stations_filter_skips_other_and_unknown(tmp_path: Path):
    import pytest
    from atcgen.dataset.local_corpus import build_corpus

    src = tmp_path / "clips"
    src.mkdir()
    rng = np.random.default_rng(1)
    for name in ["KIXD_TOWER_20250801_000001.wav", "KIXD_TOWER_20250801_000002.wav",
                 "KC_CENTER_20250801_000001.wav", "KIXD_TOWER_8-1-2025_clip7.wav"]:
        sf.write(src / name, (0.3 * rng.standard_normal(16000)).astype(np.float32), 16000)
    manifest = build_corpus(src, tmp_path / "out", stations=["KIXD_TOWER"])
    stats = json.loads((manifest.parent / "corpus_stats.json").read_text())
    assert stats["stations"] == {"KIXD_TOWER": 2}
    assert stats["filtered_out"] == 2 and stats["total_files"] == 4
    with pytest.raises(ValueError, match="KEUG_TOWER"):
        build_corpus(src, tmp_path / "out2", stations=["KIXD_TOWER", "KEUG_TOWER"])
