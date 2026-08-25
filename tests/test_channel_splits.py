"""Phase 0 channel folds and the guards that keep their artifacts disjoint."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from atcgen.channel.learned import channel_fit
from atcgen.channel.learned.preset import BAND_EDGES, Preset, load_presets
from atcgen.dataset.channel_splits import build_channel_splits, parse_timestamp
from atcgen.dataset.noise_harvest import harvest
from scripts.audit_channel_leakage import main as audit_main

SR = 16000


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _corpus(tmp_path: Path, clips: list[tuple[str, str]]) -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    rows = []
    for clip_id, station in clips:
        path = source / f"{clip_id}.wav"
        path.touch()
        rows.append({"clip_id": clip_id, "path": path.name,
                     "station": station, "split": "train"})
    return _write_jsonl(source / "corpus.jsonl", rows)


def _preset(clip_id: str, split: str = "channel_train") -> Preset:
    return Preset(
        clip_id=clip_id,
        station="TEST",
        band_gains_db=[0.0] * (len(BAND_EDGES) - 1),
        drive=1.0,
        poly=[0.0, 0.0],
        agc_tau_ms=50.0,
        agc_strength=0.0,
        noise_gain=0.0,
        snr_est=45.0,
        fit_loss=1.0,
        passband_hz=[100.0, 8000.0],
        split=split,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_timestamp_uses_receiver_clip_convention():
    assert parse_timestamp("KSDL_TOWER_20260825_143012") == datetime(
        2026, 8, 25, 14, 30, 12)
    assert parse_timestamp("KSDL_TOWER_20260230_143012") is None
    assert parse_timestamp("not_a_receiver_clip") is None


def test_gap_blocks_stay_whole_and_paths_become_absolute(tmp_path):
    manifest = _corpus(tmp_path, [
        ("ALPHA_20260101_120000", "ALPHA"),
        ("ALPHA_20260101_120500", "ALPHA"),
        ("ALPHA_20260101_123000", "ALPHA"),
        ("ALPHA_20260101_123500", "ALPHA"),
    ])
    out = tmp_path / "blocked"
    build_channel_splits(manifest, out, val_frac=0.5, gap_min=15.0, seed=0)
    rows = _read_jsonl(out / "corpus.jsonl")

    by_block: dict[str, set[str]] = {}
    for row in rows:
        by_block.setdefault(row["block_id"], set()).add(row["split"])
        assert Path(row["path"]).is_absolute()
        assert row["timestamp"] is not None
    assert len(by_block) == 2
    assert all(len(splits) == 1 for splits in by_block.values())
    assert rows[0]["block_id"] == rows[1]["block_id"]
    assert rows[2]["block_id"] == rows[3]["block_id"]


def test_single_capture_block_is_pseudo_split_at_largest_gap(tmp_path):
    manifest = _corpus(tmp_path, [
        ("BRAVO_20260101_120000", "BRAVO"),
        ("BRAVO_20260101_120100", "BRAVO"),
        ("BRAVO_20260101_121000", "BRAVO"),
        ("BRAVO_20260101_121100", "BRAVO"),
    ])
    out = tmp_path / "blocked"
    build_channel_splits(manifest, out, val_frac=0.15, gap_min=15.0, seed=2)
    rows = _read_jsonl(out / "split_manifest.jsonl")

    assert len({row["block_id"] for row in rows}) == 2
    assert all(row["pseudo_split"] is True for row in rows)
    assert rows[1]["block_id"] != rows[2]["block_id"]
    assert {row["split"] for row in rows} == {"channel_train", "channel_val"}


def test_assignment_is_seeded_and_approximates_target(tmp_path):
    clips = [(f"CHARLIE_20260101_{12 + i:02d}0000", "CHARLIE")
             for i in range(10)]
    manifest = _corpus(tmp_path, clips)
    first = tmp_path / "first"
    repeat = tmp_path / "repeat"
    other = tmp_path / "other"
    build_channel_splits(manifest, first, val_frac=0.3, gap_min=15.0, seed=7)
    build_channel_splits(manifest, repeat, val_frac=0.3, gap_min=15.0, seed=7)
    build_channel_splits(manifest, other, val_frac=0.3, gap_min=15.0, seed=8)

    assert (first / "split_manifest.jsonl").read_text() == (
        repeat / "split_manifest.jsonl").read_text()
    first_rows = _read_jsonl(first / "split_manifest.jsonl")
    other_rows = _read_jsonl(other / "split_manifest.jsonl")
    assert sum(row["split"] == "channel_val" for row in first_rows) == 3
    assert [row["split"] for row in first_rows] != [
        row["split"] for row in other_rows]


def test_never_assigns_all_blocks_to_val_and_warns_on_unparsed(tmp_path):
    manifest = _corpus(tmp_path, [
        ("DELTA_20260101_120000", "DELTA"),
        ("DELTA_20260101_130000", "DELTA"),
        ("DELTA_20260101_140000", "DELTA"),
        ("mystery", "SINGLE"),
    ])
    out = tmp_path / "blocked"
    stats = build_channel_splits(
        manifest, out, val_frac=1.0, gap_min=15.0, seed=0)
    rows = _read_jsonl(out / "split_manifest.jsonl")
    delta = [row for row in rows if row["station"] == "DELTA"]
    single = [row for row in rows if row["station"] == "SINGLE"]

    assert any(row["split"] == "channel_train" for row in delta)
    assert single[0]["split"] == "channel_train"
    assert stats["stations"]["DELTA"]["val_blocks"] == 2
    assert stats["warnings"]["single_clip_stations"] == ["SINGLE"]
    assert stats["warnings"]["unparsed_timestamps"] == ["mystery"]


def test_noise_harvest_filters_and_propagates_manifest_split(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    rows = []
    for index, split in enumerate(("channel_train", "channel_val")):
        wav = np.zeros(int(1.4 * SR), dtype=np.float32)
        tone = 0.2 * np.sin(2 * np.pi * 700 * np.arange(SR // 2) / SR)
        wav[:len(tone)] = tone
        wav[-len(tone):] = tone
        path = clips / f"clip_{index}.wav"
        sf.write(path, wav, SR)
        rows.append({"clip_id": f"clip_{index}", "path": path.name,
                     "station": "TEST", "split": split})
    manifest = _write_jsonl(clips / "corpus.jsonl", rows)

    stats = harvest(
        manifest, tmp_path / "noise", min_ms=200, split="channel_train")
    harvested = _read_jsonl(stats)
    assert harvested
    assert {row["source_clip"] for row in harvested} == {"clip_0"}
    assert {row["split"] for row in harvested} == {"channel_train"}

    bare = tmp_path / "bare"
    bare.mkdir()
    sf.write(bare / "TEST_20260101_120000.wav", np.zeros(SR), SR)
    with pytest.raises(ValueError, match="requires a corpus manifest"):
        harvest(bare, tmp_path / "rejected", split="channel_train")


def test_load_presets_partition_guard_names_all_offenders(tmp_path):
    path = _write_jsonl(tmp_path / "presets.jsonl", [
        _preset("good").as_dict(),
        _preset("bad_b", "channel_val").as_dict(),
        _preset("bad_a", "train").as_dict(),
    ])
    assert len(load_presets(path)) == 3
    with pytest.raises(ValueError, match=r"bad_a, bad_b"):
        load_presets(path, expect_split="channel_train")


def test_fit_corpus_filters_before_fitting_and_records_counts(tmp_path,
                                                             monkeypatch):
    wav_path = tmp_path / "clip.wav"
    wav = 0.1 * np.sin(2 * np.pi * 500 * np.arange(2 * SR) / SR)
    sf.write(wav_path, wav, SR)
    manifest = _write_jsonl(tmp_path / "corpus.jsonl", [
        {"clip_id": "train", "path": wav_path.name, "station": "TEST",
         "split": "channel_train"},
        {"clip_id": "val", "path": wav_path.name, "station": "TEST",
         "split": "channel_val"},
    ])
    calls = []

    class StubModel:
        def to_preset(self, **fields):
            base = _preset(fields["clip_id"], fields["split"]).as_dict()
            return Preset.from_dict(base | fields)

    def stub_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return StubModel(), [1.0]

    monkeypatch.setattr(channel_fit, "fit_clip", stub_fit)
    monkeypatch.setattr(
        channel_fit, "verify_ltas",
        lambda *args, **kwargs: np.ones(len(BAND_EDGES) - 1))
    out = tmp_path / "presets.jsonl"
    summary = channel_fit.fit_corpus(
        manifest, out, n_probes=1, split="channel_train")

    assert len(calls) == 1
    assert [preset.clip_id for preset in load_presets(out)] == ["train"]
    assert summary["split_filter"] == "channel_train"
    assert summary["input_counts"] == {
        "channel_train": 1, "channel_val": 1}
    with pytest.raises(ValueError, match="channel_test"):
        channel_fit.fit_corpus(manifest, out, split="channel_test")


def test_leakage_audit_cli_passes_with_matching_inputs(tmp_path, capsys):
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [
        {"clip_id": "train", "split": "channel_train"},
        {"clip_id": "val", "split": "channel_val"},
    ])
    presets = _write_jsonl(tmp_path / "presets.jsonl", [
        {"clip_id": "train", "split": "channel_train"}])
    noise = _write_jsonl(tmp_path / "noise.jsonl", [
        {"source_clip": "train", "split": "channel_train"}])
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({
        "presets": {"path": str(presets), "sha256": _sha256(presets)},
        "noise_stats": {"path": str(noise), "sha256": _sha256(noise)},
    }))
    out = tmp_path / "audit.json"

    report = audit_main([
        "--corpus", str(corpus), "--presets", str(presets),
        "--noise-stats", str(noise), "--run-inputs", str(inputs),
        "--out", str(out),
    ])
    assert report["ok"] is True
    assert json.loads(out.read_text()) == report
    capsys.readouterr()


def test_leakage_audit_cli_collects_all_failures(tmp_path, capsys):
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [
        {"clip_id": "train", "split": "channel_train"},
        {"clip_id": "val", "split": "channel_val"},
    ])
    presets = _write_jsonl(tmp_path / "presets.jsonl", [
        {"clip_id": "val", "split": "channel_val"}])
    noise = _write_jsonl(tmp_path / "noise.jsonl", [
        {"source_clip": "missing", "split": None}])
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({
        "presets": {"sha256": "wrong"},
        "noise_stats": {"sha256": "also-wrong"},
    }))
    out = tmp_path / "audit.json"

    with pytest.raises(SystemExit, match="1"):
        audit_main([
            "--corpus", str(corpus), "--presets", str(presets),
            "--noise-stats", str(noise), "--run-inputs", str(inputs),
            "--out", str(out),
        ])
    report = json.loads(out.read_text())
    assert report["ok"] is False
    assert len(report["forbidden"]) == 2
    assert len(report["mismatches"]) == 2
    capsys.readouterr()
