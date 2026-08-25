"""Paired-view builder tests use deterministic local audio only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from atcgen.channel.chain import ChannelRecord
from atcgen.channel.learned.residual import ResidualGenerator, save_generator
from atcgen.config import load_config
from atcgen.dataset.build import load_manifest
from atcgen.eval.qc import QCResult
from scripts import build_paired_views as paired


class FakeTTS:
    sample_rate = 24000

    def synthesize(self, text, rng, voice=None, speed=1.0):
        digest = hashlib.sha256(text.encode()).digest()
        frequency = 180 + digest[0]
        t = np.arange(int(self.sample_rate * 0.7 / speed)) / self.sample_rate
        return (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _files(tmp_path: Path):
    source = tmp_path / "texts.jsonl"
    source.write_text(json.dumps({
        "spoken": "alpha one cleared to land",
        "transcript": "alpha one cleared to land",
        "display": "ALPHA1, cleared to land",
        "role": "pilot",
        "kind": "landing",
        "category": "landing",
        "entities": [],
    }) + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(f"""
mode: procedural
seed: 99
output:
  sample_rate: 16000
  loudness_db: {{choice: [-20]}}
tts:
  voices: [af_heart, am_adam]
  speed: {{uniform: [0.95, 1.05]}}
voice_augment:
  pitch_semitones: {{choice: [0]}}
  tempo: {{choice: [1]}}
  eq_tilt_db: {{choice: [0]}}
dataset:
  noise_only_frac: 0
  pilot_double_hop_prob: 0.5
channel:
  profile: paired-test
  clean_arm_prob: 0
  chain:
    - primitive: bandpass
      prob: 1
      low: {{choice: [300]}}
      high: {{choice: [3000]}}
qc:
  enabled: true
  asr_roundtrip: false
  min_duration: 0.1
  max_duration: 10
  min_rms_db: -45
  max_rms_db: -5
""")
    return load_config(config), source


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _build_base(tmp_path: Path, name="base", *, n=4, noise=0.0):
    config, source = _files(tmp_path)
    manifest = paired.build_base(
        config, tmp_path / name, n, seed=17, text_source=str(source),
        noise_only_frac=noise, tts=FakeTTS())
    return config, manifest


def test_base_is_byte_deterministic_for_the_same_seed(tmp_path):
    _, first = _build_base(tmp_path, "first")
    _, second = _build_base(tmp_path, "second")
    first_rows, second_rows = _rows(first), _rows(second)
    assert [row["clean_sha256"] for row in first_rows] == [
        row["clean_sha256"] for row in second_rows]
    for left, right in zip(first_rows, second_rows):
        assert (first.parent / left["audio_clean"]).read_bytes() == (
            second.parent / right["audio_clean"]).read_bytes()


def test_different_views_receive_the_same_decoded_clean_audio(tmp_path, monkeypatch):
    config, manifest = _build_base(tmp_path)
    received = []

    class CaptureBackend:
        def __call__(self, wav, sr, rng, meta, interference=None, hops=1):
            received.append((wav.copy(), sr))
            return wav.copy(), ChannelRecord(hops=hops)

    monkeypatch.setattr(paired.dataset_build, "make_backend",
                        lambda config, name: CaptureBackend())
    paired.build_view(manifest.parent, "procedural_matched", config,
                      tmp_path / "matched", seed=2)
    split = len(received)
    paired.build_view(manifest.parent, "procedural_wide", config,
                      tmp_path / "wide", seed=3)
    assert split == 4 and len(received) == 8
    for left, right in zip(received[:split], received[split:]):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1] == right[1] == FakeTTS.sample_rate


def test_view_schema_and_load_manifest_round_trip(tmp_path):
    config, base_manifest = _build_base(tmp_path)
    manifest = paired.build_view(base_manifest.parent, "clean", config,
                                 tmp_path / "clean-view", seed=4)
    row = _rows(manifest)[0]
    assert {
        "audio", "text", "text_display", "role", "kind", "category",
        "duration", "entities", "base_id", "pipeline", "channel_draw_id",
        "gen", "lineage",
    } <= row.keys()
    assert row["gen"]["qc"]["attempts"] == 1
    loaded = load_manifest(manifest)
    assert loaded.column_names.count("base_id") == 1
    assert loaded[0]["base_id"] == row["base_id"]
    assert loaded[0]["pipeline"] == "clean"


def test_fastcut_alpha_zero_is_exact_source_and_keeps_draw_ids(tmp_path):
    config, base_manifest = _build_base(tmp_path)
    source = paired.build_view(base_manifest.parent, "clean", config,
                               tmp_path / "source", seed=5,
                               keep_preloudness=True)
    checkpoint = save_generator(tmp_path / "toy.pt", ResidualGenerator(base=2, n_res=1),
                                extra={"step": 12})
    derived = paired.build_view(
        base_manifest.parent, "calibrated_fastcut", config,
        tmp_path / "derived", seed=6, derive_from=source.parent,
        checkpoint=checkpoint, alpha="0", apply_prob=1.0)
    source_rows, derived_rows = _rows(source), _rows(derived)
    assert [row["base_id"] for row in derived_rows] == [
        row["base_id"] for row in source_rows]
    assert [row["channel_draw_id"] for row in derived_rows] == [
        row["channel_draw_id"] for row in source_rows]
    for source_row, derived_row in zip(source_rows, derived_rows):
        assert (source.parent / source_row["audio"]).read_bytes() == (
            derived.parent / derived_row["audio"]).read_bytes()
        step = derived_row["gen"]["channel"]["steps"][-1]
        assert step["primitive"] == "residual_translate"
        assert step["alpha"] == 0.0 and step["checkpoint_step"] == 12


def test_noise_rows_skip_fastcut_translation(tmp_path):
    config, base_manifest = _build_base(tmp_path, n=2, noise=1.0)
    source = paired.build_view(base_manifest.parent, "clean", config,
                               tmp_path / "noise-source", seed=5,
                               keep_preloudness=True)
    checkpoint = save_generator(tmp_path / "toy.pt", ResidualGenerator(base=2, n_res=1))
    derived = paired.build_view(
        base_manifest.parent, "calibrated_fastcut", config,
        tmp_path / "noise-derived", seed=6, derive_from=source.parent,
        checkpoint=checkpoint, alpha="1", apply_prob=1.0)
    for row in _rows(derived):
        step = row["gen"]["channel"]["steps"][-1]
        assert row["kind"] == "noise" and step["applied"] is False


def test_qc_failure_is_recorded_once_without_regeneration(tmp_path, monkeypatch):
    config, base_manifest = _build_base(tmp_path, n=3)
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        return QCResult(False, "level", {"duration": 1.0})

    monkeypatch.setattr(paired, "qc_sample", reject)
    manifest = paired.build_view(base_manifest.parent, "clean", config,
                                 tmp_path / "qc", seed=7)
    assert len(calls) == 3
    assert all(row["gen"]["qc"] == {
        "ok": False, "reason": "level", "attempts": 1} for row in _rows(manifest))


def test_base_artifacts_include_resolved_config_and_hashes(tmp_path):
    _, manifest = _build_base(tmp_path, n=2)
    hashes = json.loads((manifest.parent / "hashes.json").read_text())
    assert (manifest.parent / "config.resolved.yaml").exists()
    assert len(hashes["config_hash"]) == 64
    assert len(hashes["clean_sha256"]) == 2
    wav, sr = sf.read(manifest.parent / _rows(manifest)[0]["audio_clean"])
    assert sr == FakeTTS.sample_rate and len(wav)
