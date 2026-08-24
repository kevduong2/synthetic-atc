"""Fast end-to-end coverage for the real + synthetic expansion workflow."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from atcgen.config import load_config
from atcgen.dataset import expand as expand_mod
from atcgen.dataset.build import build_dataset as real_build_dataset


LIGHT_CHANNEL = """
channel:
  profile: test
  clean_arm_prob: 0.0
  chain:
    - primitive: bandpass
      prob: 1.0
      low: 300
      high: 3400
    - primitive: additive_noise
      prob: 1.0
      snr_db: 20
      color: white
"""


class FakeTTS:
    sample_rate = 24000

    def synthesize(self, text, rng, voice="af_heart", speed=1.0):
        seconds = (0.8 + min(len(text), 40) * 0.01) / speed
        t = np.arange(int(self.sample_rate * seconds)) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    manifest = real_dir / "manifest.jsonl"
    rows = []
    stations = ["KSDL_TOWER", "KSLE_GROUND", "SEATTLE_CENTER"]
    sr = 16000
    t = np.arange(sr) / sr
    for station_index, station in enumerate(stations):
        for clip_index in range(10):
            name = f"{station}_20260101_12{station_index:02d}{clip_index:02d}.wav"
            sf.write(real_dir / name,
                     0.2 * np.sin(2 * np.pi * (180 + clip_index) * t), sr)
            category = "emergency" if clip_index == 0 else "routine"
            rows.append({
                "audio": name,
                "text": f"{station} real transcript {clip_index}",
                "role": "controller" if clip_index % 2 else "pilot",
                "kind": "alert" if category == "emergency" else "clearance",
                "category": category,
            })
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    external = tmp_path / "external.jsonl"
    external.write_text("".join(
        json.dumps({"text": f"external emergency phrase {index}",
                    "category": "emergency", "kind": "alert"}) + "\n"
        for index in range(5)
    ))
    return manifest, external


def test_expand_end_to_end_with_real_split_quotas_and_provenance(tmp_path, monkeypatch):
    real_manifest, external = _make_inputs(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mode: procedural\n"
        "seed: 7\n"
        "dataset: {noise_only_frac: 0.0, category_quotas: {}}\n"
        "voice_augment:\n"
        "  pitch_semitones: {prob: 0.0, const: 0}\n"
        "  tempo: {prob: 0.0, const: 1}\n"
        "  eq_tilt_db: {prob: 0.0, const: 0}\n"
        "calibrated:\n"
        "  expansion:\n"
        f"    real_manifest: {real_manifest}\n"
        "    target_total: 55\n"
        "    category_quotas: {emergency: 0.30}\n"
        "    holdout_frac: 0.20\n"
        f"    external_texts: {external}\n"
        + LIGHT_CHANNEL
    )
    config = load_config(config_path)

    def build_with_fake_tts(*args, **kwargs):
        return real_build_dataset(*args, **kwargs, tts=FakeTTS())

    monkeypatch.setattr(expand_mod, "build_dataset", build_with_fake_tts)
    manifest = expand_mod.expand(config, tmp_path / "expanded", target_total=60)

    combined = _read(manifest)
    holdout = _read(manifest.parent / "holdout_manifest.jsonl")
    real_train = [record for record in combined if record["origin"] == "real"]
    synthetic = [record for record in combined if record["origin"] == "synthetic"]

    assert len(holdout) == 6
    assert len(real_train) == 24
    assert len(synthetic) == 36
    assert len(combined) == 60
    assert {record["station"] for record in holdout} == {
        "KSDL_TOWER", "KSLE_GROUND", "SEATTLE_CENTER"
    }
    assert {record["audio"] for record in holdout}.isdisjoint(
        record["audio"] for record in real_train)
    assert all(Path(record["audio"]).is_absolute() for record in real_train + holdout)
    assert all(record["audio"].startswith("synthetic/wavs/") for record in synthetic)
    assert all(record["gen"]["mode"] == config.mode for record in synthetic)

    allowed_texts = {record["text"] for record in real_train}
    allowed_texts.update(record["text"] for record in _read(external))
    assert {record["text"] for record in synthetic} <= allowed_texts
    holdout_texts = {record["text"] for record in holdout}
    assert not ({record["text"] for record in synthetic} & holdout_texts)

    emergency_fraction = sum(record["category"] == "emergency" for record in combined) / 60
    assert emergency_fraction == pytest.approx(0.30, abs=0.08)

    stats = json.loads((manifest.parent / "expand_stats.json").read_text())
    assert stats["target_total"] == 60
    assert stats["real_input_count"] == 30
    assert stats["real_train_count"] == 24
    assert stats["synthetic_count"] == 36
    assert stats["combined_count"] == 60
    assert stats["holdout_size"] == 6
    assert stats["category_quotas"]["targets"] == {"emergency": 0.3}
    assert stats["category_quotas"]["achieved"]["emergency"] == \
        pytest.approx(emergency_fraction, abs=0.0001)
    assert stats["tier0"]["discard_rate"] < 0.15
    assert stats["tier0"] == json.loads(
        (manifest.parent / "synthetic" / "stats.json").read_text())["qc"]
