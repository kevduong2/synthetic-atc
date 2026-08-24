import hashlib
import random
from pathlib import Path

import pytest

from atcgen.config import DistSpec, config_hash, dump_resolved, load_config


ROOT = Path(__file__).parents[1]


def test_dist_spec_parses_scalar_and_all_kinds():
    rng = random.Random(4)
    assert DistSpec.parse("pink").sample(rng) == "pink"
    assert DistSpec.parse({"const": 7}).sample(rng) == 7
    assert 2 <= DistSpec.parse({"uniform": [2, 3]}).sample(rng) <= 3
    assert DistSpec.parse({"choice": ["a", "a", "b"]}).sample(rng) in {"a", "b"}
    assert 10 <= DistSpec.parse({"beta_scaled": [2, 3, 10, 20]}).sample(rng) <= 20


def test_dist_spec_prob_gate_statistics():
    spec = DistSpec.parse({"prob": 0.25, "const": "on"})
    draws = [spec.sample(random.Random(seed)) for seed in range(4000)]
    fraction_on = sum(value == "on" for value in draws) / len(draws)
    assert 0.22 < fraction_on < 0.28


@pytest.mark.parametrize("value", [
    {"uniform": [0, 1], "choice": [1]},
    {"uniform": [0, 1], "wat": 2},
    {"choice": []},
    {"beta_scaled": [0, 1, 2, 3]},
    {"const": 1, "prob": 1.1},
    [1, 2],
])
def test_bad_dist_specs_raise(value):
    with pytest.raises(ValueError):
        DistSpec.parse(value)


def test_beta_scaled_stays_in_bounds():
    spec = DistSpec.parse({"beta_scaled": [0.5, 0.8, -4, 9]})
    values = [spec.sample(random.Random(seed)) for seed in range(1000)]
    assert min(values) >= -4
    assert max(values) <= 9


def test_default_profile_loads_and_resolved_dump_round_trips(tmp_path):
    config = load_config(ROOT / "configs/mode1_default.yaml")
    assert config.mode == "procedural"
    assert config.channel is not None
    assert len(config.channel.chain) == 13

    path, digest = dump_resolved(config, tmp_path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == config_hash(config)
    assert load_config(path) == config
    second_path, second_digest = dump_resolved(config, tmp_path / "again")
    assert second_digest == digest
    assert second_path.read_bytes() == path.read_bytes()


def test_unknown_key_reports_full_path(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode: procedural\noutput:\n  sample_raat: 16000\n")
    with pytest.raises(ValueError, match=r"output\.sample_raat"):
        load_config(path)


def test_channel_noise_beds_dir_parses_strictly_and_lazily(tmp_path):
    missing = tmp_path / "noise-beds"
    path = tmp_path / "noise.yaml"
    path.write_text(f"mode: procedural\nchannel:\n  noise:\n    beds_dir: {missing}\n")
    config = load_config(path)
    assert config.channel is not None
    assert config.channel.noise.beds_dir == str(missing)

    path.write_text("mode: procedural\nchannel:\n  noise:\n    bed_prob: 0.8\n")
    with pytest.raises(ValueError, match=r"unknown config key: channel\.noise\.bed_prob"):
        load_config(path)


def test_dot_path_overrides(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text("mode: procedural\nseed: 1\n")
    config = load_config(path, {"seed": 9, "output.sample_rate": 8000})
    assert config.seed == 9
    assert config.output.sample_rate == 8000


def test_mix_mode_backends_parse(tmp_path):
    path = tmp_path / "mix.yaml"
    path.write_text(
        "mode: mix\n"
        "backends:\n"
        "  - {backend: procedural, weight: 0.7}\n"
        "  - {backend: calibrated, weight: 0.3}\n"
    )
    config = load_config(path)
    assert [(item.backend, item.weight) for item in config.backends] == [
        ("procedural", 0.7), ("calibrated", 0.3),
    ]


def test_qc_defaults_keep_the_asr_gate_off(tmp_path):
    path = tmp_path / "plain.yaml"
    path.write_text("mode: procedural\n")
    qc = load_config(path).qc
    assert qc.enabled is True
    assert qc.asr_roundtrip is False        # loading Whisper per run is opt-in
    assert qc.max_retries == 3
    assert (qc.min_duration, qc.max_duration) == (0.5, 30.0)


def test_qc_section_parses_and_validates(tmp_path):
    path = tmp_path / "qc.yaml"
    path.write_text("mode: procedural\n"
                    "qc: {enabled: true, asr_roundtrip: true, max_retries: 1, "
                    "min_duration: 1.0, max_duration: 12.0, max_clip_frac: 0.02, "
                    "min_rms_db: -35, max_rms_db: -10, max_wer: 0.35}\n")
    qc = load_config(path).qc
    assert (qc.asr_roundtrip, qc.max_retries, qc.max_wer) == (True, 1, 0.35)
    assert (qc.min_rms_db, qc.max_rms_db) == (-35.0, -10.0)


@pytest.mark.parametrize("body,message", [
    ("qc: {asr_roundtrip: yes_please}", r"qc\.asr_roundtrip"),
    ("qc: {max_retries: -1}", r"qc\.max_retries"),
    ("qc: {max_wer: 2.0}", r"qc\.max_wer"),
    ("qc: {min_duration: 9.0, max_duration: 2.0}", r"qc\.min_duration"),
    ("qc: {min_rms_db: -3, max_rms_db: -20}", r"qc\.min_rms_db"),
    ("qc: {asr_gate: true}", r"qc\.asr_gate"),
])
def test_bad_qc_sections_raise(tmp_path, body, message):
    path = tmp_path / "bad_qc.yaml"
    path.write_text(f"mode: procedural\n{body}\n")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_calibrated_body_parses(tmp_path):
    path = tmp_path / "mode2.yaml"
    path.write_text(
        """mode: calibrated
calibrated:
  calibration:
    corpus_dir: real
    presets: presets.jsonl
    noise_bank: noise
    station_mix: {TOWER: 1.0}
    snr_jitter_db: {uniform: [-2, 2]}
    cross_station_prob: 0.2
  residual:
    enabled: false
    checkpoint: model.pt
    apply_prob: 0.0
    residual_scale_max: 0.2
  post_effects:
    squelch: {prob: 0.7, gated_floor_prob: 0.5}
    dropouts: {prob: 0.1}
    codec: {prob: 0.4, kind: mp3, quality: {uniform: [0.8, 0.9]}}
  expansion:
    real_manifest: manifest.jsonl
    target_total: 100
    category_quotas: {emergency: 0.1}
    holdout_frac: 0.2
    external_texts: emergency.jsonl
"""
    )
    config = load_config(path)
    assert config.calibrated is not None
    assert config.calibrated.calibration.station_mix == {"TOWER": 1.0}
    assert config.calibrated.residual.enabled is False
    assert config.calibrated.expansion.target_total == 100
    assert config.calibrated.expansion.external_texts == "emergency.jsonl"
