"""Fast synthetic checks for the clean-TTS voice augmentation stage."""

import json
import random

import numpy as np
import pytest

from atcgen.config import DistSpec, VoiceAugmentConfig, load_config
from atcgen.dataset.build import build_dataset
from atcgen.text.grammar import Utterance
from atcgen.tts.augment import VoiceAugment

SR = 24000


def _augment(pitch=0.0, tempo=1.0, tilt=0.0) -> VoiceAugment:
    config = VoiceAugmentConfig(
        pitch_semitones=DistSpec.parse(pitch),
        tempo=DistSpec.parse(tempo),
        eq_tilt_db=DistSpec.parse(tilt),
    )
    return VoiceAugment.from_config(config)


def _tone(freq: float, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(int(SR * seconds)) / SR
    return np.sin(2.0 * np.pi * freq * time).astype(np.float32)


def _dominant_frequency(wav: np.ndarray) -> float:
    windowed = wav * np.hanning(len(wav))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(wav), 1.0 / SR)
    return float(frequencies[np.argmax(spectrum)])


def test_pitch_plus_twelve_semitones_doubles_dominant_frequency():
    shifted, record = _augment(pitch=12.0)(_tone(440.0), SR, random.Random(0))

    assert _dominant_frequency(shifted) == pytest.approx(880.0, abs=8.0)
    assert record == {"pitch": 12.0, "tempo": 1.0, "eq_tilt_db": 0.0}
    assert shifted.dtype == np.float32


def test_tempo_half_doubles_duration():
    wav = _tone(440.0)
    stretched, record = _augment(tempo=0.5)(wav, SR, random.Random(1))

    assert len(stretched) == pytest.approx(2 * len(wav), abs=1)
    assert record["tempo"] == 0.5


def test_eq_tilt_changes_spectral_slope_sign():
    time = np.arange(2 * SR) / SR
    wav = (np.sin(2.0 * np.pi * 250.0 * time)
           + np.sin(2.0 * np.pi * 4000.0 * time)).astype(np.float32)

    def high_minus_low_db(tilt: float) -> float:
        shaped, _ = _augment(tilt=tilt)(wav, SR, random.Random(2))
        spectrum = np.abs(np.fft.rfft(shaped * np.hanning(len(shaped))))
        frequencies = np.fft.rfftfreq(len(shaped), 1.0 / SR)
        low = spectrum[np.argmin(np.abs(frequencies - 250.0))]
        high = spectrum[np.argmin(np.abs(frequencies - 4000.0))]
        return float(20.0 * np.log10(high / low))

    assert high_minus_low_db(12.0) > 6.0
    assert high_minus_low_db(-12.0) < -6.0


def test_probability_gates_return_identical_audio_and_none_record():
    off = DistSpec.parse({"prob": 0.0, "const": 9.0})
    augment = VoiceAugment(off, off, off)
    wav = _tone(330.0)

    result, record = augment(wav, SR, random.Random(7))

    assert np.array_equal(result, wav)
    assert record == {"pitch": None, "tempo": None, "eq_tilt_db": None}


class _FakeTTS:
    sample_rate = SR

    def synthesize(self, text, rng, voice="af_heart", speed=1.0):
        return _tone(220.0, seconds=1.0 / speed)


class _TextSource:
    def sample(self, rng):
        return Utterance("alpha one", "alpha one", "controller", "routine")


def test_builder_manifest_carries_voice_augment_draws(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mode: procedural\n"
        "seed: 4\n"
        "output: {loudness_db: null}\n"
        "tts: {voices: [af_heart], speed: 1.0}\n"
        "voice_augment:\n"
        "  pitch_semitones: {prob: 1.0, const: 1.25}\n"
        "  tempo: {prob: 1.0, const: 0.9}\n"
        "  eq_tilt_db: {prob: 1.0, const: -2.5}\n"
        "dataset: {noise_only_frac: 0.0, pilot_double_hop_prob: 0.0}\n"
        "qc: {enabled: false}\n"
        "channel: {profile: test, clean_arm_prob: 0.0, chain: []}\n"
    )

    manifest = build_dataset(load_config(config_path), tmp_path / "out", 1,
                             _TextSource(), _FakeTTS())
    row = json.loads(manifest.read_text())

    assert {name: row["gen"][name]
            for name in ("pitch", "tempo", "eq_tilt_db")} == {
                "pitch": 1.25, "tempo": 0.9, "eq_tilt_db": -2.5}
