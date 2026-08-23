"""Voice-level variation applied to clean TTS audio before the channel."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import librosa
import numpy as np
from scipy import signal

from ..config import DistSpec, VoiceAugmentConfig


class VoiceConverter(Protocol):
    """Interface for a future accent/voice-conversion stage.

    An accent converter can be plugged into :class:`VoiceAugment` without
    changing the TTS or channel interfaces. It is intentionally unspecified
    for Mode 1 and, when provided, runs after the procedural voice effects.
    """

    def convert(self, wav: np.ndarray, sr: int,
                rng: random.Random) -> np.ndarray: ...


@dataclass
class VoiceAugment:
    """Pitch, tempo, and timbre variation for clean mono TTS waveforms."""

    pitch_semitones: DistSpec
    tempo: DistSpec
    eq_tilt_db: DistSpec
    vc: VoiceConverter | None = None

    @classmethod
    def from_config(cls, config: VoiceAugmentConfig,
                    vc: VoiceConverter | None = None) -> "VoiceAugment":
        return cls(config.pitch_semitones, config.tempo, config.eq_tilt_db, vc)

    def __call__(self, wav: np.ndarray, sr: int,
                 rng: random.Random) -> tuple[np.ndarray, dict]:
        """Apply independently gated effects and return their sampled values."""
        pitch = self.pitch_semitones.sample(rng)
        tempo = self.tempo.sample(rng)
        tilt = self.eq_tilt_db.sample(rng)
        record = {"pitch": pitch, "tempo": tempo, "eq_tilt_db": tilt}

        x = np.asarray(wav, dtype=np.float32)
        if x.size and pitch is not None and abs(float(pitch)) > 1e-12:
            x = librosa.effects.pitch_shift(
                y=x, sr=sr, n_steps=float(pitch)).astype(np.float32)
        if x.size and tempo is not None and abs(float(tempo) - 1.0) > 1e-12:
            x = librosa.effects.time_stretch(
                y=x, rate=float(tempo)).astype(np.float32)
        if x.size and tilt is not None and abs(float(tilt)) > 1e-12:
            x = _eq_tilt(x, sr, float(tilt))
        if self.vc is not None:
            x = np.asarray(self.vc.convert(x, sr, rng), dtype=np.float32)
        return x.astype(np.float32, copy=False), record


def _shelf_sos(sr: int, pivot_hz: float, gain_db: float,
               *, high: bool) -> list[float]:
    """RBJ shelving biquad as one row accepted by ``scipy.signal.sosfilt``."""
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * min(max(pivot_hz, 20.0), 0.45 * sr) / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2.0
    root = 2.0 * np.sqrt(a_gain) * alpha
    sign = 1.0 if high else -1.0
    b = [a_gain * ((a_gain + 1.0) + sign * (a_gain - 1.0) * cos_w0 + root),
         -2.0 * sign * a_gain
         * ((a_gain - 1.0) + sign * (a_gain + 1.0) * cos_w0),
         a_gain * ((a_gain + 1.0) + sign * (a_gain - 1.0) * cos_w0 - root)]
    a = [(a_gain + 1.0) - sign * (a_gain - 1.0) * cos_w0 + root,
         2.0 * sign * ((a_gain - 1.0) - sign * (a_gain + 1.0) * cos_w0),
         (a_gain + 1.0) - sign * (a_gain - 1.0) * cos_w0 - root]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0],
            1.0, a[1] / a[0], a[2] / a[0]]


def _eq_tilt(wav: np.ndarray, sr: int, tilt_db: float,
             pivot_hz: float = 1000.0) -> np.ndarray:
    """Apply a complementary low/high shelf tilt around ``pivot_hz``."""
    sections = np.asarray([
        _shelf_sos(sr, pivot_hz, tilt_db / 2.0, high=True),
        _shelf_sos(sr, pivot_hz, -tilt_db / 2.0, high=False),
    ])
    return signal.sosfilt(sections, wav).astype(np.float32)
