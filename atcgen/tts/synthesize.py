"""TTS engines for rendering ATC utterances.

Engine-agnostic interface: `synthesize(text, rng) -> (waveform float32 mono, sr)`.
KokoroTTS is the default (Apache-2.0, ~82M params, fast on CPU/MPS, many
voices). F5-TTS voice cloning can be added later behind the same interface.
"""

import random
from typing import Protocol

import numpy as np

SAMPLE_RATE = 24000  # Kokoro native rate; channel sim resamples to 16k


class TTSEngine(Protocol):
    sample_rate: int

    def synthesize(self, text: str, rng: random.Random) -> np.ndarray: ...


# American + British English voices shipped with Kokoro that sound adult and
# clear over a degraded channel. Rate perturbation approximates fast
# controller delivery.
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "am_eric", "am_onyx",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]


class KokoroTTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, voices: list[str] | None = None, speed_range: tuple[float, float] = (0.95, 1.55)):
        from kokoro import KPipeline
        # lang_code 'a' = American English; British voices still render.
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        self.voices = voices or KOKORO_VOICES
        self.speed_range = speed_range

    def synthesize(self, text: str, rng: random.Random) -> np.ndarray:
        voice = rng.choice(self.voices)
        speed = rng.uniform(*self.speed_range)
        chunks = []
        for result in self.pipeline(text, voice=voice, speed=speed):
            audio = result.audio
            if audio is not None:
                chunks.append(audio.detach().cpu().numpy().astype(np.float32))
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for: {text!r}")
        wav = np.concatenate(chunks)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.9
        return wav
