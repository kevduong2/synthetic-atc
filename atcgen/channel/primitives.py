"""Pure DSP primitives for the VHF airband radio channel.

One function per effect, all with the signature
``effect(x, sr, rng, **params) -> np.ndarray``: no state, no draws outside
`rng` (numpy generators are derived from it), the input array is never
mutated.  `chain.py` composes them from config; the `PRIMITIVES` registry maps
config `primitive:` names to functions.

Ported from the retired `dsp.py` — same math, same parameter meanings.  Ranges
that dsp.py drew inside the effect (click length, crackle burst size, ...) stay
inside the effect, exposed as parameters so config can move them later.
"""

import io
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 16000


def _np_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.getrandbits(32))


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = np.gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def pink_noise(n: int, rng_np: np.random.Generator) -> np.ndarray:
    white = rng_np.standard_normal(n + 1)
    # -3 dB/octave via cumulative filter approximation (Voss-ish IIR)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = signal.lfilter(b, a, white)[1:]
    return (pink / (np.abs(pink).max() + 1e-9)).astype(np.float32)


class NoiseBank:
    """Real ATC noise beds (see `real_atc.export_noise_beds`) served as random crops."""

    def __init__(self, wav_dir: str | Path, sr: int = TARGET_SR):
        self.clips = []
        for path in sorted(Path(wav_dir).glob("*.wav")):
            wav, file_sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if file_sr != sr:
                wav = resample(wav, file_sr, sr)
            if len(wav) > 0:
                self.clips.append(wav.astype(np.float32))
        if not self.clips:
            raise ValueError(f"no noise-bed wavs in {wav_dir}")

    def sample(self, n: int, rng: random.Random) -> np.ndarray:
        clip = rng.choice(self.clips)
        if len(clip) < n:
            clip = np.tile(clip, n // len(clip) + 1)
        start = rng.randrange(0, len(clip) - n + 1) if len(clip) > n else 0
        return clip[start:start + n]


def narrowband_roundtrip(x: np.ndarray, sr: int, rng: random.Random,
                         narrow_sr: int = 8000) -> np.ndarray:
    """Decimate to a narrower rate and back (transmitter band-limiting + aliasing)."""
    narrow_sr = int(narrow_sr)
    if narrow_sr >= sr:
        return x.astype(np.float32)
    return resample(resample(x, sr, narrow_sr), narrow_sr, sr)


def bandpass(x: np.ndarray, sr: int, rng: random.Random,
             low: float = 300.0, high: float = 3400.0) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def agc_wander(x: np.ndarray, sr: int, rng: random.Random,
               strength: float = 0.0) -> np.ndarray:
    """Slow gain pumping from the receiver AGC."""
    if strength <= 0:
        return x.astype(np.float32)
    t = np.arange(len(x)) / sr
    rate = rng.uniform(0.3, 1.5)
    wander = 1.0 + strength * 0.3 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.28))
    return (x * wander).astype(np.float32)


def am_distortion(x: np.ndarray, sr: int, rng: random.Random,
                  depth: float = 0.0) -> np.ndarray:
    """Even-harmonic distortion of an over-modulated AM carrier."""
    if depth <= 0:
        return x.astype(np.float32)
    return (x + depth * x * x * np.sign(x)).astype(np.float32)


def soft_clip(x: np.ndarray, sr: int, rng: random.Random,
              drive: float = 1.0) -> np.ndarray:
    return (np.tanh(drive * x) / np.tanh(drive)).astype(np.float32)


def dropouts(x: np.ndarray, sr: int, rng: random.Random, dropout_prob: float = 1.0,
             count_lam: float = 1.5, min_ms: float = 10.0, max_ms: float = 50.0,
             atten_max: float = 0.1) -> np.ndarray:
    """Brief signal loss; `dropout_prob` is the chance this clip has any at all."""
    if rng.random() >= dropout_prob:
        return x.astype(np.float32)
    y = x.astype(np.float32).copy()
    n = len(y)
    rng_np = _np_rng(rng)
    lo, hi = int(sr * min_ms / 1000), int(sr * max_ms / 1000)
    for _ in range(1 + int(rng_np.poisson(count_lam))):
        start = rng.randrange(0, max(1, n - hi))
        y[start:start + rng.randint(lo, hi)] *= rng.uniform(0.0, atten_max)
    return y


def additive_noise(x: np.ndarray, sr: int, rng: random.Random, snr_db: float = 20.0,
                   color: str = "pink", pad: int = 0, noise_bank: NoiseBank | None = None,
                   bed_prob: float = 0.6) -> np.ndarray:
    """Static at `snr_db` relative to x[pad:-pad] (the speech, excluding padding)."""
    n = len(x)
    core = x[pad:n - pad] if 0 < pad < n // 2 else x
    sig_pow = float(np.mean(core ** 2)) + 1e-12
    rng_np = _np_rng(rng)
    if noise_bank is not None and rng.random() < bed_prob:
        noise = noise_bank.sample(n, rng)
    elif color == "pink":
        noise = pink_noise(n, rng_np)
    else:
        noise = rng_np.standard_normal(n).astype(np.float32)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return (x + noise * np.sqrt(noise_pow)).astype(np.float32)


def hum(x: np.ndarray, sr: int, rng: random.Random, amp: float = 0.0) -> np.ndarray:
    """Mains hum at 50/60 Hz plus its second harmonic."""
    if amp <= 0:
        return x.astype(np.float32)
    t = np.arange(len(x)) / sr
    f0 = rng.choice([60.0, 50.0])
    tone = amp * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t))
    return (x + tone).astype(np.float32)


def crackle(x: np.ndarray, sr: int, rng: random.Random, rate: float = 0.0,
            min_len: int = 8, max_len: int = 40, amp_low: float = 0.05,
            amp_high: float = 0.3) -> np.ndarray:
    """Short impulsive bursts at `rate` events per second."""
    n = len(x)
    if rate <= 0 or n <= max_len:
        return x.astype(np.float32)
    y = x.astype(np.float32).copy()
    rng_np = _np_rng(rng)
    for _ in range(int(rng_np.poisson(rate * n / sr))):
        pos = rng.randrange(0, n - max_len)
        burst = (rng_np.standard_normal(rng.randint(min_len, max_len))
                 * rng.uniform(amp_low, amp_high)).astype(np.float32)
        y[pos:pos + len(burst)] += burst
    return y


def heterodyne(x: np.ndarray, sr: int, rng: random.Random, f_low: float = 800.0,
               f_high: float = 2500.0, amp_low: float = 0.01,
               amp_high: float = 0.04) -> np.ndarray:
    """Steady whine from a second carrier beating against the first."""
    t = np.arange(len(x)) / sr
    f = rng.uniform(f_low, f_high)
    return (x + (rng.uniform(amp_low, amp_high) * np.sin(2 * np.pi * f * t))).astype(np.float32)


def squelch_clicks(x: np.ndarray, sr: int, rng: random.Random, min_len: int = 60,
                   max_len: int = 200, amp_low: float = 0.2,
                   amp_high: float = 0.5) -> np.ndarray:
    """Decaying noise bursts at PTT open/close, one at each end of the clip."""
    y = x.astype(np.float32).copy()
    n = len(y)
    rng_np = _np_rng(rng)
    for head in (True, False):
        click_len = min(rng.randint(min_len, max_len), n)
        click = (rng_np.standard_normal(click_len) * rng.uniform(amp_low, amp_high)).astype(np.float32)
        env = np.exp(-np.linspace(0, 5, click_len)).astype(np.float32)
        if head:
            y[:click_len] += click * env
        else:
            y[-click_len:] += click * env[::-1]
    return y


def cochannel_mix(x: np.ndarray, sr: int, rng: random.Random, level: float = 0.0,
                  interference: np.ndarray | None = None) -> np.ndarray:
    """A second, fainter transmission underneath this one."""
    if level <= 0 or interference is None:
        return x.astype(np.float32)
    n = len(x)
    other = np.asarray(interference, dtype=np.float32)
    if len(other) < n:
        other = np.pad(other, (0, n - len(other)))
    return (x + level * other[:n]).astype(np.float32)


def codec_roundtrip(x: np.ndarray, sr: int, rng: random.Random,
                    compression_level: float = 0.9) -> np.ndarray:
    """Encode/decode through low-bitrate CBR MP3 (LiveATC-style delivery)."""
    peak = np.abs(x).max()
    if peak > 1.0:
        x = x / peak * 0.98
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="MP3", bitrate_mode="CONSTANT",
             compression_level=compression_level)
    buf.seek(0)
    y, _ = sf.read(buf, dtype="float32")
    n = len(x)
    if len(y) > n:  # encoder delay pads the start; realign roughly
        y = y[len(y) - n:]
    elif len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


PRIMITIVES = {
    "narrowband_roundtrip": narrowband_roundtrip,
    "bandpass": bandpass,
    "agc_wander": agc_wander,
    "am_distortion": am_distortion,
    "soft_clip": soft_clip,
    "dropouts": dropouts,
    "additive_noise": additive_noise,
    "hum": hum,
    "crackle": crackle,
    "heterodyne": heterodyne,
    "squelch_clicks": squelch_clicks,
    "cochannel_mix": cochannel_mix,
    "codec_roundtrip": codec_roundtrip,
}
