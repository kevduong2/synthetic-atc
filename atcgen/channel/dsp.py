"""Parametric VHF airband radio channel simulator.

Applies a randomized degradation chain to clean speech, approximating what a
tower/ground AM transmission sounds like at the receiver:

  narrowband resample -> bandpass 300-3400 Hz -> AGC pumping -> soft clip /
  AM harmonic distortion -> additive static at randomized SNR -> hum /
  crackle -> squelch clicks at PTT open/close -> occasional dropouts,
  heterodyne tone, co-channel interference -> lossy codec round-trip
  (ATCO2/LiveATC-style low-bitrate MP3 delivery).

All parameters are drawn per-sample so a large dataset spans many "radios".
Pilot transmissions can be run through the chain twice (`hops=2`) to model
the ground-relay double hop. Noise can be drawn from real ATC noise beds
(`NoiseBank`) instead of synthetic static.
Output is 16 kHz mono float32 in [-1, 1].
"""

import io
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 16000


@dataclass
class ChannelParams:
    narrowband_sr: int
    bp_low: float
    bp_high: float
    snr_db: float
    noise_color: str          # "white" | "pink"
    clip_drive: float         # soft clip pre-gain
    am_depth: float           # harmonic distortion amount
    agc_strength: float
    hum_amp: float
    crackle_rate: float       # events per second
    squelch_click: bool
    dropout_prob: float       # chance this transmission has dropouts at all
    heterodyne: bool
    cochannel_level: float    # 0 = none
    codec_level: float        # MP3 compression_level (0 = no codec round-trip)

    @classmethod
    def sample(cls, rng: random.Random) -> "ChannelParams":
        return cls(
            narrowband_sr=rng.choice([6000, 8000, 8000, 11025]),
            bp_low=rng.uniform(250, 400),
            bp_high=rng.uniform(2800, 3600),
            snr_db=rng.uniform(3, 25),
            noise_color=rng.choice(["white", "pink", "pink"]),
            clip_drive=rng.uniform(1.0, 4.0),
            am_depth=rng.uniform(0.0, 0.25),
            agc_strength=rng.uniform(0.0, 0.6),
            hum_amp=0.0 if rng.random() < 0.7 else rng.uniform(0.002, 0.01),
            crackle_rate=rng.uniform(0.0, 6.0),
            squelch_click=rng.random() < 0.8,
            dropout_prob=rng.uniform(0.0, 0.3),
            heterodyne=rng.random() < 0.08,
            cochannel_level=rng.uniform(0.05, 0.2) if rng.random() < 0.1 else 0.0,
            codec_level=rng.uniform(0.75, 0.95) if rng.random() < 0.5 else 0.0,
        )

    @classmethod
    def mild(cls, rng: random.Random) -> "ChannelParams":
        """Light-touch variant for post-GAN diversification: the GAN already
        supplies spectral realism, this just varies SNR/band/codec per sample."""
        return cls(
            narrowband_sr=rng.choice([8000, 11025, 16000]),
            bp_low=rng.uniform(200, 320),
            bp_high=rng.uniform(3000, 3800),
            snr_db=rng.uniform(15, 30),
            noise_color=rng.choice(["white", "pink"]),
            clip_drive=rng.uniform(1.0, 1.5),
            am_depth=0.0,
            agc_strength=rng.uniform(0.0, 0.2),
            hum_amp=0.0,
            crackle_rate=rng.uniform(0.0, 1.0),
            squelch_click=rng.random() < 0.8,
            dropout_prob=0.0,
            heterodyne=False,
            cochannel_level=0.0,
            codec_level=rng.uniform(0.75, 0.95) if rng.random() < 0.5 else 0.0,
        )


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    g = np.gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def _codec_roundtrip(x: np.ndarray, sr: int, compression_level: float) -> np.ndarray:
    """Encode/decode through low-bitrate CBR MP3 (LiveATC-style delivery)."""
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


class NoiseBank:
    """Real ATC noise beds (see `real_atc.export_noise_beds`) served as random crops."""

    def __init__(self, wav_dir: str | Path, sr: int = TARGET_SR):
        self.clips = []
        for path in sorted(Path(wav_dir).glob("*.wav")):
            wav, file_sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if file_sr != sr:
                wav = _resample(wav, file_sr, sr)
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


def _pink_noise(n: int, rng_np: np.random.Generator) -> np.ndarray:
    white = rng_np.standard_normal(n + 1)
    # -3 dB/octave via cumulative filter approximation (Voss-ish IIR)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = signal.lfilter(b, a, white)[1:]
    return (pink / (np.abs(pink).max() + 1e-9)).astype(np.float32)


class RadioChannelSim:
    def __init__(self, target_sr: int = TARGET_SR, noise_bank: NoiseBank | None = None):
        self.target_sr = target_sr
        self.noise_bank = noise_bank

    def __call__(self, wav: np.ndarray, sr: int, rng: random.Random,
                 params: ChannelParams | None = None,
                 interference: np.ndarray | None = None,
                 hops: int = 1) -> tuple[np.ndarray, ChannelParams]:
        """Degrade `wav` (float32 mono at `sr`). Returns (16 kHz wav, params used).

        `hops=2` runs the transmission chain twice with independently sampled
        radios (pilot audio relayed through a ground station); squelch clicks,
        co-channel interference and the codec apply once at the receiving end.
        """
        p = params or ChannelParams.sample(rng)
        rng_np = np.random.default_rng(rng.getrandbits(32))
        tsr = self.target_sr

        x = wav.astype(np.float32)
        if sr != tsr:
            x = _resample(x, sr, tsr)

        # pad with a little silence so squelch clicks and noise floor frame the speech
        pad = int(tsr * 0.15)
        x = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad, np.float32)])

        hop_params = p
        for hop in range(hops):
            if hop > 0:
                # second radio in the relay: fresh params, quieter noise floor
                hop_params = replace(ChannelParams.sample(rng),
                                     snr_db=rng.uniform(10, 25))
            x = self._hop(x, hop_params, rng, rng_np, pad)
        n = len(x)

        # co-channel interference: a second, fainter transmission underneath
        if p.cochannel_level > 0 and interference is not None:
            i = interference.astype(np.float32)
            if len(i) < n:
                i = np.pad(i, (0, n - len(i)))
            x = x + p.cochannel_level * i[:n]

        # squelch clicks at PTT open/close
        if p.squelch_click:
            for pos in (0, n - 1):
                click_len = rng.randint(60, 200)
                click = (rng_np.standard_normal(click_len) * rng.uniform(0.2, 0.5)).astype(np.float32)
                env = np.exp(-np.linspace(0, 5, click_len)).astype(np.float32)
                if pos == 0:
                    x[:click_len] += click * env
                else:
                    x[-click_len:] += click * env[::-1]

        # lossy delivery codec (LiveATC-style MP3 stream)
        if p.codec_level > 0:
            peak = np.abs(x).max()
            if peak > 1.0:
                x = x / peak * 0.98
            x = _codec_roundtrip(x, tsr, p.codec_level)

        peak = np.abs(x).max()
        if peak > 1.0:
            x = x / peak * 0.98
        return x.astype(np.float32), p

    def _hop(self, x: np.ndarray, p: ChannelParams, rng: random.Random,
             rng_np: np.random.Generator, pad: int) -> np.ndarray:
        """One transmit/receive pass: band-limiting, distortion, noise, artifacts."""
        tsr = self.target_sr

        # narrowband round trip
        if p.narrowband_sr < tsr:
            x = _resample(_resample(x, tsr, p.narrowband_sr), p.narrowband_sr, tsr)
            # length can drift by a sample or two
        n = len(x)

        # bandpass
        sos = signal.butter(4, [p.bp_low, p.bp_high], btype="bandpass", fs=tsr, output="sos")
        x = signal.sosfilt(sos, x).astype(np.float32)

        # AGC pumping: slow gain wander
        if p.agc_strength > 0:
            t = np.arange(n) / tsr
            wander = 1.0 + p.agc_strength * 0.3 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t + rng.uniform(0, 6.28))
            x = x * wander.astype(np.float32)

        # AM-style harmonic distortion + soft clipping
        if p.am_depth > 0:
            x = x + p.am_depth * x * x * np.sign(x)
        x = np.tanh(p.clip_drive * x) / np.tanh(p.clip_drive)

        # dropouts (brief signal loss)
        if rng.random() < p.dropout_prob:
            n_drops = 1 + rng_np.poisson(1.5)
            for _ in range(n_drops):
                start = rng.randrange(0, max(1, n - 800))
                length = rng.randint(160, 800)  # 10-50 ms
                x[start:start + length] *= rng.uniform(0.0, 0.1)

        # additive noise at target SNR: real ATC noise bed when available
        sig_pow = float(np.mean(x[pad:n - pad] ** 2)) + 1e-12
        if self.noise_bank is not None and rng.random() < 0.6:
            noise = self.noise_bank.sample(n, rng)
        elif p.noise_color == "pink":
            noise = _pink_noise(n, rng_np)
        else:
            noise = rng_np.standard_normal(n).astype(np.float32)
        noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
        noise_pow = sig_pow / (10 ** (p.snr_db / 10))
        x = x + noise * np.sqrt(noise_pow)

        # mains hum
        if p.hum_amp > 0:
            t = np.arange(n) / tsr
            f0 = rng.choice([60.0, 50.0])
            hum = p.hum_amp * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t))
            x = x + hum.astype(np.float32)

        # crackle: short impulsive bursts
        n_crackles = rng_np.poisson(p.crackle_rate * n / tsr)
        for _ in range(int(n_crackles)):
            pos = rng.randrange(0, n - 40)
            burst = (rng_np.standard_normal(rng.randint(8, 40)) * rng.uniform(0.05, 0.3)).astype(np.float32)
            x[pos:pos + len(burst)] += burst

        # heterodyne whine
        if p.heterodyne:
            t = np.arange(n) / tsr
            f = rng.uniform(800, 2500)
            x = x + (rng.uniform(0.01, 0.04) * np.sin(2 * np.pi * f * t)).astype(np.float32)

        return x
