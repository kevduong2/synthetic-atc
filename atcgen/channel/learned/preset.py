"""Fitted per-clip channel presets: the JSONL format and its numpy evaluator.

A preset is what `channel_fit` produces for one real calibration clip — the
parameters of a small physical chain that, driven with clean speech, reproduces
that clip's long-term spectrum and envelope statistics:

    x -> tanh drive -> low-order polynomial -> AGC -> EQ (band gains)
      -> normalize -> + noise at the fitted SNR

Order follows the physical path: the transmitter clips, the receiver's gain
control rides the IF, and the audio filter shapes what comes out last.  Putting
the filter last is also what makes the fit work at all — a gain riding a 5 ms
grid splatters broadband sidebands, and anything after the filter puts a floor
around -50 dB on the output spectrum, some 40 dB above the real clips'.
The EQ is stored as gains on the same log band grid the Tier 1 LTAS uses, so a
preset is directly comparable to the statistic it was fitted against; the taps
are re-derived from those gains by frequency sampling (`fir_taps`).

`channel_fit` fits in PyTorch, this module evaluates in numpy — generation never
imports torch.  `tests/test_channel_fit.py` asserts the two agree.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal

TARGET_SR = 16000
BAND_EDGES = [round(float(v), 3) for v in np.geomspace(100.0, 8000.0, 33)]
N_TAPS = 513                 # ~125 Hz transition width at 16 kHz, ~80 dB of range
AGC_HOP_MS = 5.0
ACTIVE_REL_DB = -40.0        # frames this far below the loudest are not "speech"
EPS = 1e-12


@dataclass
class Preset:
    """One fitted clip's channel, as written to `presets.jsonl`."""

    clip_id: str
    station: str
    band_gains_db: list[float]
    drive: float
    poly: list[float]                       # [c2, c3] on the tanh output
    agc_tau_ms: float
    agc_strength: float
    noise_gain: float                       # amplitude, relative to unit-RMS speech
    snr_est: float                          # -20*log10(noise_gain), for readability
    fit_loss: float
    ltas_l1_db: float = 0.0                 # fit QC: per-band |synth - real| median
    band_edges_hz: list[float] = field(default_factory=lambda: list(BAND_EDGES))
    passband_hz: list[float] = field(default_factory=list)   # -6 dB band edges
    split: str = "train"
    duration: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preset":
        names = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"unknown preset field(s): {', '.join(unknown)}")
        return cls(**data)


def write_presets(path: str | Path, presets: Iterable[Preset]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for preset in presets:
            handle.write(json.dumps(preset.as_dict()) + "\n")
    return out


def load_presets(path: str | Path) -> list[Preset]:
    presets = [Preset.from_dict(json.loads(line))
               for line in Path(path).read_text().splitlines() if line.strip()]
    if not presets:
        raise ValueError(f"no presets in {path}")
    return presets


def band_centers(edges: Iterable[float]) -> np.ndarray:
    """Geometric centres of the band grid — the EQ's control frequencies."""
    e = np.asarray(list(edges), dtype=np.float64)
    return np.sqrt(e[:-1] * e[1:])


def response_db(band_gains_db, band_edges_hz, freqs_hz) -> np.ndarray:
    """The EQ's gain at `freqs_hz`: band gains interpolated in log-frequency.

    Held flat outside the grid, which at 16 kHz means only below 100 Hz — the
    grid's top edge is Nyquist.
    """
    centres = band_centers(band_edges_hz)
    f = np.maximum(np.asarray(freqs_hz, dtype=np.float64), 1.0)
    return np.interp(np.log(f), np.log(centres),
                     np.asarray(band_gains_db, dtype=np.float64))


def fir_taps(band_gains_db, band_edges_hz, sr: int = TARGET_SR,
             n_taps: int = N_TAPS) -> np.ndarray:
    """Linear-phase FIR realizing the band gains, by frequency sampling.

    Zero phase by construction: only magnitude statistics are fitted, and a
    symmetric response keeps the fit's torch mirror trivially identical.
    """
    n_fft = 1 << int(np.ceil(np.log2(max(n_taps * 4, 1024))))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    magnitude = 10.0 ** (response_db(band_gains_db, band_edges_hz, freqs) / 20.0)
    impulse = np.fft.irfft(magnitude, n=n_fft)
    half = n_taps // 2
    taps = np.concatenate([impulse[-half:], impulse[:half + 1]])
    return (taps * np.hanning(len(taps))).astype(np.float64)


def passband_edges(band_gains_db, band_edges_hz, drop_db: float = 6.0
                   ) -> tuple[float, float]:
    """Lowest and highest band centre within `drop_db` of the EQ's peak gain."""
    gains = np.asarray(band_gains_db, dtype=np.float64)
    centres = band_centers(band_edges_hz)
    inside = np.flatnonzero(gains >= gains.max() - drop_db)
    if inside.size == 0:
        return float(centres[0]), float(centres[-1])
    return float(centres[inside[0]]), float(centres[inside[-1]])


def speech_rms(x: np.ndarray, sr: int = TARGET_SR,
               rel_db: float = ACTIVE_REL_DB) -> float:
    """RMS over the audible frames only.

    Chain padding and TTS's own lead-in silence would otherwise dilute the
    reference level, and the level is what `drive` and the noise SNR are
    relative to.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    frame = max(1, int(sr * 0.020))
    if len(x) < 2 * frame:
        return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    n = len(x) // frame
    power = np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1)
    peak = power.max()
    if peak <= 0:
        return 0.0
    active = power >= peak * 10.0 ** (rel_db / 10.0)
    return float(np.sqrt(power[active].mean()))


def _exp_kernel(tau_ms: float, hop_ms: float) -> np.ndarray:
    """Causal one-pole smoother as a truncated, normalized FIR on the frame grid."""
    tau = max(tau_ms, 1e-3) / hop_ms
    length = int(min(max(np.ceil(4.0 * tau), 1), 512))
    kernel = np.exp(-np.arange(length) / tau)
    return kernel / kernel.sum()


def agc(x: np.ndarray, sr: int, tau_ms: float, strength: float,
        hop_ms: float = AGC_HOP_MS) -> np.ndarray:
    """Receiver AGC: divide by the smoothed envelope, `strength` of the way.

    `strength` 0 is a bypass, 1 flattens the envelope completely.  Envelope and
    gain live on a `hop_ms` frame grid and the gain is interpolated back up,
    which is far finer than any plausible time constant.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if strength <= 0 or len(x) == 0:
        return x.astype(np.float32)
    hop = max(1, int(sr * hop_ms / 1000))
    frames = len(x) // hop
    if frames < 4:
        return x.astype(np.float32)
    power = np.mean(x[:frames * hop].reshape(frames, hop) ** 2, axis=1)
    kernel = _exp_kernel(tau_ms, hop_ms)
    smoothed = np.convolve(power, kernel, mode="full")[:frames]
    envelope = np.sqrt(smoothed + EPS)
    gain = (np.sqrt(power.mean() + EPS) / envelope) ** float(strength)
    centres = np.arange(frames) * hop + hop / 2.0
    return (x * np.interp(np.arange(len(x)), centres, gain)).astype(np.float32)


def nonlinearity(x: np.ndarray, drive: float, poly: Iterable[float]) -> np.ndarray:
    """Transmitter clipping: tanh drive plus a low-order polynomial trim.

    The polynomial's even term is mean-removed so it adds asymmetric harmonics
    without a DC offset.
    """
    d = max(float(drive), 1e-3)
    u = np.tanh(d * np.asarray(x, dtype=np.float64)) / d
    c2, c3 = (list(poly) + [0.0, 0.0])[:2]
    if c2 or c3:
        square = u * u
        u = u + c2 * (square - square.mean()) + c3 * u * square
    return u


def apply_preset(x: np.ndarray, sr: int, preset: Preset,
                 noise: np.ndarray | None = None, snr_db: float | None = None,
                 filter_noise: bool = True, taps: np.ndarray | None = None
                 ) -> np.ndarray:
    """Push `x` through one fitted channel. Returns audio at the input's level.

    `noise` (any length ≥ len(x), any level) replaces the fitted chain's
    synthetic floor with a real harvested bed; `snr_db` overrides the fitted
    SNR.  `filter_noise` decides whether the noise goes through the EQ: true for
    synthetic noise (the fit's own convention — broadband noise entering the
    receiver's audio filter), false for a harvested bed, which already carries a
    real receiver's band shape and would otherwise be filtered twice.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    reference = speech_rms(x, sr)
    if reference <= 0 or len(x) == 0:
        return x.astype(np.float32)

    y = nonlinearity(x / reference, preset.drive, preset.poly)
    y = agc(y, sr, preset.agc_tau_ms, preset.agc_strength).astype(np.float64)
    if taps is None:
        taps = fir_taps(preset.band_gains_db, preset.band_edges_hz, sr)
    y = signal.fftconvolve(y, taps, mode="same")

    level = speech_rms(y, sr)
    if level > 0:
        y = y / level
    gain = (float(preset.noise_gain) if snr_db is None
            else 10.0 ** (-float(snr_db) / 20.0))
    if gain > 0:
        bed = _noise_at(noise, len(y), sr)
        if filter_noise:
            bed = signal.fftconvolve(bed, taps, mode="same")
            bed = bed / (np.sqrt(np.mean(bed ** 2)) + EPS)
        y = y + gain * bed
    return (y * reference).astype(np.float32)


def _noise_at(noise: np.ndarray | None, n: int, sr: int) -> np.ndarray:
    """`n` samples of unit-RMS noise: the supplied bed, or white as a fallback.

    White because that is what `channel_fit` fits against — broadband noise
    entering the receiver, coloured by the fitted EQ on its way out.  Generation
    passes a harvested bed instead and turns `filter_noise` off.
    """
    if noise is None:
        bed = np.random.default_rng(0).standard_normal(n)
    else:
        bed = np.asarray(noise, dtype=np.float64).reshape(-1)
        if len(bed) < n:
            bed = np.pad(bed, (0, n - len(bed)), mode="wrap")
        bed = bed[:n]
    return bed / (np.sqrt(np.mean(bed ** 2)) + EPS)
