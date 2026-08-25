"""Per-clip differentiable channel fitting (04 §2.1), statistics-matching variant.

MicAugment identifies a recording channel by pushing a probe through a
differentiable chain and descending on the distance to a target recording.  We
have no clean counterpart to our real clips, so the fit is weakly supervised:
the target is the real clip's own *statistics* — long-term average spectrum,
per-bin spectral variability, frame-energy dynamic range and modulation
spectrum — and the probe is clean speech that has nothing to do with it.  What
survives that objective is the channel, because it is the only thing the probe
and the target have in common.

The chain is `preset.apply_preset`'s, mirrored in torch:

    speech -> tanh drive + polynomial -> AGC -> EQ band gains -> + noise

Only the EQ has many parameters (32 band gains on the Tier 1 LTAS grid), and it
alone can nearly satisfy the LTAS term; the nonlinearity, AGC and noise level
are pinned by the other three terms, which the EQ cannot reach — a linear filter
cannot change the *shape* of the frame-energy distribution or the modulation
spectrum, only the noise floor and the envelope processing can.

Fits are independent per clip and float32 CPU throughout; 99 clips at 300 steps
is minutes on a laptop.  On the 5080 the same command runs with
`--device cuda`, which is worth it only for the full ~1k corpus.

    uv run python -m atcgen.channel.learned.channel_fit \\
        runs/calib_v1/corpus.jsonl runs/calib_v2/presets.jsonl \\
        --probe-dir runs/p2_smoke/s0_tts_matched

`--probe-dir` should hold clean 16 kHz TTS renders — the same voice pool
generation will use, so the fitted EQ maps *that* spectrum onto the real one.
Without it the probe falls back to synthetic speech-shaped noise, which is
enough for tests but leaves a systematic tilt in real fits.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ...eval.channel_stats import clip_stats
from .preset import (BAND_EDGES, EPS, N_TAPS, TARGET_SR, Preset, apply_preset,
                     band_centers, passband_edges, speech_rms, write_presets)

STFT_SCALES = (256, 512, 2048)
LTAS_NFFT = 2048                     # 7.8 Hz bins: every LTAS band gets one
STATS_NFFT = 512                     # frames / modulation
STATS_HOP = 128                      # -> a 125 Hz frame grid
FRAME_QUANTILES = (0.05, 0.15, 0.25, 0.5, 0.75, 0.9, 0.95)
FRAME_FLOOR_DB = -60.0               # below the median: squelched silence, not channel
MOD_EDGES = np.geomspace(0.5, 20.0, 9)
WEIGHTS = {"ltas": 1.0, "bins": 0.5, "frames": 1.0, "mod": 1.0}
SMOOTH_REG = 2e-4                    # on the EQ's second difference
POLY_REG = 1e-2
EQ_LR = 0.5                          # dB per step: band gains travel tens of dB
SNR_RANGE = (0.0, 45.0)     # 45 = "no floor measurable in the pauses"
PROBE_SEC = (1.5, 5.0)               # probe length, clamped around the clip's
DEFAULT_STEPS = 300
DEFAULT_PROBES = 2
QC_MAD_K = 5.0                       # fit-loss outlier cut, in MADs above median


# --------------------------------------------------------------------------- #
# differentiable statistics
# --------------------------------------------------------------------------- #

def _band_matrix(freqs: np.ndarray, edges) -> torch.Tensor:
    """(K, F) selector summing FFT bin powers into the LTAS bands."""
    edges = np.asarray(list(edges), dtype=np.float64)
    index = np.digitize(freqs, edges) - 1
    matrix = np.zeros((len(edges) - 1, len(freqs)), dtype=np.float32)
    for band in range(matrix.shape[0]):
        matrix[band] = (index == band)
    return torch.from_numpy(matrix)


def _interp_matrix(freqs: np.ndarray, edges) -> torch.Tensor:
    """(F, K) matrix realizing `preset.response_db`'s log-frequency interpolation."""
    centres = np.log(band_centers(edges))
    f = np.log(np.maximum(np.asarray(freqs, dtype=np.float64), 1.0))
    matrix = np.zeros((len(f), len(centres)), dtype=np.float32)
    right = np.clip(np.searchsorted(centres, f), 1, len(centres) - 1)
    left = right - 1
    span = centres[right] - centres[left]
    weight = np.clip((f - centres[left]) / span, 0.0, 1.0)
    matrix[np.arange(len(f)), left] = 1.0 - weight
    matrix[np.arange(len(f)), right] += weight
    return torch.from_numpy(matrix)


class ClipStatistics:
    """The fit's target vector: everything measured about one clip, in dB."""

    def __init__(self, sr: int = TARGET_SR, edges=BAND_EDGES,
                 device: str = "cpu"):
        self.sr = sr
        freqs = np.fft.rfftfreq(LTAS_NFFT, 1.0 / sr)
        self.bands = _band_matrix(freqs, edges).to(device)
        self.windows = {n: torch.hann_window(n, device=device) for n in STFT_SCALES}
        self._mod_cache: dict[int, torch.Tensor] = {}   # keyed by envelope length
        self.device = device

    def _power(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        spec = torch.stft(x, n_fft, n_fft // 4, window=self.windows[n_fft],
                          return_complex=True, center=True, pad_mode="reflect")
        return spec.real ** 2 + spec.imag ** 2 + EPS      # (B, F, T)

    def _mod_matrix(self, n_frames: int, frame_rate: float) -> torch.Tensor:
        if n_frames not in self._mod_cache:
            freqs = np.fft.rfftfreq(n_frames, 1.0 / frame_rate)
            self._mod_cache[n_frames] = _band_matrix(freqs, MOD_EDGES).to(self.device)
        return self._mod_cache[n_frames]

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """`x` is (B, N) at unit speech RMS; every entry is batch-averaged."""
        out: dict[str, torch.Tensor] = {}

        powers = {n_fft: self._power(x, n_fft) for n_fft in STFT_SCALES}
        band = self.bands @ powers[LTAS_NFFT].mean(dim=2).T         # (K, B)
        out["ltas"] = (10.0 * torch.log10(band / band.sum(0, keepdim=True)
                                          + EPS)).mean(1)

        for n_fft, power_spectrum in powers.items():
            log_power = 10.0 * torch.log10(power_spectrum)
            mean = log_power.mean(dim=2)
            out[f"bins_mean_{n_fft}"] = (mean - mean.mean(1, keepdim=True)).mean(0)
            out[f"bins_std_{n_fft}"] = log_power.std(dim=2).mean(0)

        power = powers[STATS_NFFT]
        frame_db = 10.0 * torch.log10(power.mean(dim=1))            # (B, T)
        quantiles = torch.quantile(
            frame_db, torch.tensor(FRAME_QUANTILES, device=x.device), dim=1)
        # centred on the median and floored: how far a squelch-gated clip's
        # silence falls is the post-effects' business, not the fitted chain's
        out["frames"] = (quantiles - quantiles[len(FRAME_QUANTILES) // 2]
                         ).clamp(min=FRAME_FLOOR_DB).mean(1)

        envelope = torch.sqrt(power.mean(dim=1))
        envelope = envelope - envelope.mean(dim=1, keepdim=True)
        window = torch.hann_window(envelope.shape[1], device=x.device)
        spectrum = torch.fft.rfft(envelope * window).abs() ** 2 + EPS
        matrix = self._mod_matrix(envelope.shape[1], self.sr / STATS_HOP)
        mod = matrix @ spectrum.T                                   # (M, B)
        out["mod"] = (10.0 * torch.log10(mod / mod.sum(0, keepdim=True) + EPS)).mean(1)
        return out


def distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
    """Weighted L1 between two statistic bundles, all terms already in dB."""
    total = torch.zeros((), dtype=torch.float32, device=a["ltas"].device)
    for key, value in a.items():
        group = key.split("_")[0]
        weight = WEIGHTS[group] / (len(STFT_SCALES) * 2 if group == "bins" else 1)
        total = total + weight * (value - b[key]).abs().mean()
    return total


# --------------------------------------------------------------------------- #
# the fitted chain, in torch
# --------------------------------------------------------------------------- #

def _speech_rms(x: torch.Tensor, sr: int) -> torch.Tensor:
    """`preset.speech_rms` per batch row; the frame selection is stop-gradient."""
    frame = max(1, int(sr * 0.020))
    n = x.shape[-1] // frame
    if n < 2:
        return torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + EPS)
    power = (x[..., :n * frame] ** 2).reshape(*x.shape[:-1], n, frame).mean(-1)
    active = (power >= power.max(dim=-1, keepdim=True).values * 1e-4).detach().float()
    return torch.sqrt((power * active).sum(-1, keepdim=True)
                      / active.sum(-1, keepdim=True) + EPS)


class FittedChannel(torch.nn.Module):
    """`preset.apply_preset` as a differentiable module over its own parameters."""

    def __init__(self, n_samples: int, sr: int = TARGET_SR, edges=BAND_EDGES,
                 n_taps: int = N_TAPS, device: str = "cpu"):
        super().__init__()
        self.sr, self.edges, self.n_taps = sr, list(edges), n_taps
        self.n_fft_taps = 1 << int(np.ceil(np.log2(max(n_taps * 4, 1024))))
        self.n_fft = 1 << int(np.ceil(np.log2(n_samples + n_taps)))
        self.n_samples = n_samples

        self.to_taps = _interp_matrix(
            np.fft.rfftfreq(self.n_fft_taps, 1.0 / sr), self.edges).to(device)
        self.register_buffer("window", torch.from_numpy(
            np.hanning(n_taps).astype(np.float32)))

        centres = band_centers(self.edges)
        # init: Mode 1's default band, flat in 300-2600 Hz and rolling off outside
        init = np.where((centres >= 300) & (centres <= 2600), 0.0,
                        -12.0 * np.abs(np.log2(np.clip(centres, 1, None) / 1200.0)))
        self.band_gains = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.drive_raw = torch.nn.Parameter(torch.tensor(0.5))
        self.poly_raw = torch.nn.Parameter(torch.zeros(2))
        self.tau_raw = torch.nn.Parameter(torch.zeros(()))
        self.strength_raw = torch.nn.Parameter(torch.tensor(-1.0))
        self.snr_raw = torch.nn.Parameter(torch.zeros(()))

    # -- parameter views ---------------------------------------------------- #
    @property
    def drive(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.drive_raw) + 0.05

    @property
    def poly(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.poly_raw)

    @property
    def tau_ms(self) -> torch.Tensor:
        return 5.0 + 195.0 * torch.sigmoid(self.tau_raw)

    @property
    def strength(self) -> torch.Tensor:
        return torch.sigmoid(self.strength_raw)

    @property
    def snr_db(self) -> torch.Tensor:
        return SNR_RANGE[0] + (SNR_RANGE[1] - SNR_RANGE[0]) * torch.sigmoid(self.snr_raw)

    def set_snr(self, snr_db: float) -> None:
        """Place `snr_db` on the raw parameter, inverting `snr_db`'s squashing."""
        low, high = SNR_RANGE
        fraction = float(np.clip((snr_db - low) / (high - low), 1e-4, 1 - 1e-4))
        self.snr_raw.fill_(float(np.log(fraction / (1.0 - fraction))))

    # -- the chain ---------------------------------------------------------- #
    def taps(self) -> torch.Tensor:
        magnitude = 10.0 ** ((self.to_taps @ self.band_gains) / 20.0)
        impulse = torch.fft.irfft(
            torch.complex(magnitude, torch.zeros_like(magnitude)), n=self.n_fft_taps)
        half = self.n_taps // 2
        taps = torch.cat([impulse[-half:], impulse[:half + 1]])
        return taps * self.window

    def _filter(self, x: torch.Tensor, taps_f: torch.Tensor) -> torch.Tensor:
        """`scipy.signal.fftconvolve(x, taps, mode="same")`, batched."""
        y = torch.fft.irfft(torch.fft.rfft(x, n=self.n_fft) * taps_f, n=self.n_fft)
        start = (self.n_taps - 1) // 2
        return y[..., start:start + x.shape[-1]]

    def _agc(self, x: torch.Tensor) -> torch.Tensor:
        hop = max(1, int(self.sr * 5.0 / 1000))
        frames = x.shape[-1] // hop
        if frames < 4:
            return x
        power = (x[..., :frames * hop] ** 2).reshape(*x.shape[:-1], frames, hop).mean(-1)
        tau = self.tau_ms / 5.0
        length = int(min(max(float(4.0 * tau.detach()), 1), 512))
        kernel = torch.exp(-torch.arange(length, device=x.device) / tau)
        kernel = (kernel / kernel.sum()).flip(0).reshape(1, 1, -1)
        smoothed = torch.nn.functional.conv1d(
            torch.nn.functional.pad(power.unsqueeze(1), (length - 1, 0)), kernel
        ).squeeze(1)
        envelope = torch.sqrt(smoothed + EPS)
        gain = (torch.sqrt(power.mean(-1, keepdim=True) + EPS)
                / envelope) ** self.strength
        up = torch.nn.functional.interpolate(
            gain.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False)
        return x * up.squeeze(1)

    def forward(self, probe: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        taps_f = torch.fft.rfft(self.taps(), n=self.n_fft)
        drive = self.drive
        y = torch.tanh(drive * probe) / drive
        c2, c3 = self.poly[0], self.poly[1]
        square = y * y
        y = y + c2 * (square - square.mean(-1, keepdim=True)) + c3 * y * square
        y = self._filter(self._agc(y), taps_f)
        y = y / _speech_rms(y, self.sr)

        bed = self._filter(noise, taps_f)
        bed = bed / torch.sqrt((bed ** 2).mean(-1, keepdim=True) + EPS)
        return y + (10.0 ** (-self.snr_db / 20.0)) * bed

    def to_preset(self, **fields) -> Preset:
        gains = [round(float(v), 3) for v in self.band_gains.detach().cpu().numpy()]
        low, high = passband_edges(gains, self.edges)
        snr = float(self.snr_db.detach())
        return Preset(
            band_gains_db=gains,
            drive=round(float(self.drive.detach()), 4),
            poly=[round(float(v), 5) for v in self.poly.detach().cpu().numpy()],
            agc_tau_ms=round(float(self.tau_ms.detach()), 2),
            agc_strength=round(float(self.strength.detach()), 4),
            noise_gain=round(10.0 ** (-snr / 20.0), 6),
            snr_est=round(snr, 2),
            band_edges_hz=list(self.edges),
            passband_hz=[round(low, 1), round(high, 1)],
            **fields,
        )


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #

def synthetic_probe(n: int, rng: np.random.Generator, sr: int = TARGET_SR
                    ) -> np.ndarray:
    """Speech-shaped noise with a syllable-rate envelope and real pauses.

    The fallback probe.  It is not speech, but it has what the fit reads: a
    broadband spectrum with a speech-like tilt, an envelope that modulates
    around 4 Hz, and silent stretches, so the frame-energy and modulation terms
    have something to bite on.
    """
    from scipy import signal as scipy_signal

    noise = rng.standard_normal(n + sr)
    sos = scipy_signal.butter(2, [120.0, 4000.0], btype="bandpass", fs=sr, output="sos")
    x = scipy_signal.sosfilt(sos, noise)[sr:]
    hop = max(1, int(sr * 0.010))
    frames = max(4, n // hop + 1)
    syllable = np.abs(np.sin(np.pi * np.arange(frames) * rng.uniform(3.0, 5.0)
                             * hop / sr + rng.uniform(0, np.pi)))
    voiced = np.ones(frames)
    for start in range(0, frames, int(0.6 / (hop / sr))):
        if rng.random() < 0.35:
            voiced[start:start + int(0.25 / (hop / sr))] = 0.02
    envelope = np.interp(np.arange(n), np.arange(frames) * hop,
                         (0.25 + 0.75 * syllable) * voiced)
    return (x[:n] * envelope).astype(np.float32)


def active_span(x: np.ndarray, sr: int, margin_ms: float = 100.0,
                rel_db: float = -40.0) -> tuple[int, int]:
    """First and last audible sample, widened by `margin_ms`.

    Real clips are recorded with the squelch open before and after the
    transmission, and TTS renders its own lead-in silence; neither belongs in a
    statistic that is meant to describe the transmission.  How deep the silence
    around a transmission is stays a post-effect (`squelch_gate`), so the fit
    never sees it.
    """
    frame = max(1, int(sr * 0.020))
    n = len(x) // frame
    if n < 2:
        return 0, len(x)
    power = np.mean(np.asarray(x, dtype=np.float64)[:n * frame]
                    .reshape(n, frame) ** 2, axis=1)
    peak = power.max()
    if peak <= 0:
        return 0, len(x)
    active = np.flatnonzero(power >= peak * 10.0 ** (rel_db / 10.0))
    margin = int(sr * margin_ms / 1000)
    return (max(0, int(active[0]) * frame - margin),
            min(len(x), (int(active[-1]) + 1) * frame + margin))


def measure_snr(x: np.ndarray, sr: int) -> float:
    """The clip's own in-transmission SNR, from its VAD-detected pauses.

    Not fitted.  The statistics the fit descends on cannot identify a noise
    floor: a probe cropped to a few seconds of continuous speech has pauses set
    by phoneme dynamics rather than by the channel, so every noise level the
    chain can produce only *reduces* the probe's dynamic range and the fit walks
    the SNR up until the noise is inaudible.  The floor is directly measurable
    instead — the same energy VAD that harvested the noise bank (M2.1) separates
    speech frames from the gaps between them, and the gap level relative to the
    speech level is the number the chain needs.

    Measured on whatever span the caller passes; `fit_corpus` passes the
    transmission, so the silence *around* it never counts as the floor.  Clipped
    to `SNR_RANGE`: a clip whose pauses are digitally silent — squelch closing
    mid-transmission, or a codec's own noise gate — has no measurable floor, and
    the ceiling is the honest answer rather than the hundreds of dB the
    arithmetic gives.
    """
    from ...dataset.local_corpus import _frame_db, _speech_frames

    db = _frame_db(np.asarray(x, dtype=np.float32), sr)
    active = _speech_frames(np.asarray(x, dtype=np.float32), sr)
    speech_db = (10.0 * np.log10(speech_rms(x, sr) ** 2 + EPS))
    gaps = db[~active]
    floor_db = float(np.percentile(gaps if gaps.size >= 3 else db,
                                   50.0 if gaps.size >= 3 else 10.0))
    return float(np.clip(speech_db - floor_db, *SNR_RANGE))


def probe_batch(n: int, count: int, rng: np.random.Generator,
                probe_dir: str | Path | None = None,
                sr: int = TARGET_SR) -> np.ndarray:
    """(count, n) of clean probe material at unit speech RMS.

    Windows are taken from inside each probe's active span, so a crop never
    lands in a silent stretch and leaves the fit reading noise as speech.
    """
    clips: list[np.ndarray] = []
    if probe_dir is not None:
        paths = sorted(Path(probe_dir).glob("*.wav"))
        if not paths:
            raise ValueError(f"no probe wavs in {probe_dir}")
        for path in rng.choice(len(paths), size=count, replace=len(paths) < count):
            wav, file_sr = sf.read(paths[int(path)], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if file_sr != sr:
                raise ValueError(f"probe {paths[int(path)]} is {file_sr} Hz, need {sr}")
            clips.append(wav)
    else:
        clips = [synthetic_probe(n, rng, sr) for _ in range(count)]

    out = np.zeros((count, n), dtype=np.float32)
    for index, clip in enumerate(clips):
        lo, hi = active_span(clip, sr)
        clip = clip[lo:hi]
        if len(clip) < n:
            clip = np.tile(clip, int(np.ceil(n / max(len(clip), 1))))
        start = int(rng.integers(0, max(1, len(clip) - n + 1)))
        piece = clip[start:start + n]
        level = speech_rms(piece, sr)
        out[index] = piece / level if level > 0 else piece
    return out


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def fit_clip(wav: np.ndarray, sr: int, probes: np.ndarray, steps: int = DEFAULT_STEPS,
             lr: float = 0.05, seed: int = 0, device: str = "cpu",
             snr_db: float | None = None) -> tuple[FittedChannel, list[float]]:
    """Fit one clip. Returns the fitted module and the loss at every step.

    `snr_db` pins the noise floor (see `measure_snr`) and holds it out of the
    optimization; without it the noise gain is fitted like everything else,
    which is only sensible on a probe that has real pauses.
    """
    torch.manual_seed(seed)
    target_wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    level = speech_rms(target_wav, sr)
    if level <= 0:
        raise ValueError("clip is silent")

    statistics = ClipStatistics(sr, device=device)
    target = {key: value.detach() for key, value in statistics(
        torch.from_numpy(target_wav / level).reshape(1, -1).to(device)).items()}

    probe = torch.from_numpy(np.asarray(probes, dtype=np.float32)).to(device)
    generator = np.random.default_rng(seed)
    noise = torch.from_numpy(
        generator.standard_normal(probe.shape).astype(np.float32)).to(device)

    model = FittedChannel(probe.shape[-1], sr, device=device).to(device)
    # warm start: for a linear chain the EQ that maps the probe's spectrum onto
    # the target's is just their difference in dB.  Descent then only has to
    # cover what the nonlinearity, the AGC and the noise floor change about it —
    # without this the band gains would need hundreds of steps to travel the
    # 40-90 dB between a flat init and a real receiver's stopband.
    with torch.no_grad():
        probe_ltas = statistics(probe / _speech_rms(probe, sr))["ltas"]
        model.band_gains.copy_(
            (target["ltas"] - probe_ltas).clamp(-120.0, 40.0))
        if snr_db is not None:
            model.set_snr(snr_db)
    model.snr_raw.requires_grad_(snr_db is None)

    frozen = {"band_gains"} | ({"snr_raw"} if snr_db is not None else set())
    others = [p for name, p in model.named_parameters() if name not in frozen]
    optimizer = torch.optim.Adam(
        [{"params": [model.band_gains], "lr": EQ_LR}, {"params": others, "lr": lr}])
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, max(1, steps // 3), gamma=0.5)

    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(probe, noise)
        loss = distance(statistics(output / _speech_rms(output, sr)), target)
        loss = (loss + SMOOTH_REG * torch.diff(model.band_gains, n=2).pow(2).mean()
                + POLY_REG * model.poly.pow(2).mean())
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append(float(loss.detach()))
    return model, history


def verify_ltas(preset: Preset, probe: np.ndarray, target_wav: np.ndarray,
                sr: int = TARGET_SR) -> np.ndarray:
    """Per-band |synth - real| LTAS error, measured with the Tier 1 estimator.

    The fit's own loss uses its own STFT grid; this re-measures with
    `atcgen.eval.channel_stats`, so the acceptance number is the one the
    evaluation module would report and not a self-graded one.
    """
    synth = apply_preset(probe, sr, preset)
    a = np.asarray(clip_stats(synth, sr)["ltas_db"], dtype=np.float64)
    b = np.asarray(clip_stats(target_wav, sr)["ltas_db"], dtype=np.float64)
    return np.abs(a - b)


def _corpus_rows(corpus: Path) -> list[dict]:
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    for row in rows:
        path = Path(row["path"])
        row["_path"] = path if path.is_absolute() else corpus.parent / path
    return rows


def fit_corpus(corpus_manifest: str | Path, out_path: str | Path,
               probe_dir: str | Path | None = None, steps: int = DEFAULT_STEPS,
               n_probes: int = DEFAULT_PROBES, seed: int = 0, limit: int | None = None,
               device: str = "cpu", qc_k: float = QC_MAD_K,
               progress: bool = False, split: str | None = None) -> dict:
    """Fit every clip in a corpus manifest, QC the results, write presets.jsonl."""
    corpus = Path(corpus_manifest)
    all_rows = _corpus_rows(corpus)
    input_counts = Counter(
        row.get("split") if row.get("split") is not None else "null"
        for row in all_rows)
    rows = [row for row in all_rows
            if split is None or row.get("split") == split]
    rows = rows[:limit]
    if not rows:
        suffix = f" for split {split!r}" if split is not None else ""
        raise ValueError(f"no clips in {corpus}{suffix}")

    presets: list[Preset] = []
    band_errors: list[np.ndarray] = []
    started = time.time()
    for index, row in enumerate(rows):
        wav, sr = sf.read(row["_path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        duration = round(len(wav) / sr, 3)
        lo, hi = active_span(wav, sr)
        wav = wav[lo:hi]
        n = int(sr * float(np.clip(len(wav) / sr, *PROBE_SEC)))
        probes = probe_batch(n, n_probes, np.random.default_rng(seed + index),
                             probe_dir, sr)
        model, history = fit_clip(wav, sr, probes, steps=steps, seed=seed + index,
                                  device=device, snr_db=measure_snr(wav, sr))
        preset = model.to_preset(
            clip_id=row.get("clip_id", Path(row["path"]).stem),
            station=row.get("station", "unknown"),
            split=row.get("split", "train"),
            duration=duration,
            fit_loss=round(history[-1], 5))
        errors = verify_ltas(preset, probes[0], wav, sr)
        preset.ltas_l1_db = round(float(errors.mean()), 3)
        presets.append(preset)
        band_errors.append(errors)
        if progress:
            print(f"  [{index + 1}/{len(rows)}] {preset.clip_id} "
                  f"loss {history[0]:.3f}->{history[-1]:.3f} "
                  f"ltas_l1 {preset.ltas_l1_db:.2f} dB snr {preset.snr_est:.1f}",
                  flush=True)

    kept, dropped = _qc(presets, qc_k)
    write_presets(out_path, kept)
    summary = _summary(kept, dropped, np.concatenate(band_errors) if band_errors
                       else np.zeros(1))
    summary.update({"presets": str(out_path), "steps": steps, "n_probes": n_probes,
                    "probe_dir": str(probe_dir) if probe_dir else None,
                    "seconds": round(time.time() - started, 1),
                    "split_filter": split,
                    "input_counts": dict(sorted(input_counts.items()))})
    Path(out_path).with_name("presets_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def _qc(presets: list[Preset], k: float) -> tuple[list[Preset], list[Preset]]:
    """Drop presets whose fit loss is an outlier — the fit did not converge."""
    losses = np.array([p.fit_loss for p in presets], dtype=np.float64)
    finite = np.isfinite(losses)
    median = float(np.median(losses[finite])) if finite.any() else 0.0
    mad = float(np.median(np.abs(losses[finite] - median))) if finite.any() else 0.0
    cutoff = median + k * max(mad, 1e-6)
    keep = [p for p in presets if np.isfinite(p.fit_loss) and p.fit_loss <= cutoff]
    kept_ids = {id(p) for p in keep}
    return keep, [p for p in presets if id(p) not in kept_ids]


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size),
            **{f"p{q}": round(float(np.percentile(values, q)), 3)
               for q in (10, 50, 90)},
            "max": round(float(values.max()), 3),
            "mean": round(float(values.mean()), 3)}


def _summary(kept: list[Preset], dropped: list[Preset],
             band_errors: np.ndarray) -> dict:
    by_station: dict[str, list[Preset]] = {}
    for preset in kept:
        by_station.setdefault(preset.station, []).append(preset)
    return {
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_clips": [p.clip_id for p in dropped],
        "fit_loss": _percentiles(np.array([p.fit_loss for p in kept])),
        "ltas_l1_db": _percentiles(np.array([p.ltas_l1_db for p in kept])),
        "ltas_band_abs_err_db": _percentiles(band_errors),
        "snr_est": _percentiles(np.array([p.snr_est for p in kept])),
        "drive": _percentiles(np.array([p.drive for p in kept])),
        "agc_strength": _percentiles(np.array([p.agc_strength for p in kept])),
        "stations": {
            station: {
                "n": len(items),
                "passband_low_hz": _percentiles(
                    np.array([p.passband_hz[0] for p in items])),
                "passband_high_hz": _percentiles(
                    np.array([p.passband_hz[1] for p in items])),
                "snr_est": _percentiles(np.array([p.snr_est for p in items])),
                "band_gains_db_mean": [
                    round(float(v), 2) for v in
                    np.mean([p.band_gains_db for p in items], axis=0)],
            }
            for station, items in sorted(by_station.items())
        },
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="corpus.jsonl from atcgen.dataset.local_corpus")
    ap.add_argument("out", help="presets.jsonl to write")
    ap.add_argument("--probe-dir", default=None,
                    help="directory of clean 16 kHz TTS wavs to probe with")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--qc-mad-k", type=float, default=QC_MAD_K)
    ap.add_argument("--split", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    summary = fit_corpus(
        args.corpus, args.out, args.probe_dir, args.steps, args.probes,
        args.seed, args.limit, args.device, args.qc_mad_k,
        progress=not args.quiet, split=args.split)
    print(json.dumps({k: v for k, v in summary.items() if k != "stations"}, indent=2))
    return summary


if __name__ == "__main__":
    main()
