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
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 16000

# Measured mapping from libsndfile's MP3 `compression_level` to the CBR bitrate
# LAME actually emits at 16 kHz mono (see `codec_roundtrip`).
MP3_CBR_COMPRESSION = {
    8: 0.98, 16: 0.94, 24: 0.89, 32: 0.835, 40: 0.78, 48: 0.73, 56: 0.68,
    64: 0.615, 80: 0.52, 96: 0.415, 112: 0.31, 128: 0.205, 144: 0.10, 160: 0.02,
}


def _np_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.getrandbits(32))


def _speech_span(n: int, pad: int) -> tuple[int, int]:
    """The carrier/speech extent of a chain-padded clip, as sample indices."""
    return (pad, n - pad) if 0 < pad < n // 2 else (0, n)


def _active_span(x: np.ndarray, pad: int, rel_db: float = -40.0) -> tuple[int, int]:
    """`_speech_span` narrowed to where there is actually signal.

    TTS output carries its own leading and trailing silence inside the chain's
    padding, so the pad boundaries are not where the words start.
    """
    lo, hi = _speech_span(len(x), pad)
    core = np.abs(x[lo:hi])
    if core.size == 0 or core.max() <= 0:
        return lo, hi
    active = np.flatnonzero(core > core.max() * 10.0 ** (rel_db / 20.0))
    return (lo + int(active[0]), lo + int(active[-1]) + 1) if active.size else (lo, hi)


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
        self.sr = sr
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

    def sample(self, n: int, rng: random.Random, fade_ms: float = 5.0) -> np.ndarray:
        """`n` samples of bed, stitched from as many random clips as it takes.

        Harvested beds are short — the calibration bank's median is ~0.3 s —
        so filling a several-second utterance from a single one would repeat it
        bit for bit a dozen times and stamp a periodic pattern on the clip.
        Successive clips are drawn independently and crossfaded over `fade_ms`,
        which keeps the bed aperiodic and the joins inaudible.
        """
        fade = max(1, int(self.sr * fade_ms / 1000))
        bed = np.zeros(0, dtype=np.float32)
        while len(bed) < n:
            clip = rng.choice(self.clips)
            if len(bed) >= fade and len(clip) > fade:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                join = bed[-fade:] * (1.0 - ramp) + clip[:fade] * ramp
                bed = np.concatenate([bed[:-fade], join, clip[fade:]])
            else:
                bed = np.concatenate([bed, clip])
        start = rng.randrange(0, len(bed) - n + 1) if len(bed) > n else 0
        return bed[start:start + n]


def narrowband_roundtrip(x: np.ndarray, sr: int, rng: random.Random,
                         narrow_sr: int = 8000) -> np.ndarray:
    """Decimate to a narrower rate and back (transmitter band-limiting + aliasing)."""
    narrow_sr = int(narrow_sr)
    if narrow_sr >= sr:
        return x.astype(np.float32)
    return resample(resample(x, sr, narrow_sr), narrow_sr, sr)


def resample_chain(x: np.ndarray, sr: int, rng: random.Random,
                   narrow_sr: int = 5000, alias: bool = True) -> np.ndarray:
    """Round-trip through a very narrow rate *without* an anti-alias filter.

    `narrowband_roundtrip` decimates cleanly; this one drops samples outright,
    so content above `narrow_sr / 2` folds back into the band instead of
    disappearing.  That folded energy is the "aliasing character" of cheap
    resampling stages in a stream delivery path (03 §2, survey takeaway 2).
    `alias=False` falls back to the clean round-trip.
    """
    narrow_sr = int(narrow_sr)
    n = len(x)
    if n == 0 or narrow_sr <= 0 or narrow_sr >= sr:
        return x.astype(np.float32)
    if not alias:
        return narrowband_roundtrip(x, sr, rng, narrow_sr)
    m = max(1, int(round(n * narrow_sr / sr)))
    idx = np.minimum((np.arange(m) * sr / narrow_sr).astype(np.int64), n - 1)
    y = resample(np.asarray(x, dtype=np.float32)[idx], narrow_sr, sr)
    if len(y) > n:
        y = y[:n]
    elif len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def bandpass(x: np.ndarray, sr: int, rng: random.Random,
             low: float = 300.0, high: float = 3400.0) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def lowpass(x: np.ndarray, sr: int, rng: random.Random,
            cutoff_hz: float = 3800.0, order: int = 8,
            zero_phase: bool = True) -> np.ndarray:
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    fn = signal.sosfiltfilt if zero_phase else signal.sosfilt
    return fn(sos, x).astype(np.float32)


def _peaking_sos(sr: int, f0: float, gain_db: float, q: float) -> list[float]:
    """RBJ peaking-EQ biquad as one `sosfilt` row."""
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * min(max(f0, 20.0), 0.45 * sr) / sr
    alpha = np.sin(w0) / (2 * max(q, 0.1))
    cos_w0 = np.cos(w0)
    b = [1 + alpha * a_gain, -2 * cos_w0, 1 - alpha * a_gain]
    a = [1 + alpha / a_gain, -2 * cos_w0, 1 - alpha / a_gain]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def _shelf_sos(sr: int, f0: float, gain_db: float, high: bool) -> list[float]:
    """RBJ low/high shelving biquad as one `sosfilt` row."""
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * min(max(f0, 20.0), 0.45 * sr) / sr
    cos_w0, alpha = np.cos(w0), np.sin(w0) / 2.0
    root = 2 * np.sqrt(a_gain) * alpha
    sign = 1.0 if high else -1.0
    b = [a_gain * ((a_gain + 1) + sign * (a_gain - 1) * cos_w0 + root),
         -2 * sign * a_gain * ((a_gain - 1) + sign * (a_gain + 1) * cos_w0),
         a_gain * ((a_gain + 1) + sign * (a_gain - 1) * cos_w0 - root)]
    a = [(a_gain + 1) - sign * (a_gain - 1) * cos_w0 + root,
         2 * sign * ((a_gain - 1) - sign * (a_gain + 1) * cos_w0),
         (a_gain + 1) - sign * (a_gain - 1) * cos_w0 - root]
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def mic_coloration(x: np.ndarray, sr: int, rng: random.Random, tilt_db: float = 0.0,
                   peaks: int = 1, peak_gain_db: float = 6.0, peak_f_low: float = 300.0,
                   peak_f_high: float = 3000.0, q_low: float = 0.7, q_high: float = 2.0,
                   pivot_hz: float = 1000.0) -> np.ndarray:
    """Handset/boom-mic timbre: a broad spectral tilt plus a few resonances.

    `tilt_db` is the total high-minus-low level change across `pivot_hz`
    (positive = brighter), realized as complementary half-gain shelves.  Each
    of `peaks` peaking filters draws its own centre frequency, Q and gain in
    +-`peak_gain_db`.  Low order on purpose: real mic colour is a couple of
    broad features, not a measured impulse response.
    """
    sections = []
    if abs(tilt_db) > 1e-6:
        sections.append(_shelf_sos(sr, pivot_hz, tilt_db / 2.0, high=True))
        sections.append(_shelf_sos(sr, pivot_hz, -tilt_db / 2.0, high=False))
    for _ in range(max(0, int(peaks))):
        sections.append(_peaking_sos(sr, rng.uniform(peak_f_low, peak_f_high),
                                     rng.uniform(-peak_gain_db, peak_gain_db),
                                     rng.uniform(q_low, q_high)))
    if not sections:
        return x.astype(np.float32)
    return signal.sosfilt(np.array(sections), x).astype(np.float32)


def ptt_truncation(x: np.ndarray, sr: int, rng: random.Random, head_ms: float = 0.0,
                   tail_ms: float = 0.0, ramp_ms: float = 3.0, pad: int = 0) -> np.ndarray:
    """PTT pressed late / released early: the first/last phonemes never go out.

    The array keeps its length — only the *speech extent* shrinks, with the
    lost span silenced rather than removed.  That is what the receiver
    actually gets (no carrier yet, so no audio), it leaves the chain's padding
    structure intact for the primitives downstream, and it matches real ATC
    labels, whose transcripts still contain the clipped words.

    The cut is anchored on the first and last *audible* samples, not on the
    padding: TTS renders its own leading silence, so cutting from the pad
    boundary would trim that silence and leave every phoneme intact.
    """
    n = len(x)
    y = x.astype(np.float32).copy()
    lo, hi = _active_span(x, pad)
    span = hi - lo
    if span <= 0:
        return y
    ramp_n = max(1, int(sr * ramp_ms / 1000))
    for milliseconds, head in ((head_ms, True), (tail_ms, False)):
        cut = min(int(sr * max(milliseconds, 0.0) / 1000), span // 3)
        if cut <= 0:
            continue
        ramp = np.linspace(0.0, 1.0, min(ramp_n, span - cut), dtype=np.float32)
        if head:
            y[lo:lo + cut] = 0.0
            y[lo + cut:lo + cut + len(ramp)] *= ramp
        else:
            y[hi - cut:hi] = 0.0
            y[hi - cut - len(ramp):hi - cut] *= ramp[::-1]
    return y


def fading(x: np.ndarray, sr: int, rng: random.Random, rate_hz: float = 1.0,
           depth_db: float = 4.0) -> np.ndarray:
    """Slow multiplicative envelope (mobile / edge-of-range signal strength).

    `depth_db` is the peak-to-trough swing; distinct from `dropouts`, which is
    sudden and total rather than gradual and partial.
    """
    if depth_db <= 0 or rate_hz <= 0:
        return x.astype(np.float32)
    t = np.arange(len(x)) / sr
    gain_db = 0.5 * depth_db * np.sin(2 * np.pi * rate_hz * t + rng.uniform(0, 2 * np.pi))
    return (x * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def agc_attack(x: np.ndarray, sr: int, rng: random.Random, attack_ms: float = 100.0,
               surge_db: float = 6.0, pad: int = 0) -> np.ndarray:
    """Receiver AGC settling: the transmission's first moments come in hot.

    Gain starts `surge_db` high at squelch open and relaxes to unity with time
    constant `attack_ms`.  Squelch open is the *carrier* onset — the pad
    boundary — not the first phoneme: the AGC starts settling the moment the
    talker keys the mic, so the surge usually lands on the noise bed and the
    first word or two, which is exactly the audible artifact.
    """
    n = len(x)
    if n == 0 or surge_db <= 0 or attack_ms <= 0:
        return x.astype(np.float32)
    start, _ = _speech_span(n, pad)
    gain = np.ones(n, dtype=np.float64)
    t = np.arange(n - start) / sr
    gain[start:] += (10.0 ** (surge_db / 20.0) - 1.0) * np.exp(-t / (attack_ms / 1000.0))
    return (x * gain).astype(np.float32)


def squelch_gate(x: np.ndarray, sr: int, rng: random.Random, floor_db: float = -40.0,
                 attack_ms: float = 10.0, release_ms: float = 30.0,
                 threshold_db: float = -20.0, hold_ms: float = 60.0,
                 tail_burst_prob: float = 0.3, tail_burst_min_ms: float = 20.0,
                 tail_burst_max_ms: float = 80.0, tail_burst_amp: float = 0.3,
                 pad: int = 0) -> np.ndarray:
    """Carrier-presence gate: the noise bed only exists while the radio is keyed.

    The measured calibration clips (01 §2) have near-digital-silence floors
    outside the transmission, which a continuous noise bed cannot produce --
    the single most audible mismatch in the old chain.  Everything outside the
    speech extent, plus any intra-speech stretch whose smoothed envelope sits
    more than `threshold_db` below the loudest part of the transmission, is
    attenuated to `floor_db`, with `attack_ms`/`release_ms` ramps and a
    `hold_ms` hangover so the gate does not chatter between syllables.

    `threshold_db` is relative to that envelope peak rather than absolute, so
    the intra-speech half of the gate self-disables where it should: at a low
    chain SNR the noise between words sits above the threshold and the gate
    stays open for the whole transmission, which is what a carrier-operated
    squelch does.  Only the pad gating is unconditional.

    With probability `tail_burst_prob` a decaying noise burst is added at the
    final gate close -- the squelch tail you hear when the carrier drops
    before the gate catches up.  Its amplitude is `tail_burst_amp` relative to
    the transmission's envelope peak.

    Gate decisions are made on a 1 ms frame grid and interpolated back to
    sample rate; that is far finer than the ramps, and keeps the asymmetric
    attack/release smoother cheap.
    """
    n = len(x)
    y = x.astype(np.float32).copy()
    hop = max(1, int(sr * 0.001))
    frame_ms = hop * 1000 / sr
    frames = n // hop
    if frames < 4:
        return y

    env = np.sqrt(np.mean(y[:frames * hop].astype(np.float64).reshape(frames, hop) ** 2,
                          axis=1))
    smooth = max(1, int(20.0 / frame_ms))                 # ~20 ms envelope smoothing
    env = np.convolve(env, np.ones(smooth) / smooth, mode="same")

    lo, hi = _speech_span(n, pad)
    inside = np.zeros(frames, dtype=bool)
    inside[lo // hop:max(lo // hop + 1, hi // hop)] = True
    reference = float(env[inside].max()) if inside.any() else float(env.max())
    open_gate = inside & (env > reference * 10.0 ** (threshold_db / 20.0))
    hold = max(1, int(hold_ms / frame_ms))
    open_gate = np.convolve(open_gate.astype(np.float64), np.ones(hold),
                            mode="full")[:frames] > 0

    floor = 10.0 ** (floor_db / 20.0)
    target = np.where(open_gate, 1.0, floor)
    rise = np.exp(-frame_ms / max(attack_ms, 1e-3))
    fall = np.exp(-frame_ms / max(release_ms, 1e-3))
    gains = np.empty(frames)
    level = float(target[0])
    for index, goal in enumerate(target):
        level = goal + (level - goal) * (rise if goal > level else fall)
        gains[index] = level
    y *= np.interp(np.arange(n), np.arange(frames) * hop + hop / 2.0,
                   gains).astype(np.float32)

    closes = np.flatnonzero(open_gate)
    if len(closes) and rng.random() < tail_burst_prob and tail_burst_amp > 0:
        start = int(closes[-1]) * hop
        length = min(int(sr * rng.uniform(tail_burst_min_ms, tail_burst_max_ms) / 1000),
                     n - start)
        if length > 0:
            burst = _np_rng(rng).standard_normal(length) * (tail_burst_amp * reference)
            y[start:start + length] += (burst * np.exp(-np.linspace(0, 4, length))
                                        ).astype(np.float32)
    return y


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


def _align(y: np.ndarray, n: int) -> np.ndarray:
    """Trim the encoder's leading delay padding / pad a short decode."""
    if len(y) > n:
        return y[len(y) - n:]
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y


def _aac_roundtrip(x: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray | None:
    """AAC via an ffmpeg subprocess; None when ffmpeg is unavailable or fails."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    with tempfile.TemporaryDirectory() as work:
        raw, encoded, decoded = (Path(work) / name
                                 for name in ("in.wav", "mid.m4a", "out.wav"))
        sf.write(raw, x, sr)
        try:
            for args in ((raw, encoded, ["-c:a", "aac", "-b:a", f"{bitrate_kbps}k"]),
                         (encoded, decoded, [])):
                subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                                "-i", str(args[0]), *args[2], str(args[1])],
                               check=True, capture_output=True)
            y, _ = sf.read(decoded, dtype="float32")
        except (subprocess.CalledProcessError, OSError, RuntimeError):
            return None
    return y.astype(np.float32)


def codec_roundtrip(x: np.ndarray, sr: int, rng: random.Random,
                    compression_level: float = 0.9, bitrate_kbps: int | None = None,
                    codec: str = "mp3") -> np.ndarray:
    """Encode/decode through a low-bitrate stream codec (LiveATC-style delivery).

    `bitrate_kbps`, when given, selects a real CBR bitrate; otherwise the
    libsndfile-native `compression_level` (0 = best quality) is used directly,
    which is what the pre-P1 default profile passes.

    libsndfile has no bitrate setter of its own: in `bitrate_mode="CONSTANT"`
    it maps `compression_level` linearly onto LAME's bitrate table, so the
    table has to be measured rather than computed.  `MP3_CBR_COMPRESSION`
    holds that measurement for 16 kHz mono under libsndfile 1.2.2 (encoded
    size scales as expected: 0.94 -> 16 kbps, 0.89 -> 24, 0.835 -> 32,
    0.615 -> 64).  Requested bitrates snap to the nearest entry, so 03 §3's
    23 kbps tier resolves to LAME's neighbouring 24 kbps step -- 23 is not in
    the MPEG-2 Layer III table at all.

    `codec="aac"` shells out to ffmpeg instead and is a no-op when ffmpeg is
    not installed, so it stays opt-in and never breaks a generation run.
    """
    peak = np.abs(x).max()
    if peak > 1.0:
        x = x / peak * 0.98
    n = len(x)

    if codec == "aac":
        y = _aac_roundtrip(x, sr, int(bitrate_kbps or 32))
        return x.astype(np.float32) if y is None else _align(y, n).astype(np.float32)
    if bitrate_kbps is not None:
        compression_level = MP3_CBR_COMPRESSION[
            min(MP3_CBR_COMPRESSION, key=lambda kbps: abs(kbps - bitrate_kbps))]

    buf = io.BytesIO()
    sf.write(buf, x, sr, format="MP3", bitrate_mode="CONSTANT",
             compression_level=compression_level)
    buf.seek(0)
    y, _ = sf.read(buf, dtype="float32")
    return _align(y, n).astype(np.float32)


PRIMITIVES = {
    "mic_coloration": mic_coloration,
    "ptt_truncation": ptt_truncation,
    "narrowband_roundtrip": narrowband_roundtrip,
    "resample_chain": resample_chain,
    "bandpass": bandpass,
    "lowpass": lowpass,
    "agc_wander": agc_wander,
    "agc_attack": agc_attack,
    "am_distortion": am_distortion,
    "soft_clip": soft_clip,
    "dropouts": dropouts,
    "fading": fading,
    "additive_noise": additive_noise,
    "hum": hum,
    "crackle": crackle,
    "heterodyne": heterodyne,
    "squelch_gate": squelch_gate,
    "squelch_clicks": squelch_clicks,
    "cochannel_mix": cochannel_mix,
    "codec_roundtrip": codec_roundtrip,
}
