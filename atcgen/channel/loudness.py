"""EBU R128 integrated-loudness normalization (research-findings §4.3).

The post stage has always targeted a plain RMS level, which is fine for a
single voice but drifts across voices and channels: RMS counts the sub-300 Hz
rumble and the noise floor that the passband and the gate leave behind, so two
clips at the same dBFS RMS can sit a couple of dB apart in perceived level.
R128 measures K-weighted loudness over gated 400 ms blocks instead — the
standard every broadcast and most ASR preprocessing pipelines normalize to —
which is what `normalize_lufs` targets here.

Two edge cases matter for generated ATC clips and both fall back to RMS rather
than failing:

*   **Short clips.**  A clip below one 400 ms block has no integrated reading
    at all.  Plain RMS in dBFS is within a dB or two of LUFS for band-limited
    speech, so the fallback keeps short clips on the same scale as the rest of
    the corpus instead of leaving them unnormalized.
*   **Silence.**  The noise-only arm and a hard-gated clip can measure -inf
    LUFS (R128's absolute gate drops everything below -70 LUFS).  There is no
    gain that makes silence hit a target, so the clip passes through untouched.

This module is the primitive only; `output.loudness_mode: lufs` is what wires
it into generation.
"""

from functools import lru_cache

import numpy as np

TARGET_LUFS = -23.0           # EBU R128's broadcast reference
BLOCK_SEC = 0.400             # R128 integration block: no reading below this
PEAK_CEILING = 0.99


@lru_cache(maxsize=8)
def _meter(sr: int):
    """A `pyloudnorm` meter for `sr` (it designs K-weighting filters per rate)."""
    import pyloudnorm

    return pyloudnorm.Meter(sr)


def integrated_lufs(wav: np.ndarray, sr: int) -> float:
    """R128 integrated loudness, or -inf when the clip is too short or silent."""
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size < int(sr * BLOCK_SEC) or not np.any(x):
        return float("-inf")
    value = float(_meter(sr).integrated_loudness(x))
    return value if np.isfinite(value) else float("-inf")


def normalize_lufs(wav: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS,
                   peak_ceiling: float = PEAK_CEILING) -> np.ndarray:
    """Scale `wav` to `target_lufs`, then hold the peak under `peak_ceiling`.

    Falls back to RMS normalization when the clip is shorter than one R128
    block or measures no loudness at all, returns digital silence untouched,
    and never returns a non-finite sample.
    """
    x = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1),
                      nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return x

    loudness = integrated_lufs(x, sr)
    if not np.isfinite(loudness):
        loudness = 20.0 * np.log10(rms)          # RMS fallback, see the module docstring
    y = x * np.float32(10.0 ** ((float(target_lufs) - loudness) / 20.0))

    peak = float(np.abs(y).max())
    if peak > peak_ceiling > 0:
        y = y * np.float32(peak_ceiling / peak)
    return y.astype(np.float32) if np.isfinite(y).all() else x
