"""Long-term average spectrum of a clip directory, dB relative to curve peak.

1024-pt Hann Welch PSD per clip, power-averaged across clips, then converted to
dB and offset so the curve's maximum is 0 dB.  Sampled at the measurement
frequencies the real KIXD reference was reported on.

    uv run python ltas_check.py <dir> [<dir> ...] [--label name ...] [--json out]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

POINTS = [100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0, 3000.0, 3400.0, 4000.0]
REAL_REF = [-21.8, -10.4, -5.7, 0.0, -8.8, -12.1, -24.0, -35.4, -55.4]
NFFT = 1024
SR = 16000


def clip_psd(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    x, sr = sf.read(str(path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < NFFT:
        return None
    if not np.isfinite(x).all() or float(np.max(np.abs(x))) <= 0.0:
        return None
    nperseg = min(NFFT, x.size)
    f, pxx = signal.welch(x, fs=sr, window="hann", nperseg=nperseg,
                          noverlap=nperseg // 2, scaling="density")
    return f, pxx


def dir_ltas(d: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Power-average the per-clip PSDs.

    Each clip is normalized by its own total power first, so a handful of loud
    clips cannot dominate the shape -- the reference curve is a shape, not a
    level.
    """
    wavs = sorted(d.rglob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    acc = None
    freqs = None
    used = 0
    for w in wavs:
        got = clip_psd(w)
        if got is None:
            continue
        f, pxx = got
        if freqs is None:
            freqs = f
        elif len(f) != len(freqs):
            continue
        tot = pxx.sum()
        if tot <= 0:
            continue
        acc = pxx / tot if acc is None else acc + pxx / tot
        used += 1
    if used == 0:
        raise SystemExit(f"no usable wavs in {d}")
    return freqs, acc / used, used


def sample_db(freqs: np.ndarray, pxx: np.ndarray) -> tuple[list[float], float]:
    db = 10.0 * np.log10(pxx + 1e-30)
    db = db - db.max()
    peak_hz = float(freqs[int(np.argmax(pxx))])
    vals = [float(np.interp(p, freqs, db)) for p in POINTS]
    return vals, peak_hz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    labels = a.label + [Path(d).name for d in a.dirs[len(a.label):]]
    out = {"points_hz": POINTS, "real_reference_db": REAL_REF, "curves": {}}

    hdr = "     Hz " + "".join(f"{p:>8.0f}" for p in POINTS)
    print(hdr)
    print(f"{'real ref':>8} " + "".join(f"{v:>8.1f}" for v in REAL_REF))
    print("-" * len(hdr))

    for label, d in zip(labels, a.dirs):
        freqs, pxx, n = dir_ltas(Path(d), a.limit)
        vals, peak_hz = sample_db(freqs, pxx)
        gaps = [v - r for v, r in zip(vals, REAL_REF)]
        mid = [g for p, g in zip(POINTS, gaps) if 1000.0 <= p <= 3000.0]
        out["curves"][label] = {
            "dir": str(d), "clips": n, "peak_hz": peak_hz,
            "db": [round(v, 2) for v in vals],
            "gap_db": [round(g, 2) for g in gaps],
            "max_abs_gap_1k_3k": round(max(abs(g) for g in mid), 2),
            "mean_abs_gap_all": round(float(np.mean(np.abs(gaps))), 2),
        }
        print(f"{label:>8} " + "".join(f"{v:>8.1f}" for v in vals)
              + f"   (n={n}, peak {peak_hz:.0f}Hz)")
        print(f"{'  gap':>8} " + "".join(f"{g:>+8.1f}" for g in gaps)
              + f"   max|gap| 1-3k = {out['curves'][label]['max_abs_gap_1k_3k']:.1f} dB")
        print()

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
