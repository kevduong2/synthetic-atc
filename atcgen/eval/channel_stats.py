"""Tier 1 channel statistics (see docs/plans/05-evaluation-plan.md §2).

Per-clip acoustic descriptors of the transmission channel, aggregated into
percentile summaries and compared between a synthetic set and the real
calibration set:

  spectral_edge_hz  frequency below which 98% of the power sits (real median
                    ~2.4 kHz), plus the matching 2% low edge
  snr_db            frame-energy SNR estimate (p90 vs p15 frame power)
  rms_db / peak_db  loudness and peak level
  mod_4hz           modulation-spectrum energy in the 2-8 Hz syllable-rate
                    region, as a fraction of 0.5-20 Hz envelope energy
  ltas_db           long-term average spectrum on a fixed log band grid

Estimator definitions are calibrated against `data/real/calibration/` so the
numbers reproduce those quoted in 01-codebase-analysis.md §2.

CLI: python -m atcgen.eval.channel_stats <dir> [--ref <dir>] [--out stats.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.stats import wasserstein_distance

NFFT = 1024
FRAME_SEC = 0.025
HOP_SEC = 0.010
EDGE_FRAC = 0.98
POWER_FLOOR = 1e-9          # -90 dB: squelch-gated clips hit digital silence
NOISE_PCTL = 15             # frame-power percentile taken as the noise floor
SPEECH_PCTL = 90            # ... and as the speech level
LTAS_EDGES = np.geomspace(100.0, 8000.0, 33)   # 32 log-spaced bands
MOD_BAND = (2.0, 8.0)       # syllable-rate region
MOD_TOTAL = (0.5, 20.0)

SCALAR_KEYS = ("duration", "rms_db", "peak_db", "spectral_edge_hz",
               "spectral_low_hz", "snr_db", "mod_4hz")
PERCENTILES = (10, 50, 90)


def _db(power: float | np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, POWER_FLOOR))


def _frame_power(x: np.ndarray, sr: int) -> np.ndarray:
    n, hop = int(sr * FRAME_SEC), int(sr * HOP_SEC)
    if len(x) < n:
        return np.array([float(np.mean(x ** 2))])
    starts = np.arange(0, len(x) - n + 1, hop)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[starts]
    return np.mean(frames.astype(np.float64) ** 2, axis=1)


def _spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-clip long-term average power spectrum."""
    nperseg = min(NFFT, len(x)) if len(x) >= 64 else len(x)
    f, pxx = signal.welch(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2,
                          scaling="spectrum")
    return f, pxx


def _edges(f: np.ndarray, pxx: np.ndarray) -> tuple[float, float]:
    """Frequencies bounding the central EDGE_FRAC of the spectral power."""
    total = pxx.sum()
    if total <= 0:
        return 0.0, 0.0
    cum = np.cumsum(pxx) / total
    lo = f[min(int(np.searchsorted(cum, 1.0 - EDGE_FRAC)), len(f) - 1)]
    hi = f[min(int(np.searchsorted(cum, EDGE_FRAC)), len(f) - 1)]
    return float(lo), float(hi)


def _ltas_db(f: np.ndarray, pxx: np.ndarray) -> list[float]:
    """Band powers on the fixed grid, in dB relative to the clip's total."""
    total = pxx.sum() + 1e-20
    idx = np.digitize(f, LTAS_EDGES) - 1
    bands = np.zeros(len(LTAS_EDGES) - 1)
    for b in range(len(bands)):
        sel = idx == b
        if sel.any():
            bands[b] = pxx[sel].sum() / total
    return [round(v, 3) for v in _db(bands)]


def _mod_4hz(x: np.ndarray, sr: int) -> float:
    """Fraction of envelope-modulation energy in the syllable-rate band."""
    env = np.sqrt(_frame_power(x, sr))
    if len(env) < 8:
        return 0.0
    env_sr = 1.0 / HOP_SEC
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), 1.0 / env_sr)
    band = spec[(freqs >= MOD_BAND[0]) & (freqs < MOD_BAND[1])].sum()
    total = spec[(freqs >= MOD_TOTAL[0]) & (freqs < MOD_TOTAL[1])].sum()
    return float(band / total) if total > 0 else 0.0


def clip_stats(wav: np.ndarray, sr: int) -> dict:
    """Tier 1 statistics for one clip (JSON-serializable)."""
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("empty clip")
    f, pxx = _spectrum(x, sr)
    lo, hi = _edges(f, pxx)
    p = _frame_power(x, sr)
    noise = float(np.percentile(p, NOISE_PCTL))
    speech = float(np.percentile(p, SPEECH_PCTL))
    return {
        "duration": round(len(x) / sr, 3),
        "rms_db": round(float(_db(np.mean(x ** 2))), 2),
        "peak_db": round(float(_db(np.max(x ** 2) if x.size else 0.0)), 2),
        "spectral_edge_hz": round(hi, 1),
        "spectral_low_hz": round(lo, 1),
        "snr_db": round(float(_db(max(speech - noise, POWER_FLOOR)) - _db(noise)), 2),
        "mod_4hz": round(_mod_4hz(x, sr), 4),
        "ltas_db": _ltas_db(f, pxx),
    }


def _iter_clips(source, sr: int | None):
    """Yield (name, wav, sr) from a directory, list of paths, or arrays."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        items = sorted(path.glob("*.wav")) if path.is_dir() else [path]
    else:
        items = list(source)
    for i, item in enumerate(items):
        if isinstance(item, (str, Path)):
            wav, file_sr = sf.read(item, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            yield Path(item).name, wav, file_sr
        else:
            if sr is None:
                raise ValueError("sr is required when passing raw arrays")
            yield f"clip_{i:06d}", np.asarray(item), sr


def compute_stats(source, sr: int | None = None) -> dict:
    """Per-clip records plus p10/p50/p90 summaries for a set of clips.

    `source` is a directory of wavs, a list of wav paths, or a list of arrays
    (then `sr` is required).
    """
    clips = []
    for name, wav, file_sr in _iter_clips(source, sr):
        if len(wav) == 0:
            continue
        rec = clip_stats(wav, file_sr)
        rec["name"] = name
        clips.append(rec)
    if not clips:
        raise ValueError(f"no clips found in {source}")

    summary = {}
    for key in SCALAR_KEYS:
        vals = np.array([c[key] for c in clips], dtype=float)
        summary[key] = {f"p{q}": round(float(np.percentile(vals, q)), 3)
                        for q in PERCENTILES}
        summary[key]["mean"] = round(float(vals.mean()), 3)
    ltas = np.array([c["ltas_db"] for c in clips], dtype=float)
    return {
        "n": len(clips),
        "clips": clips,
        "summary": summary,
        "ltas_hz": [round(float(v), 1) for v in
                    np.sqrt(LTAS_EDGES[:-1] * LTAS_EDGES[1:])],
        "ltas_db_mean": [round(float(v), 3) for v in ltas.mean(axis=0)],
    }


def compare(synthetic: dict, real: dict) -> dict:
    """Per-statistic Wasserstein distance + 'synthetic median inside real
    p10-p90?' flag (the P2 acceptance test of 05 §3)."""
    out = {"n_synthetic": synthetic["n"], "n_real": real["n"], "stats": {}}
    for key in SCALAR_KEYS:
        s = np.array([c[key] for c in synthetic["clips"]], dtype=float)
        r = np.array([c[key] for c in real["clips"]], dtype=float)
        p10, p90 = float(np.percentile(r, 10)), float(np.percentile(r, 90))
        med = float(np.median(s))
        spread = p90 - p10
        out["stats"][key] = {
            "wasserstein": round(float(wasserstein_distance(s, r)), 4),
            "wasserstein_norm": round(float(wasserstein_distance(s, r) / spread), 4)
            if spread > 0 else None,
            "synthetic_p50": round(med, 3),
            "real_p10": round(p10, 3),
            "real_p90": round(p90, 3),
            "median_in_range": bool(p10 <= med <= p90),
        }
    a = np.array(synthetic["ltas_db_mean"])
    b = np.array(real["ltas_db_mean"])
    out["ltas_l1_db"] = round(float(np.mean(np.abs(a - b))), 3)
    out["all_medians_in_range"] = all(v["median_in_range"]
                                      for v in out["stats"].values())
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Tier 1 channel statistics")
    ap.add_argument("wav_dir", help="directory of wavs to measure")
    ap.add_argument("--ref", help="reference (real) wav directory to compare against")
    ap.add_argument("--out", help="write the full JSON report here")
    args = ap.parse_args(argv)

    stats = compute_stats(args.wav_dir)
    if args.ref:
        stats["comparison"] = compare(stats, compute_stats(args.ref))

    print(f"{stats['n']} clips from {args.wav_dir}")
    for key in SCALAR_KEYS:
        s = stats["summary"][key]
        line = f"  {key:17s} p10={s['p10']:>9.2f}  p50={s['p50']:>9.2f}  p90={s['p90']:>9.2f}"
        if "comparison" in stats:
            c = stats["comparison"]["stats"][key]
            line += f"   W={c['wasserstein']:>9.3f}  in_range={c['median_in_range']}"
        print(line)
    if "comparison" in stats:
        print(f"  LTAS L1 distance: {stats['comparison']['ltas_l1_db']:.2f} dB")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2))
        print(f"wrote {out}")
    return stats


if __name__ == "__main__":
    main()
