"""Build level-matched, silence-trimmed copies of the KID sets.

The reference clips carry 21.5% exact-zero samples (closed-squelch padding) and
sit ~7 dB colder than the synthetic renders.  Both are properties of the
recording, not of the channel we are trying to measure, and WavLM encodes them
happily -- so KID may be reading padding rather than timbre.

This writes a 2x2: a fixed random reference subset raw, the same subset
trimmed+normalized, and the two synthetic sets both ways.  Trim drops leading
and trailing frames below an energy threshold relative to the clip's own peak;
normalize sets a common RMS.  Same treatment on both sides, so the comparison
stays honest.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 320                 # 20 ms
TRIM_REL_DB = -35.0         # keep frames within 35 dB of the clip's peak frame
TARGET_RMS_DB = -26.0


def trim_and_norm(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = (len(x) // FRAME) * FRAME
    if n < FRAME * 5:
        return None
    fr = x[:n].reshape(-1, FRAME)
    p = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
    thr = p.max() * (10.0 ** (TRIM_REL_DB / 20.0))
    keep = np.where(p >= thr)[0]
    if keep.size == 0:
        return None
    y = fr[keep[0]:keep[-1] + 1].reshape(-1)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms <= 0:
        return None
    y = y * (10.0 ** (TARGET_RMS_DB / 20.0) / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:                     # keep it out of the clipper
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def build(src_files: list[Path], dst: Path, matched: bool) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = 0
    for p in src_files:
        x, sr = sf.read(str(p), dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            continue
        y = trim_and_norm(x) if matched else x.astype(np.float32)
        if y is None or len(y) < FRAME * 5:
            continue
        sf.write(str(dst / p.name), y, SR, subtype="PCM_16")
        n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory that receives the matched sets")
    ap.add_argument("--real-dir", default="runs/calib_kixd/clips",
                    help="real 16 kHz reference clips (default %(default)s)")
    ap.add_argument("--n-ref", type=int, default=1000,
                    help="fixed reference subset size; 1,000 reproduces the full-set KID")
    ap.add_argument("--seed", type=int, default=0, help="reference subset draw")
    ap.add_argument("--syn", action="append", default=[], metavar="TAG=DIR",
                    help="synthetic wav dir to match, e.g. --syn mode2=runs/e1_mode2_kixd/wavs")
    args = ap.parse_args(argv)
    out = Path(args.out)

    rng = random.Random(args.seed)
    reals = sorted(Path(args.real_dir).glob("*.wav"))
    if len(reals) < args.n_ref:
        sys.exit(f"{args.real_dir}: {len(reals)} clips, need {args.n_ref}")
    ref = rng.sample(reals, args.n_ref)
    sets = {
        "ref_raw": (ref, False),
        "ref_matched": (ref, True),
    }
    for item in args.syn:
        tag, sep, d = item.partition("=")
        if not sep:
            sys.exit(f"--syn expects TAG=DIR, got {item!r}")
        files = sorted(Path(d).glob("*.wav"))
        if not files:
            sys.exit(f"{d}: no wavs")
        sets[f"{tag}_raw"] = (files, False)
        sets[f"{tag}_matched"] = (files, True)

    for name, (files, matched) in sets.items():
        n = build(files, out / name, matched)
        durs, zf, rms = [], [], []
        for p in sorted((out / name).glob("*.wav"))[:400]:
            x, _ = sf.read(str(p), dtype="float64")
            durs.append(len(x) / SR)
            zf.append(float(np.mean(x == 0.0)))
            rms.append(10 * np.log10(np.mean(x ** 2) + 1e-20))
        print(f"{name:<14} n={n:<5} dur_med={np.median(durs):6.2f}s  "
              f"zerofrac={np.mean(zf):.3f}  rms_db_med={np.median(rms):7.2f}",
              flush=True)
    tags = [t.partition("=")[0] for t in args.syn]
    if tags:
        print("\nnext, matched KID per synthetic set (quote only these):")
        for t in tags:
            print(f"  uv run python -m atcgen.eval.embed_dist {out / (t + '_matched')} "
                  f"{out / 'ref_matched'} --device cuda --out {out / ('kid_' + t + '_matched.json')}")


if __name__ == "__main__":
    main()
