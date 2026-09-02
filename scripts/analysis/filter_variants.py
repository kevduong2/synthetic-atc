#!/usr/bin/env python
"""Offline band-edge variants of a rendered wav set, for the fidelity branch (D3).

Both channel modes leak +13–24 dB at 4 kHz and mode 2 runs hot at 100 Hz vs
real receivers.  Before changing the frozen channel config, test the cheap fix
offline: copy a render with a steep low-pass, and with low-pass + high-pass,
then re-measure LTAS and matched KID on the copies.  No pipeline change.

    uv run python scripts/analysis/filter_variants.py runs/prod_fid/wavs --out runs/prod_fid/variants
    # writes runs/prod_fid/variants/lp/*.wav  (8th-order Butterworth LP at 3800 Hz)
    #        runs/prod_fid/variants/lp_hp/*.wav (the same + 4th-order HP at 150 Hz)

Then: make_matched_sets.py --syn off=<wavs> --syn lp=<variants>/lp --syn lp_hp=<variants>/lp_hp
and ltas_check.py over the same directories.  Zero-phase (`sosfiltfilt`), so
speech timing is untouched and the transcripts stay valid.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def design(sr: int, lp_hz: float | None, lp_order: int,
           hp_hz: float | None, hp_order: int) -> np.ndarray | None:
    sections = []
    nyq = sr / 2.0
    if lp_hz is not None and lp_hz < nyq:
        sections.append(butter(lp_order, lp_hz / nyq, btype="low", output="sos"))
    if hp_hz is not None and hp_hz > 0:
        sections.append(butter(hp_order, hp_hz / nyq, btype="high", output="sos"))
    return np.vstack(sections) if sections else None


def filter_dir(src: Path, dst: Path, lp_hz: float | None, hp_hz: float | None,
               lp_order: int = 8, hp_order: int = 4) -> int:
    wavs = sorted(src.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no wavs under {src}")
    dst.mkdir(parents=True, exist_ok=True)
    cache: dict[int, np.ndarray | None] = {}
    for path in wavs:
        wav, sr = sf.read(path, dtype="float32")
        if sr not in cache:
            cache[sr] = design(sr, lp_hz, lp_order, hp_hz, hp_order)
        sos = cache[sr]
        out = wav if sos is None else sosfiltfilt(sos, wav, axis=0).astype(np.float32)
        sf.write(dst / path.name, np.clip(out, -1.0, 1.0), sr)
    return len(wavs)


def split_residual_cohorts(src: Path, manifest: Path, out: Path) -> tuple[Path, Path]:
    cohorts = {False: out / "off", True: out / "on"}
    for directory in cohorts.values():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    counts = {False: 0, True: 0}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        steps = row.get("gen", {}).get("channel", {}).get("steps", [])
        residual_on = any(step.get("primitive") == "residual_translate" for step in steps)
        source = src / Path(row["audio"]).name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, cohorts[residual_on] / source.name)
        counts[residual_on] += 1
    if not all(counts.values()):
        raise ValueError(f"manifest must contain residual off and on clips, got {counts}")
    print(f"off    {counts[False]} wavs -> {cohorts[False]}")
    print(f"on     {counts[True]} wavs -> {cohorts[True]}")
    return cohorts[False], cohorts[True]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wavs", help="directory of rendered wavs (the residual-on render)")
    ap.add_argument("--out", required=True, help="directory that receives lp/ and lp_hp/")
    ap.add_argument("--lp-hz", type=float, default=3800.0)
    ap.add_argument("--lp-order", type=int, default=8)
    ap.add_argument("--hp-hz", type=float, default=150.0)
    ap.add_argument("--hp-order", type=int, default=4)
    ap.add_argument("--manifest", help="render manifest; builds B3 off/on/on_lp/on_lp_hp cohorts")
    args = ap.parse_args(argv)
    src, out = Path(args.wavs), Path(args.out)
    filter_src = src
    lp_name, lp_hp_name = "lp", "lp_hp"
    if args.manifest:
        _, filter_src = split_residual_cohorts(src, Path(args.manifest), out)
        lp_name, lp_hp_name = "on_lp", "on_lp_hp"
    n = filter_dir(filter_src, out / lp_name, args.lp_hz, None, args.lp_order, args.hp_order)
    print(f"{lp_name:<6} {n} wavs -> {out / lp_name}  (LP {args.lp_hz:.0f} Hz, order {args.lp_order})")
    n = filter_dir(filter_src, out / lp_hp_name, args.lp_hz, args.hp_hz, args.lp_order, args.hp_order)
    print(f"{lp_hp_name:<6} {n} wavs -> {out / lp_hp_name}  (+ HP {args.hp_hz:.0f} Hz, order {args.hp_order})")
    print("next: make_matched_sets.py and ltas_check.py on the emitted directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
