#!/usr/bin/env python
"""Resample probe-TTS wavs in place to the channel sample rate (16 kHz).

`scripts/build_paired_views.py base` writes Kokoro-native 24 kHz audio into
`<out>/clean/`; `atcgen.channel.learned.channel_fit` refuses probes that are
not 16 kHz.  This is the runbook's resample step as a script, so it runs the
same way in PowerShell and bash:

    uv run python scripts/lab/resample_probes.py runs/gan_a_base_v1/clean runs/gan_val_base_v1/clean

Each directory's wavs are rewritten at `TARGET_SR`; a sibling `manifest.jsonl`
(one level up, as `build_paired_views` lays it out) has its `sr` field updated
so the manifest keeps describing what is on disk.  Already-16 kHz files are
left untouched, so the script is idempotent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from atcgen.channel.primitives import TARGET_SR, resample


def resample_dir(directory: Path, target_sr: int = TARGET_SR) -> dict[str, int]:
    wavs = sorted(directory.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no wavs under {directory}")
    changed = 0
    for path in wavs:
        wav, sr = sf.read(path, dtype="float32")
        if sr == target_sr:
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sf.write(path, resample(np.asarray(wav, np.float32), sr, target_sr), target_sr)
        changed += 1
    manifest = directory.parent / "manifest.jsonl"
    manifest_rows = 0
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        for row in rows:
            if row.get("sr") not in (None, target_sr):
                row["sr"] = target_sr
                manifest_rows += 1
        if manifest_rows:
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return {"wavs": len(wavs), "resampled": changed, "manifest_rows": manifest_rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="directories of wavs (e.g. runs/<probe>/clean)")
    ap.add_argument("--sr", type=int, default=TARGET_SR)
    args = ap.parse_args(argv)
    for d in args.dirs:
        stats = resample_dir(Path(d), args.sr)
        print(f"{d}: {stats['resampled']}/{stats['wavs']} wavs resampled to {args.sr} Hz"
              + (f", manifest sr fixed on {stats['manifest_rows']} rows" if stats["manifest_rows"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
