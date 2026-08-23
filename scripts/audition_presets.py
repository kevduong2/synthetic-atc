"""Render one TTS phrase through 10 fitted channel presets, for listening.

The M2.2 acceptance check that no statistic can make: do presets from different
receivers actually *sound* different?  One phrase, one voice, ten presets drawn
round-robin across stations so the set spans them, plus the clean reference and
the real clip each preset was fitted to — so a preset can be judged against its
own target rather than in the abstract.

    uv run python scripts/audition_presets.py --presets runs/calib_v2/presets.jsonl \\
        --out runs/audition_m22

Needs Kokoro (CPU/MPS is fine); the wavs land in a gitignored `runs/` directory.
`index.json` next to them lists which preset each file used.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from atcgen.channel.learned.preset import TARGET_SR, apply_preset, load_presets
from atcgen.channel.primitives import resample

PHRASE = ("delta four seven two, runway one six right, wind two one zero at one two, "
          "cleared to land")
PAD_SEC = 0.15


def _stratified(presets, count: int, rng: random.Random):
    """`count` presets, taken round-robin across stations so the set spans them."""
    by_station = defaultdict(list)
    for preset in presets:
        by_station[preset.station].append(preset)
    for items in by_station.values():
        rng.shuffle(items)

    picked, stations = [], sorted(by_station)
    while len(picked) < min(count, len(presets)):
        for station in stations:
            if by_station[station] and len(picked) < count:
                picked.append(by_station[station].pop())
    return picked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", default="runs/calib_v2/presets.jsonl")
    ap.add_argument("--out", default="runs/audition_m22")
    ap.add_argument("--corpus", default="runs/calib_v1/corpus.jsonl",
                    help="corpus manifest, to copy each preset's own real clip in")
    ap.add_argument("-n", "--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from atcgen.tts.synthesize import KokoroTTS

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tts = KokoroTTS()
    speech = resample(tts.synthesize(PHRASE, rng), tts.sample_rate, TARGET_SR)
    pad = np.zeros(int(TARGET_SR * PAD_SEC), np.float32)
    clean = np.concatenate([pad, speech, pad])
    sf.write(out / "00_clean.wav", clean, TARGET_SR)

    corpus = {}
    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        for line in corpus_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                path = Path(row["path"])
                corpus[row["clip_id"]] = (path if path.is_absolute()
                                          else corpus_path.parent / path)

    index = []
    for slot, preset in enumerate(_stratified(load_presets(args.presets), args.count,
                                              rng), start=1):
        name = f"{slot:02d}_{preset.station}_{preset.clip_id}"
        degraded = apply_preset(clean, TARGET_SR, preset)
        peak = float(np.abs(degraded).max())
        if peak > 0.99:
            degraded = degraded * (0.99 / peak)
        sf.write(out / f"{name}.wav", degraded, TARGET_SR)
        if preset.clip_id in corpus:
            real, real_sr = sf.read(corpus[preset.clip_id], dtype="float32")
            sf.write(out / f"{name}_REAL.wav", real, real_sr)
        index.append({"file": f"{name}.wav", "station": preset.station,
                      "clip_id": preset.clip_id, "snr_est": preset.snr_est,
                      "drive": preset.drive, "passband_hz": preset.passband_hz,
                      "agc_strength": preset.agc_strength,
                      "ltas_l1_db": preset.ltas_l1_db})

    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(index)} presets + clean reference to {out}")
    for row in index:
        print(f"  {row['file']:52s} snr {row['snr_est']:5.1f}  "
              f"band {row['passband_hz']}  drive {row['drive']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
