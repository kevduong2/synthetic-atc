"""Render one TTS phrase through each P1 primitive in isolation, for listening.

Each new primitive gets its own wav at a deliberately audible parameter setting
(the top of its configured range), so the effect can be judged on its own rather
than buried in a full chain.  A `_clean` reference and a `_full_chain` pass
through the config's chain bracket the set.

    uv run python scripts/audition_primitives.py [--config configs/mode1_wide.yaml]
                                                 [--out runs/audition_p1]

Needs Kokoro (CPU/MPS is fine); the wavs land in a gitignored `runs/` directory.
"""

import argparse
import inspect
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from atcgen.channel import primitives as P
from atcgen.channel.chain import PAD_SEC, ProceduralChannel
from atcgen.channel.primitives import TARGET_SR, resample
from atcgen.config import load_config
from atcgen.tts.synthesize import KokoroTTS

PHRASE = ("delta four seven two, runway one six right, wind two one zero at one two, "
          "cleared to land")

# name -> params at the loud end of 03 §3's ranges: an audition wants the effect
# obvious, not typical.  `pad` is filled in per call.
AUDITIONS = {
    "mic_coloration_bright": dict(tilt_db=4.0, peaks=2, peak_gain_db=6.0),
    "mic_coloration_dark": dict(tilt_db=-4.0, peaks=2, peak_gain_db=6.0),
    "ptt_truncation": dict(head_ms=120.0, tail_ms=120.0),
    "resample_chain": dict(narrow_sr=4000, alias=True),
    "fading": dict(rate_hz=0.5, depth_db=6.0),
    "agc_attack": dict(attack_ms=200.0, surge_db=8.0),
    "squelch_gate": dict(floor_db=-60.0, attack_ms=5.0, release_ms=50.0,
                         threshold_db=-20.0, tail_burst_prob=1.0, tail_burst_amp=0.4),
    "codec_16kbps": dict(bitrate_kbps=16),
    "codec_23kbps": dict(bitrate_kbps=23),
    "codec_32kbps": dict(bitrate_kbps=32),
    "codec_64kbps": dict(bitrate_kbps=64),
}
PRIMITIVE_OF = {
    "mic_coloration_bright": "mic_coloration", "mic_coloration_dark": "mic_coloration",
    "codec_16kbps": "codec_roundtrip", "codec_23kbps": "codec_roundtrip",
    "codec_32kbps": "codec_roundtrip", "codec_64kbps": "codec_roundtrip",
}


def _speech(rng: random.Random) -> np.ndarray:
    """One rendered phrase at 16 kHz, framed by the chain's usual padding."""
    tts = KokoroTTS()
    wav = resample(tts.synthesize(PHRASE, rng), tts.sample_rate, TARGET_SR)
    pad = np.zeros(int(TARGET_SR * PAD_SEC), np.float32)
    return np.concatenate([pad, wav, pad])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mode1_wide.yaml")
    ap.add_argument("--out", default="runs/audition_p1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pad = int(TARGET_SR * PAD_SEC)
    x = _speech(random.Random(args.seed))
    sf.write(out / "00_clean.wav", x, TARGET_SR)

    # A noise bed first, so the gate has something to gate and the codecs have
    # something to mangle -- these effects are inaudible on clean speech.
    noisy = P.additive_noise(x, TARGET_SR, random.Random(args.seed), snr_db=15.0,
                             color="pink", pad=pad)
    sf.write(out / "01_noise_15db.wav", noisy, TARGET_SR)

    rows = []
    for index, (label, params) in enumerate(AUDITIONS.items(), start=2):
        fn = P.PRIMITIVES[PRIMITIVE_OF.get(label, label)]
        # the gate, the AGC surge and the codecs are all inaudible on clean speech
        needs_noise = label.startswith("codec") or label in {"squelch_gate", "agc_attack"}
        source = noisy if needs_noise else x
        kwargs = dict(params)
        if "pad" in inspect.signature(fn).parameters:
            kwargs["pad"] = pad
        y = fn(source, TARGET_SR, random.Random(args.seed), **kwargs)
        path = out / f"{index:02d}_{label}.wav"
        sf.write(path, np.clip(y, -1.0, 1.0), TARGET_SR)
        rows.append((path.name, y))

    config = load_config(args.config)
    chain = ProceduralChannel.from_config(config.channel)
    raw = x[pad:len(x) - pad]
    for hops in (1, 2):
        full, record = chain(raw, TARGET_SR, random.Random(args.seed), hops=hops)
        path = out / f"90_full_chain_{hops}hop.wav"
        sf.write(path, full, TARGET_SR)
        rows.append((path.name, full))
        print(f"{path.name}: {', '.join(record.applied())}")

    print(f"\n{len(rows) + 2} wavs in {out}")
    for name, wav in [("00_clean.wav", x), ("01_noise_15db.wav", noisy), *rows]:
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        print(f"  {name:34s} {len(wav) / TARGET_SR:5.2f}s  "
              f"rms={20 * np.log10(max(rms, 1e-9)):7.2f} dB  peak={np.abs(wav).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
